import subprocess, tempfile, warnings, shutil, time, os, cv2

from astropy.io import fits
from transport.stage import Stage
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

SCALE_LOW_DEG = 15
SCALE_HIGH_DEG = 18
ASTROMETRY_CFG = os.path.join(os.path.dirname(__file__), "astrometry.cfg")
BLIND_SOLVE_TIMEOUT_S = 60
HINTED_SOLVE_TIMEOUT_S = 30

class PlateSolverStage(Stage):
    def __init__(self, source, sink):
        if shutil.which("solve-field") is None:
            raise RuntimeError("solve-field not found on PATH. Install astrometry.net.")
        if not os.path.isfile(ASTROMETRY_CFG):
            raise RuntimeError(f"Astrometry config not found: {ASTROMETRY_CFG}")

        super().__init__("plate_solver", source, sink)
        # last successful WCS, kept so the next solve can hint solve-field with a
        # search box instead of going blind. Single thread writes it, no lock.
        self._prev_wcs = None

    def process(self, frame, capture_time):
        print("Plate solving frame...")
        solved = self._solve_frame(frame)
        if solved is None:
            print("Frame did not solve.")
            return None

        wcs, star_matches = solved
        self._prev_wcs = wcs
        print(f"Solved. Center RA/Dec = {wcs.wcs.crval[0]:.4f}, {wcs.wcs.crval[1]:.4f} | {len(star_matches)} reference stars")
        # payload: everything a downstream consumer needs, atomic by construction
        return (wcs, frame, star_matches)

    def _solve_frame(self, frame):
        with tempfile.TemporaryDirectory(prefix="platesolve_") as tmp_dir:
            image_path = os.path.join(tmp_dir, "frame.png")
            if not cv2.imwrite(image_path, frame):
                raise RuntimeError("Failed to write frame to disk for solving.")

            cmd = [
                "solve-field", image_path,
                "--config", ASTROMETRY_CFG,
                "--scale-units", "degwidth",
                "--parity", "neg",
                "--downsample", "4",
                "--no-plots",
                "--no-remove-lines",
                "--uniformize", "0",
                "--overwrite",
                "--dir", tmp_dir,
            ]

            if self._prev_wcs is not None:
                cmd += ["--ra", str(self._prev_wcs.wcs.crval[0]), "--dec", str(self._prev_wcs.wcs.crval[1]), "--radius", "20"]
                timeout = HINTED_SOLVE_TIMEOUT_S
            else:
                timeout = BLIND_SOLVE_TIMEOUT_S

            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if self._stop.is_set(): # Stage's stop event, kill the solve
                    proc.kill()
                    proc.wait()
                    return None
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    print(f"solve-field timed out ({timeout}s).")
                    return None
                time.sleep(0.25)

            wcs_path = os.path.join(tmp_dir, "frame.wcs")
            if not os.path.isfile(wcs_path):
                return None

            wcs = WCS(fits.getheader(wcs_path))

            matches = []
            corr_path = os.path.join(tmp_dir, "frame.corr")
            if os.path.isfile(corr_path):
                with fits.open(corr_path) as hdul:
                    d = hdul[1].data
                    matches = list(zip(map(float, d["field_x"]), map(float, d["field_y"]), map(float, d["index_x"]), map(float, d["index_y"])))

            return wcs, matches