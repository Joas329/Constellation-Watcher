import subprocess, threading, tempfile, warnings, shutil, time, os, cv2

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

POLL_INTERVAL_S = 0.5
SCALE_LOW_DEG = 15
SCALE_HIGH_DEG = 18
ASTROMETRY_CFG = os.path.join(os.path.dirname(__file__), "astrometry.cfg")
BLIND_SOLVE_TIMEOUT_S = 60
HINTED_SOLVE_TIMEOUT_S = 30

class LocalPlateSolverExecutive:
    def __init__(self, camera_manager):
        if shutil.which("solve-field") is None:
            raise RuntimeError("solve-field not found on PATH. Install astrometry.net.")
        if not os.path.isfile(ASTROMETRY_CFG):
            raise RuntimeError(f"Astrometry config not found: {ASTROMETRY_CFG}")

        self.camera_manager = camera_manager
        self._solver_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_attempted_time = None
        self.latest_solution = None  # (capture_time, wcs, frame, star_matches)

    def start(self):
        with self._lock:
            if self._solver_thread is not None and self._solver_thread.is_alive():
                return "Plate Solver already running."

            self._stop_event.clear()

            self._solver_thread = threading.Thread(target=self._run_solver_logic, daemon=False)
            self._solver_thread.start()

        print("----Plate Solver started---------------")
        return "Plate Solver started."

    def stop(self):
        with self._lock:
            if self._solver_thread is None:
                return "Plate Solver already stopped."

            self._stop_event.set()
            thread = self._solver_thread

        thread.join(timeout=10.0)

        with self._lock:
            self._solver_thread = None

        print("Plate Solver stopped.")
        return "Plate Solver stopped."

    def get_latest_solution(self):
        with self._lock:
            return self.latest_solution

    # Back-compat accessors (older consumers, e.g. server endpoint)

    def get_latest_wcs(self):
        with self._lock:
            if self.latest_solution is None:
                return None
            capture_time, wcs, _, _ = self.latest_solution
            return (capture_time, wcs)

    def get_latest_solved_frame(self):
        with self._lock:
            if self.latest_solution is None:
                return None
            capture_time, _, frame, matches = self.latest_solution
            return (capture_time, frame, matches)

    def _run_solver_logic(self):
        while not self._stop_event.is_set():
            result = self.camera_manager.get_latest_frame_with_time()

            if result is None:
                self._stop_event.wait(POLL_INTERVAL_S)
                continue

            frame, capture_time = result

            if capture_time is not None and capture_time == self._last_attempted_time:
                self._stop_event.wait(POLL_INTERVAL_S)
                continue

            if self._stop_event.is_set():
                break
            print("Plate solving frame...")
            solved = self._solve_frame(frame)
            self._last_attempted_time = capture_time

            if solved is None:
                print("Frame did not solve.")
            else:
                wcs, star_matches = solved
                with self._lock:
                    self.latest_solution = (capture_time, wcs, frame, star_matches)
                print(f"Solved. Center RA/Dec = {wcs.wcs.crval[0]:.4f}, {wcs.wcs.crval[1]:.4f} | {len(star_matches)} reference stars")

        print("Plate Solver thread stopped.")

    def _solve_frame(self, frame):
        with tempfile.TemporaryDirectory(prefix="platesolve_") as tmp_dir:
            image_path = os.path.join(tmp_dir, "frame.png")
            if not cv2.imwrite(image_path, frame):
                raise RuntimeError("Failed to write frame to disk for solving.")

            cmd = [
                "solve-field", image_path,
                "--config", ASTROMETRY_CFG,
                "--scale-units", "degwidth",
                "--scale-low", str(SCALE_LOW_DEG),
                "--scale-high", str(SCALE_HIGH_DEG),
                "--parity", "neg",
                "--downsample", "4",
                "--no-plots",
                "--no-remove-lines",
                "--uniformize", "0",
                "--overwrite",
                "--dir", tmp_dir,
            ]

            with self._lock:
                previous = self.latest_solution

            if previous is not None:
                _, prev_wcs, _, _ = previous
                cmd += ["--ra", str(prev_wcs.wcs.crval[0]),
                        "--dec", str(prev_wcs.wcs.crval[1]),
                        "--radius", "20"]
                timeout = HINTED_SOLVE_TIMEOUT_S
            else:
                timeout = BLIND_SOLVE_TIMEOUT_S

            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if self._stop_event.is_set():
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