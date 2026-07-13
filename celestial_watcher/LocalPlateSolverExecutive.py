import subprocess, threading, tempfile, warnings, shutil, time, os, cv2

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter("ignore", FITSFixedWarning)

POLL_INTERVAL_S = 0.5
SOLVE_INTERVAL_S = 5.0
SCALE_LOW_DEG = 15
SCALE_HIGH_DEG = 18
ASTROMETRY_CFG = os.path.join(os.path.dirname(__file__), "astrometry.cfg")
BLIND_SOLVE_TIMEOUT_S = 300
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
        self.latest_wcs = None  # (timestamp, astropy.wcs.WCS)

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

        thread.join(timeout=BLIND_SOLVE_TIMEOUT_S + 5.0)

        with self._lock:
            self._solver_thread = None

        print("Plate Solver stopped.")
        return "Plate Solver stopped."

    def get_latest_wcs(self):
        with self._lock:
            return self.latest_wcs

    def _run_solver_logic(self):
        while not self._stop_event.is_set():
            result = self.camera_manager.get_latest_frame_with_time()

            if result is None:
                self._stop_event.wait(POLL_INTERVAL_S)
                continue

            frame, capture_time = result

            if self._stop_event.is_set():
                break
            print("Plate solving frame...")
            wcs = self._solve_frame(frame)

            if wcs is None:
                print("Frame did not solve.")
            else:
                with self._lock:
                    self.latest_wcs = (capture_time, wcs)
                print(f"Solved. Center RA/Dec = "
                      f"{wcs.wcs.crval[0]:.4f}, {wcs.wcs.crval[1]:.4f}")

            self._stop_event.wait(SOLVE_INTERVAL_S)

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
                previous = self.latest_wcs

            if previous is not None:
                _, prev_wcs = previous
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

            return WCS(fits.getheader(wcs_path))