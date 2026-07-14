import os
import threading

import numpy as np
from datetime import timedelta
from skyfield.api import load, wgs84

POLL_INTERVAL_S = 1.0
VELOCITY_BASELINE_S = 1.0

# Esrange Space Center, Kiruna: TODO: We need to get these from a gps tracker
OBSERVER_LAT_DEG = 67.89
OBSERVER_LON_DEG = 21.10
OBSERVER_ELEV_M = 330

FRAME_W = 4096
FRAME_H = 3000
FRAME_RADIUS_DEG = 10.0  # half-diagonal of a ~16x12 deg field, rounded up
MIN_ALTITUDE_DEG = 5.0

TLE_FILE = os.path.join(os.path.dirname(__file__), "tles.txt")
EPHEMERIS_FILE = "de421.bsp"


class RSOTracker:
    def __init__(self, plate_solver):
        if not os.path.isfile(TLE_FILE):
            raise RuntimeError(
                f"TLE file not found: {TLE_FILE}. Download TLEs matching the "
                f"observation epoch and save them there.")

        self.plate_solver = plate_solver

        self._ts = load.timescale()
        self._eph = load(EPHEMERIS_FILE)
        self._observer = wgs84.latlon(OBSERVER_LAT_DEG, OBSERVER_LON_DEG, OBSERVER_ELEV_M)

        self._satellites = load.tle_file(TLE_FILE)
        if not self._satellites:
            raise RuntimeError(f"No satellites parsed from TLE file: {TLE_FILE}")
        print(f"RSOTracker loaded {len(self._satellites)} satellites from {TLE_FILE}.")

        self._tracker_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_matched_time = None
        self.latest_hits = None  # (capture_time, hits, frame, star_matches)

    def start(self):
        with self._lock:
            if self._tracker_thread is not None and self._tracker_thread.is_alive():
                return "RSO Tracker already running."

            self._stop_event.clear()

            self._tracker_thread = threading.Thread(target=self._run_tracker_logic, daemon=False)
            self._tracker_thread.start()

        print("RSO Tracker started.")
        return "RSO Tracker started."

    def stop(self):
        with self._lock:
            if self._tracker_thread is None:
                return "RSO Tracker already stopped."

            self._stop_event.set()
            thread = self._tracker_thread

        thread.join(timeout=10.0)

        with self._lock:
            self._tracker_thread = None

        print("RSO Tracker stopped.")
        return "RSO Tracker stopped."

    def get_latest_hits(self):
        with self._lock:
            return self.latest_hits

    def _run_tracker_logic(self):
        while not self._stop_event.is_set():
            result = self.plate_solver.get_latest_wcs()

            if result is None:
                self._stop_event.wait(POLL_INTERVAL_S)
                continue

            capture_time, wcs = result

            if capture_time is None:
                # Frame had no parseable timestamp; cannot correlate.
                self._stop_event.wait(POLL_INTERVAL_S)
                continue

            if capture_time == self._last_matched_time:
                self._stop_event.wait(POLL_INTERVAL_S)
                continue

            # Grab the frame belonging to this WCS *before* the slow matching,
            # to minimize the window in which the solver replaces it.
            frame_for_hits = None
            star_matches = []
            solved = self.plate_solver.get_latest_solved_frame()
            if solved is not None and solved[0] == capture_time:
                _, frame_for_hits, star_matches = solved

            hits = self._satellites_in_frame(wcs, capture_time)
            self._last_matched_time = capture_time

            with self._lock:
                self.latest_hits = (capture_time, hits, frame_for_hits, star_matches)

            if hits:
                names = ", ".join(f"{name} ({x:.0f},{y:.0f})" for name, _, x, y, _, _ in hits)
                print(f"RSOs in frame at {capture_time.isoformat()}: {names}")
            else:
                print(f"No RSOs in frame at {capture_time.isoformat()}.")

        print("RSO Tracker thread stopped.")

    def _satellites_in_frame(self, wcs, capture_time_utc):
        t = self._ts.from_datetime(capture_time_utc)
        t_later = self._ts.from_datetime(
            capture_time_utc + timedelta(seconds=VELOCITY_BASELINE_S))
        center_ra, center_dec = wcs.wcs.crval

        sin_dec_c = np.sin(np.radians(center_dec))
        cos_dec_c = np.cos(np.radians(center_dec))

        hits = []
        for sat in self._satellites:
            topocentric = (sat - self._observer).at(t)

            alt, _, _ = topocentric.altaz()
            if alt.degrees < MIN_ALTITUDE_DEG:
                continue

            ra, dec, _ = topocentric.radec()
            ra_deg, dec_deg = ra._degrees, dec.degrees

            cos_sep = (np.sin(np.radians(dec_deg)) * sin_dec_c
                       + np.cos(np.radians(dec_deg)) * cos_dec_c
                       * np.cos(np.radians(ra_deg - center_ra)))
            if np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) > FRAME_RADIUS_DEG:
                continue

            x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if not (0 <= x < FRAME_W and 0 <= y < FRAME_H):
                continue

            if not sat.at(t).is_sunlit(self._eph):
                continue

            # Pixel velocity from a short position baseline
            topo_later = (sat - self._observer).at(t_later)
            ra2, dec2, _ = topo_later.radec()
            x2, y2 = wcs.world_to_pixel_values(ra2._degrees, dec2.degrees)
            if np.isfinite(x2) and np.isfinite(y2):
                vx = (float(x2) - float(x)) / VELOCITY_BASELINE_S
                vy = (float(y2) - float(y)) / VELOCITY_BASELINE_S
            else:
                vx = vy = 0.0

            hits.append((sat.name, sat.model.satnum, float(x), float(y), vx, vy))

        return hits