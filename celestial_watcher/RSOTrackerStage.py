import os
import numpy as np

from datetime import timedelta
from transport.stage import Stage
from skyfield.api import load, wgs84

VELOCITY_BASELINE_S = 1.0

# Esrange Space Center, Kiruna: TODO: get these from a GPS tracker
OBSERVER_LAT_DEG = 67.89
OBSERVER_LON_DEG = 21.10
OBSERVER_ELEV_M = 330

FRAME_W = 4096
FRAME_H = 3000
FRAME_RADIUS_DEG = 10.0  # half-diagonal of a ~16x12 deg field, rounded up
MIN_ALTITUDE_DEG = 5.0

TLE_FILE = os.path.join(os.path.dirname(__file__), "tles.txt")
EPHEMERIS_FILE = "de421.bsp"

# Arequipa, Peru (temporary — TODO: get these from a GPS tracker)
OBSERVER_LAT_DEG = -16.397107
OBSERVER_LON_DEG = -71.558758
OBSERVER_ELEV_M = 2335

class RSOTrackerStage(Stage):
    def __init__(self, source, sink):
        if not os.path.isfile(TLE_FILE):
            raise RuntimeError(
                f"TLE file not found: {TLE_FILE}. Download TLEs matching the "
                f"observation epoch and save them there.")
        super().__init__("rso_tracker", source, sink)

        self._ts = load.timescale()
        self._eph = load(EPHEMERIS_FILE)
        self._observer = wgs84.latlon(OBSERVER_LAT_DEG, OBSERVER_LON_DEG, OBSERVER_ELEV_M)

        self._satellites = load.tle_file(TLE_FILE)
        if not self._satellites:
            raise RuntimeError(f"No satellites parsed from TLE file: {TLE_FILE}")
        print(f"RSOTracker loaded {len(self._satellites)} satellites from {TLE_FILE}.")

    def process(self, payload, capture_time):
        wcs, frame, star_matches = payload

        hits = self._satellites_in_frame(wcs, capture_time)

        if hits:
            names = ", ".join(f"{name} ({x:.0f},{y:.0f})" for name, _, x, y, _, _ in hits)
            print(f"RSOs in frame at {capture_time.isoformat()}: {names}")
        else:
            print(f"No RSOs in frame at {capture_time.isoformat()}.")

        # carry the frame and stars through so the overlay endpoint has everything
        # for this capture_time in one atomic payload
        return (hits, frame, star_matches)

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

            cos_sep = (np.sin(np.radians(dec_deg)) * sin_dec_c + np.cos(np.radians(dec_deg)) * cos_dec_c * np.cos(np.radians(ra_deg - center_ra)))
            if np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) > FRAME_RADIUS_DEG:
                continue

            x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if not (0 <= x < FRAME_W and 0 <= y < FRAME_H):
                continue

            if not sat.at(t).is_sunlit(self._eph):
                continue

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