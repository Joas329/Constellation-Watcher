import os
import numpy as np

from datetime import timedelta
from transport.stage import Stage
from skyfield.api import load, wgs84

VELOCITY_BASELINE_S = 1.0

FRAME_W = 4096
FRAME_H = 3000
FRAME_RADIUS_DEG = 10.0  # half-diagonal of a ~16x12 deg field, rounded up
MIN_ALTITUDE_DEG = 5.0

TLE_FILE = os.path.join(os.path.dirname(__file__), "tles.txt")
EPHEMERIS_FILE = "de421.bsp"

# Named observer sites for replay mode (frames from a known location + epoch).
# Live mode ignores these and reads the position from the GPS each frame.
ESRANGE_KIRUNA = (67.89, 21.10, 330)         # the October replay dataset was shot here
AREQUIPA_PERU  = (-16.397107, -71.558758, 2335)

class RSOTrackerStage(Stage):
    def __init__(self, source, sink, observer=None, gps=None, gps_max_age_s=10.0):
        # exactly one location source:
        #   observer=(lat,lon,elev)  -> fixed site, for replaying a known dataset
        #   gps=<GPSManager>         -> live position, re-read per frame
        # a fixed observer is resolved once; a GPS observer is resolved each frame
        # so a moving platform (or a fix that arrives late) is always current.
        if (observer is None) == (gps is None):
            raise RuntimeError("Provide exactly one of observer=(lat,lon,elev) or gps=GPSManager.")

        if not os.path.isfile(TLE_FILE):
            raise RuntimeError(
                f"TLE file not found: {TLE_FILE}. Download TLEs matching the "
                f"observation epoch and save them there.")
        super().__init__("rso_tracker", source, sink)

        self._ts = load.timescale()
        self._eph = load(EPHEMERIS_FILE)

        self._gps = gps
        self._gps_max_age_s = gps_max_age_s
        # fixed-site observer is built once; live-GPS observer is built per frame
        self._fixed_observer = wgs84.latlon(*observer) if observer is not None else None
        self._last_gps_latlon = None   # cache so we don't rebuild the topos every frame if unchanged

        self._satellites = load.tle_file(TLE_FILE)
        if not self._satellites:
            raise RuntimeError(f"No satellites parsed from TLE file: {TLE_FILE}")
        mode = "fixed site" if observer is not None else "live GPS"
        print(f"RSOTracker loaded {len(self._satellites)} satellites ({mode}).")

    def _current_observer(self):
        # fixed mode: the site never moves, return the prebuilt observer
        if self._fixed_observer is not None:
            return self._fixed_observer

        # live mode: pull the latest GPS fix. If there's no usable fix, we can't
        # correlate this frame — return None and let process() skip it.
        latlon = self._gps.get_latlon_elev(max_age_s=self._gps_max_age_s)
        if latlon is None:
            return None

        # rebuild the Skyfield observer only when the position actually changed,
        # so a stationary mount doesn't reconstruct it 2x/second for nothing
        if latlon != self._last_gps_latlon:
            self._last_gps_latlon = latlon
            self._gps_observer = wgs84.latlon(*latlon)
        return self._gps_observer

    def process(self, payload, capture_time):
        wcs, frame, star_matches = payload

        observer = self._current_observer()
        if observer is None:
            # live mode with no GPS fix yet: pass the frame through with no hits
            # rather than correlating against a wrong/last-known location
            print(f"No GPS fix; skipping RSO correlation at {capture_time.isoformat()}.")
            return ([], frame, star_matches)

        hits = self._satellites_in_frame(wcs, capture_time, observer)

        if hits:
            names = ", ".join(f"{name} ({x:.0f},{y:.0f})" for name, _, x, y, _, _ in hits)
            print(f"RSOs in frame at {capture_time.isoformat()}: {names}")
        else:
            print(f"No RSOs in frame at {capture_time.isoformat()}.")

        return (hits, frame, star_matches)

    def _satellites_in_frame(self, wcs, capture_time_utc, observer):
        t = self._ts.from_datetime(capture_time_utc)
        t_later = self._ts.from_datetime(
            capture_time_utc + timedelta(seconds=VELOCITY_BASELINE_S))
        center_ra, center_dec = wcs.wcs.crval

        sin_dec_c = np.sin(np.radians(center_dec))
        cos_dec_c = np.cos(np.radians(center_dec))

        hits = []
        for sat in self._satellites:
            topocentric = (sat - observer).at(t)

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

            topo_later = (sat - observer).at(t_later)
            ra2, dec2, _ = topo_later.radec()
            x2, y2 = wcs.world_to_pixel_values(ra2._degrees, dec2.degrees)

            if np.isfinite(x2) and np.isfinite(y2):
                vx = (float(x2) - float(x)) / VELOCITY_BASELINE_S
                vy = (float(y2) - float(y)) / VELOCITY_BASELINE_S
            else:
                vx = vy = 0.0

            hits.append((sat.name, sat.model.satnum, float(x), float(y), vx, vy))

        return hits

    def _current_observer_or_none(self):
        return self._current_observer()