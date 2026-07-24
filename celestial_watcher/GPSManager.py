import time
import serial
import threading
from datetime import datetime, timezone

class GPSManager:
    # Reads NMEA from a u-blox 7 USB dongle (G28U7FUSB) on a serial port and
    # keeps the latest fix. Runs its own thread like the camera; readers call
    # get_fix() for the most recent position, non-blocking.
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self._serial = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # latest fix, written together under _lock so a reader never mixes a new
        # lat with a stale lon
        self._lat = None
        self._lon = None
        self._alt_m = None
        self._n_sats = None
        self._fix_quality = 0 # 0 = no fix
        self._updated_monotonic = None

        # time sync (from RMC, which carries UTC date + time; GGA has time only)
        self._gps_utc = None # UTC datetime of the last valid fix
        self._host_offset_s = None # host_clock - gps_clock, seconds

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            print("GPS already running.")
            return

        # open here, not in __init__, so construction can't fail on an unplugged dongle; failure to open is an explicit raise the caller can handle
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
        except serial.SerialException as e:
            raise RuntimeError(f"Could not open GPS on {self.port}: {e}") from e

        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="gps", daemon=False)
        self._thread.start()
        print(f"GPS started on {self.port} @ {self.baud} baud.")

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._serial.readline()
            except serial.SerialException as e:
                if not self._stop.is_set():
                    print(f"GPS read error: {e}")
                break

            if not raw:
                continue # timeout, no line this cycle; re-check stop

            line = raw.decode("ascii", errors="replace").strip()
            if not self._checksum_ok(line):
                continue

            # the talker prefix varies (GP for GPS, GN for multi-constellation),
            # so match on the sentence type. GGA -> position, RMC -> date+time.
            kind = line[3:6]
            if kind == "GGA":
                self._parse_gga(line)
            elif kind == "RMC":
                self._parse_rmc(line)

        print("GPS thread exited.")

    def _parse_gga(self, line):
        f = line.split(",")
        if len(f) < 10:
            return

        try:
            quality = int(f[6]) if f[6] else 0
        except ValueError:
            return

        if quality == 0:
            # no fix: record that we lost it, keep last known position untouched
            with self._lock:
                self._fix_quality = 0
            return

        lat = self._nmea_to_deg(f[2], f[3])
        lon = self._nmea_to_deg(f[4], f[5])
        if lat is None or lon is None:
            return

        alt = float(f[9]) if f[9] else None
        n_sats = int(f[7]) if f[7] else None

        with self._lock:
            self._lat = lat
            self._lon = lon
            self._alt_m = alt
            self._n_sats = n_sats
            self._fix_quality = quality
            self._updated_monotonic = time.monotonic()

    def _parse_rmc(self, line):
        # RMC carries UTC date AND time (GGA has time only). Field 2 is the
        # A/V validity flag, field 1 is hhmmss.ss, field 9 is ddmmyy.
        f = line.split(",")
        if len(f) < 10 or f[2] != "A":     # A = valid fix; V = no fix, skip
            return

        t, d = f[1], f[9]
        if not t or not d:
            return

        try:
            gps_utc = datetime.strptime(
                d + t.split(".")[0], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return

        host_utc = datetime.now(timezone.utc)
        with self._lock:
            self._gps_utc = gps_utc
            # positive => host clock is ahead of GPS. Includes ~tens of ms of
            # serial/USB latency, so treat sub-100ms as noise; seconds means the
            # host clock is genuinely wrong and every capture timestamp is off.
            self._host_offset_s = (host_utc - gps_utc).total_seconds()

    def get_fix(self):
        # returns the latest fix as a dict, or None if we have never had one.
        # 'age_s' lets a caller reject a stale fix (e.g. antenna lost sky).
        with self._lock:
            if self._lat is None:
                return None
            age = (time.monotonic() - self._updated_monotonic
                   if self._updated_monotonic else None)
            return {
                "lat": self._lat,
                "lon": self._lon,
                "alt_m": self._alt_m,
                "n_sats": self._n_sats,
                "fix_quality": self._fix_quality,
                "age_s": age,
            }

    def get_latlon_elev(self, max_age_s: float = 10.0):
        # convenience for the RSO tracker: returns (lat, lon, elev) or None if
        # there's no fix or the fix is older than max_age_s
        fix = self.get_fix()
        if fix is None or fix["fix_quality"] == 0:
            return None
        if fix["age_s"] is not None and fix["age_s"] > max_age_s:
            return None
        return (fix["lat"], fix["lon"], fix["alt_m"] if fix["alt_m"] is not None else 0.0)

    def get_time_status(self):
        # GPS UTC and the host-vs-GPS offset, or None if no valid RMC seen yet.
        # This is a clock-correctness health check, NOT a capture-timestamp
        # source: GPS-over-USB lands tens of ms late, worse than the camera's
        # device-clock sync. Use it to catch a host clock that is wrong in
        # absolute terms (offset of seconds+), which the camera sync cannot see.
        with self._lock:
            if self._gps_utc is None:
                return None
            return {
                "gps_utc": self._gps_utc.isoformat(),
                "host_offset_s": self._host_offset_s,
            }

    @staticmethod
    def _nmea_to_deg(value, hemi):
        # NMEA packs lat as ddmm.mmmm, lon as dddmm.mmmm. Integer part of
        # (value/100) is degrees; the remainder is minutes, /60 to decimal.
        if not value:
            return None
        try:
            v = float(value)
        except ValueError:
            return None
        deg = int(v / 100)
        minutes = v - deg * 100
        result = deg + minutes / 60.0
        if hemi in ("S", "W"):
            result = -result
        return result

    @staticmethod
    def _checksum_ok(sentence):
        # XOR of everything between '$' and '*' must equal the trailing hex byte
        if not sentence.startswith("$") or "*" not in sentence:
            return False
        body, _, cksum = sentence[1:].partition("*")
        calc = 0
        for ch in body:
            calc ^= ord(ch)
        try:
            return calc == int(cksum[:2], 16)
        except ValueError:
            return False

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._thread = None
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        print("GPS stopped.")

    def close(self):
        self.stop()