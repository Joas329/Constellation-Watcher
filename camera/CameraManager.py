import threading
import time
from typing import Any
from datetime import datetime, timedelta, timezone

from pypylon import pylon, genicam


class CameraManager:
    def __init__(self, grayscale: bool = False, bit_depth: int = 8,
                 require_device_time: bool = False, resync_interval: float = 60.0):
        print("Initializing Camera Manager")

        self.grayscale = grayscale
        self.bit_depth = bit_depth
        self.require_device_time = require_device_time
        self.resync_interval = resync_interval

        self.camera = None
        self.converter = None
        self.active_device = None

        self.tl_factory = pylon.TlFactory.GetInstance()
        self.basler_devices = self.detect_basler_devices()

        # timing state, resolved once in start_acquisition so the grab loop
        # stays branch-free and the chosen source is visible in the log
        self.time_source = None
        self.clock_synced = False
        self.clock_uncertainty = None
        self._tick_hz = None
        self._clock_offset = None
        self._exposure_s = 0.0
        self._chunk_exposure = False
        self._last_sync = 0.0

        # frame and capture time are written together under frame_lock so
        # get_latest_frame_with_time() can never mix a frame with a stale time
        self.last_frame = None
        self.last_capture_time = None
        self.frame_lock = threading.Lock()

        self.acquisition_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

        if not self.basler_devices:
            print("No Basler cameras detected.")
        else:
            print(f"Detected {len(self.basler_devices)} Basler camera(s).")

    def detect_basler_devices(self) -> list[dict[str, Any]]:
        devices = []

        for d in self.tl_factory.EnumerateDevices():
            devices.append({
                "device_info": d,
                "name": d.GetFriendlyName(),
                "model": d.GetModelName(),
                "serial": d.GetSerialNumber(),
                "class": d.GetDeviceClass(),
            })

        return devices

    def _latch_device_ticks(self):
        # Get the current camera ticks from the camera interface itself
        try:
            self.camera.TimestampLatch.Execute()
            return self.camera.TimestampLatchValue.Value
        except (genicam.GenericException, AttributeError):
            pass

        try:
            self.camera.GevTimestampControlLatch.Execute()
            return self.camera.GevTimestampValue.Value
        except (genicam.GenericException, AttributeError):
            return None

    def _get_tick_hz(self) -> float:
        try:
            return float(self.camera.GevTimestampTickFrequency.Value)
        except (genicam.GenericException, AttributeError):
            # USB3 Vision and GigE Vision 2.0 report nanoseconds
            return 1e9

    def _get_exposure_seconds(self) -> float:
        for name in ("ExposureTime", "ExposureTimeAbs"):
            try:
                return float(getattr(self.camera, name).Value) * 1e-6
            except (genicam.GenericException, AttributeError):
                continue
        return 0.0

    def _set_pixel_format(self):
        if not self.grayscale or self.bit_depth <= 8:
            return

        for fmt in ("Mono12", "Mono12p"):
            try:
                self.camera.PixelFormat.Value = fmt
                print(f"Sensor pixel format: {fmt}")
                return
            except (genicam.GenericException, AttributeError):
                continue

        raise RuntimeError("Camera supports no 12-bit mono pixel format.")

    def set_gain_db(self, gain_db: float) -> float:
        if self.camera is None or not self.camera.IsOpen():
            raise RuntimeError("Camera is not open. Press Start Viewfinder before setting gain.")

        try:
            self.camera.GainAuto.Value = "Off"
        except (genicam.GenericException, AttributeError):
            pass

        try:
            node = self.camera.Gain
        except (genicam.GenericException, AttributeError):
            raise RuntimeError("Camera has no Gain node (SFNC 1.x uses GainRaw).")

        try:
            low, high = float(node.Min), float(node.Max)
        except (genicam.GenericException, AttributeError) as e:
            raise RuntimeError(f"Could not read gain limits: {e}") from e

        if not low <= gain_db <= high:
            raise ValueError(f"Gain {gain_db} dB outside range [{low}, {high}] dB")

        try:
            node.Value = gain_db
            applied = float(node.Value)
        except genicam.GenericException as e:
            raise RuntimeError(f"Camera rejected gain: {type(e).__name__}: {e}") from e

        print(f"Gain set: {applied:.2f} dB")
        return applied

    def set_black_level(self, level: float) -> float:
        if self.camera is None or not self.camera.IsOpen():
            raise RuntimeError("Camera is not open.")

        for name in ("BlackLevel", "BlackLevelRaw"):
            try:
                node = getattr(self.camera, name)
            except (genicam.GenericException, AttributeError):
                continue

            try:
                low, high = float(node.Min), float(node.Max)
                if not low <= level <= high:
                    raise ValueError(f"Black level {level} outside [{low}, {high}]")
                node.Value = level
                applied = float(node.Value)
            except genicam.GenericException as e:
                raise RuntimeError(f"Camera rejected black level: {e}") from e

            print(f"Black level set: {applied}")
            return applied

        raise RuntimeError("Camera has no BlackLevel node.")

    def read_analog_settings(self) -> dict:
        # reported alongside pixel stats so a histogram is never ambiguous about
        # which camera settings produced it
        out = {}
        for key, names in (("gain_db", ("Gain",)),
                           ("black_level", ("BlackLevel", "BlackLevelRaw")),
                           ("pixel_format", ("PixelFormat",))):
            for name in names:
                try:
                    out[key] = getattr(self.camera, name).Value
                    break
                except (genicam.GenericException, AttributeError):
                    continue
            out.setdefault(key, None)
        return out

    def _enable_exposure_chunk(self) -> bool:
        # per-frame exposure closes the race where changing exposure mid-flight
        # retro-corrupts the timestamp of frames already integrating
        try:
            self.camera.ChunkModeActive.Value = True
            self.camera.ChunkSelector.Value = "ExposureTime"
            self.camera.ChunkEnable.Value = True
            return True
        except (genicam.GenericException, AttributeError):
            return False

    def _exposure_node(self):
        for name in ("ExposureTime", "ExposureTimeAbs"):
            try:
                return getattr(self.camera, name)
            except (genicam.GenericException, AttributeError):
                continue
        return None

    def _resulting_fps(self):
        for name in ("ResultingFrameRate", "ResultingFrameRateAbs"):
            try:
                return float(getattr(self.camera, name).Value)
            except (genicam.GenericException, AttributeError):
                continue
        return None

    def set_exposure_us(self, exposure_us: float) -> float:
        if self.camera is None or not self.camera.IsOpen():
            raise RuntimeError(
                "Camera is not open. Press Start Viewfinder before setting exposure."
            )

        try:
            self.camera.ExposureAuto.Value = "Off"
        except (genicam.GenericException, AttributeError):
            pass  # model has no auto exposure, nothing to disable

        node = self._exposure_node()
        if node is None:
            raise RuntimeError("Camera exposes neither ExposureTime nor ExposureTimeAbs.")

        try:
            low, high = float(node.Min), float(node.Max)
        except (genicam.GenericException, AttributeError) as e:
            raise RuntimeError(f"Could not read exposure limits: {e}") from e

        if not low <= exposure_us <= high:
            raise ValueError(
                f"Exposure {exposure_us / 1000.0:.3f} ms outside camera range "
                f"[{low / 1000.0:.3f}, {high / 1000.0:.3f}] ms"
            )

        # some models only accept multiples of a hardware increment and reject
        # anything else outright rather than rounding
        try:
            inc = float(node.Inc)
        except (genicam.GenericException, AttributeError):
            inc = None

        if inc:
            exposure_us = round(exposure_us / inc) * inc

        try:
            node.Value = exposure_us
            applied_us = float(node.Value)
        except genicam.GenericException as e:
            raise RuntimeError(
                f"Camera rejected exposure {exposure_us / 1000.0:.3f} ms: "
                f"{type(e).__name__}: {e}"
            ) from e

        self._exposure_s = applied_us * 1e-6

        fps = self._resulting_fps()
        ceiling = 1e6 / applied_us
        print(f"Exposure: requested {exposure_us / 1000.0:.3f} ms, "
              f"applied {applied_us / 1000.0:.3f} ms, "
              f"frame rate ceiling {ceiling:.2f} fps"
              + (f", camera reports {fps:.2f} fps" if fps is not None else ""))

        return applied_us

    def _sync_device_clock(self, samples: int = 7) -> bool:
        best_rtt = None
        offset = None

        for _ in range(samples):
            t0 = time.time()
            ticks = self._latch_device_ticks()
            t1 = time.time()

            if ticks is None:
                self.clock_synced = False
                return False

            rtt = t1 - t0
            if best_rtt is None or rtt < best_rtt:
                best_rtt = rtt
                device_seconds = ticks / self._tick_hz
                host_midpoint = (t0 + t1) / 2.0 # Approximation of when has the frame been taken by the hose machine.
                offset = host_midpoint - device_seconds

        self._clock_offset = offset
        self.clock_uncertainty = best_rtt / 2.0
        self._last_sync = time.monotonic()
        self.clock_synced = True
        return True

    def _resync_device_clock(self):
        previous = self._clock_offset

        if self._sync_device_clock():
            drift = self._clock_offset - previous
            print(f"Clock resync: drift {drift * 1e3:+.3f} ms, "
                f"uncertainty {self.clock_uncertainty * 1e3:.3f} ms")
            return

        # keep the last good offset so timestamps stay usable, but stop claiming they're synced, and don't retry on every grab
        self._last_sync = time.monotonic()
        print("WARNING: clock resync failed, timestamps now drifting from last offset.")

    def _capture_time(self, grab):
        # mid-exposure, because a streaking RSO centroid corresponds to the
        # middle of the integration window, not its start
        exposure_s = self._exposure_s
        if self._chunk_exposure:
            try:
                exposure_s = grab.ChunkExposureTime.Value * 1e-6
            except (genicam.GenericException, AttributeError):
                pass

        half_exposure = timedelta(seconds=exposure_s / 2.0)

        if self.time_source == "device" and grab.TimeStamp:
            epoch = grab.TimeStamp / self._tick_hz + self._clock_offset
            return datetime.fromtimestamp(epoch, tz=timezone.utc) + half_exposure

        # host fallback: retrieval happens after readout, so this ignores an
        # unknown transfer latency on top of the half-exposure correction
        return datetime.now(timezone.utc) - half_exposure

    def start_acquisition(self, index: int = 0) -> None:
        with self._state_lock:
            if self.acquisition_thread is not None and self.acquisition_thread.is_alive():
                print("Camera acquisition is already acquiring.")
                return

            with self.frame_lock:
                self.last_frame = None
                self.last_capture_time = None

            if not self.basler_devices:
                raise RuntimeError("No Basler cameras detected.")

            selected = self.basler_devices[index]
            self.active_device = selected

            self.camera = pylon.InstantCamera(
                self.tl_factory.CreateDevice(selected["device_info"])
            )

            self.camera.Open()

            self._set_pixel_format()
            self._tick_hz = self._get_tick_hz()
            self._exposure_s = self._get_exposure_seconds()

            if self._sync_device_clock():
                self.time_source = "device"
                print(f"Device clock: {self._tick_hz:.0f} Hz ticks, "
                      f"offset uncertainty {self.clock_uncertainty * 1e3:.3f} ms, "
                      f"exposure {self._exposure_s * 1e3:.1f} ms")
            elif self.require_device_time:
                raise RuntimeError(
                    f"{selected['model']} exposes no latchable device clock "
                    "and require_device_time is set."
                )
            else:
                self.time_source = "host"
                print("WARNING: no device clock available. Falling back to host "
                      "retrieval time, expect tens of ms of jitter.")

            self.converter = pylon.ImageFormatConverter()
            # mirrors FakeCameraManager: grayscale -> single channel, else 3-channel BGR
            if self.grayscale:
                self.converter.OutputPixelFormat = (pylon.PixelType_Mono16 if self.bit_depth > 8 else pylon.PixelType_Mono8)
            else:
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # must precede StartGrabbing: chunk mode changes the payload layout
            self._chunk_exposure = self._enable_exposure_chunk()
            if not self._chunk_exposure:
                print("Note: no per-frame exposure chunk. Mid-exposure correction "
                      "uses the last value set, which is stale for frames already "
                      "integrating when exposure changes.")

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self._stop_event.clear()

            self.acquisition_thread = threading.Thread(
                target=self._acquisition_loop,
                daemon=False
            )
            self.acquisition_thread.start()

        print(f"Started acquisition: {selected['name']}")

    def _acquisition_loop(self):
        while not self._stop_event.is_set():
            grab = None

            try:
                if self.camera is None or not self.camera.IsGrabbing():
                    self._stop_event.wait(0.05)
                    continue

                if (self.time_source == "device" and time.monotonic() - self._last_sync > self.resync_interval):
                    self._resync_device_clock()

                grab = self.camera.RetrieveResult(1500, pylon.TimeoutHandling_Return)

                if grab is None:
                    continue

                try:
                    if not grab.GrabSucceeded():
                        continue

                    capture_time = self._capture_time(grab)

                    image = self.converter.Convert(grab)
                    frame = image.GetArray()

                    with self.frame_lock:
                        self.last_frame = frame.copy()
                        self.last_capture_time = capture_time

                finally:
                    grab.Release()

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Acquisition loop error: {e}")

        print("Acquisition thread exited.")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None

            return self.last_frame.copy()

    def get_latest_frame_with_time(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None

            return self.last_frame.copy(), self.last_capture_time

    def stop_acquisition(self) -> None:
        with self._state_lock:
            if self.acquisition_thread is None:
                print("Camera acquisition already stopped.")
                return

            self._stop_event.set()

            if self.camera is not None and self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            thread = self.acquisition_thread

        thread.join(timeout=5.0)

        with self._state_lock:
            self.acquisition_thread = None

        print("Stopped acquisition.")

    def close(self):
        self.stop_acquisition()

        if self.camera is not None and self.camera.IsOpen():
            self.camera.Close()
            print("Closed camera.")

        self.camera = None
        self.converter = None
        self.active_device = None
        self.time_source = None
        self._chunk_exposure = False

        with self.frame_lock:
            self.last_frame = None
            self.last_capture_time = None