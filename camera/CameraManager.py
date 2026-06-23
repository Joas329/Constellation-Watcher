import threading
from typing import Any
from pypylon import pylon


class CameraManager:
    def __init__(self):
        print("Initializing Camera Manager")

        self.camera = None
        self.converter = None
        self.active_device = None

        self.tl_factory = pylon.TlFactory.GetInstance()
        self.basler_devices = self.detect_basler_devices()

        self.last_frame = None
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

    def start_acquisition(self, index: int = 0):
        with self._state_lock:
            if self.acquisition_thread is not None and self.acquisition_thread.is_alive():
                return "Camera acquisition is already acquiring."

            with self.frame_lock:
                self.last_frame = None

            if not self.basler_devices:
                raise RuntimeError("No Basler cameras detected.")

            selected = self.basler_devices[index]
            self.active_device = selected

            self.camera = pylon.InstantCamera(
                self.tl_factory.CreateDevice(selected["device_info"])
            )

            self.camera.Open()

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self._stop_event.clear()

            self.acquisition_thread = threading.Thread(
                target=self._acquisition_loop,
                daemon=False
            )
            self.acquisition_thread.start()

        print(f"Started acquisition: {selected['name']}")
        return f"Started acquisition on camera: {selected['name']}"

    def _acquisition_loop(self):
        while not self._stop_event.is_set():
            grab = None

            try:
                if self.camera is None or not self.camera.IsGrabbing():
                    self._stop_event.wait(0.05)
                    continue

                grab = self.camera.RetrieveResult(1500, pylon.TimeoutHandling_Return)

                if grab is None:
                    continue

                try:
                    if not grab.GrabSucceeded():
                        continue

                    image = self.converter.Convert(grab)
                    frame = image.GetArray()

                    with self.frame_lock:
                        self.last_frame = frame.copy()

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

    def stop_acquisition(self):
        with self._state_lock:
            if self.acquisition_thread is None:
                return "Camera acquisition already stopped."

            self._stop_event.set()

            if self.camera is not None and self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            thread = self.acquisition_thread

        thread.join(timeout=5.0)

        with self._state_lock:
            self.acquisition_thread = None

        print("Stopped acquisition.")
        return "Stopped acquisition."

    def close(self):
        self.stop_acquisition()

        if self.camera is not None and self.camera.IsOpen():
            self.camera.Close()
            print("Closed camera.")

        self.camera = None
        self.converter = None
        self.active_device = None

        with self.frame_lock:
            self.last_frame = None