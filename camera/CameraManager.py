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
        if not self.basler_devices:
            raise RuntimeError("No Basler cameras detected.")

        if self.camera is not None and self.camera.IsOpen():
            if not self.camera.IsGrabbing():
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            return

        selected = self.basler_devices[index]
        self.active_device = selected

        self.camera = pylon.InstantCamera(self.tl_factory.CreateDevice(selected["device_info"]))

        self.camera.Open()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        print(f"Started acquisition: {selected['name']}")

    def stop_acquisition(self):
        if self.camera is not None and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
            print("Stopped acquisition.")

    def get_frame_bgr(self):
        if self.camera is None or not self.camera.IsOpen():
            raise RuntimeError("Camera is not open. Call start_acquisition() first.")

        if not self.camera.IsGrabbing():
            raise RuntimeError("Camera is not grabbing. Call start_acquisition() first.")

        # Wait 5000 ms for a frame and retrieve it
        grab = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        try:
            if not grab.GrabSucceeded():
                raise RuntimeError("Failed to grab frame from Basler camera.")

            image = self.converter.Convert(grab)
            return image.GetArray()

        finally:
            grab.Release()

    def close(self):
        self.stop_acquisition()

        if self.camera is not None and self.camera.IsOpen():
            self.camera.Close()
            print("Closed camera.")

        self.camera = None
        self.converter = None
        self.active_device = None