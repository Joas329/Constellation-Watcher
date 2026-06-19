import shutil
import subprocess
from pypylon import pylon

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class CameraManager:
    def __init__(self):
        self.verbose = True

        print("Initializzing Camera Manager")

        self.basler_devices = self.detect_basler_devices()

        if(self.verbose):
            print(f"Detected {len(self.basler_devices)} Basler camera(s):")
            for device in self.basler_devices:
                print(f"    Name: {device['name']}")
                print(f"    Model:  {device['model']}")
                print(f"    Serial: {device['serial']}")
                print(f"    Class:  {device['class']}")

        # Choose the first detected camera for now
        if self.basler_devices:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.basler_devices[0]))
            self.camera.Open()
            print(f"Opened camera: {self.camera.GetDeviceInfo().GetFriendlyName()}")
        else:
            print("No Basler cameras detected.")
            self.camera = None


    def detect_basler_devices(self):
        devices = []
        for d in pylon.TlFactory.GetInstance().EnumerateDevices():
            devices.append({
                "name": d.GetFriendlyName(),
                "model": d.GetModelName(),
                "serial": d.GetSerialNumber(),
                "class": d.GetDeviceClass(),
            })
        return devices
