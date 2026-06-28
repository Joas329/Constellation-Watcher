#!/usr/bin/env python3

import getpass
import subprocess
from pathlib import Path

SERVICE_NAME = "constellation-watcher.service"

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_DIR / "celestial" / "bin" / "python"
LAUNCH = PROJECT_DIR / "launch.py"
USER = getpass.getuser()

SERVICE = f"""[Unit]
Description=Constellation Watcher
After=network.target

[Service]
Type=simple
User={USER}
WorkingDirectory={PROJECT_DIR}
ExecStart={PYTHON} {LAUNCH}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""

SERVICE_PATH = Path("/etc/systemd/system") / SERVICE_NAME

with open(SERVICE_PATH, "w") as f:
    f.write(SERVICE)

subprocess.run(["systemctl", "daemon-reload"], check=True)
subprocess.run(["systemctl", "enable", SERVICE_NAME], check=True)
subprocess.run(["systemctl", "restart", SERVICE_NAME], check=True)

print("Daemon installed successfully.")
print(f"Service file: {SERVICE_PATH}")
