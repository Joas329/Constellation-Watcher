# Constellation Watcher

## Description

Constellation Watcher is a live sky-imaging and satellite-tracking software.

It runs a camera, plate-solves each frame against a star catalog, and
correlates the solved sky position against a TLE catalog to identify
resident space objects (RSOs), satellites, visible in the field of view,
in real time.

The system is built as a concurrent pipes-and-filters pipeline: a camera
producer publishes frames onto a channel, a background-subtraction and
detection stage cleans each frame, a plate-solving stage derives a WCS
(where the camera was pointing), and an RSO tracking stage projects known
satellites onto the frame using that WCS. Each stage runs on its own
thread and communicates only through single-slot, latest-wins channels, so
a slow stage (like a 30-second blind plate solve) never blocks frame
acquisition. Observer location comes either from a fixed site (for
replaying archived frames from a known location and epoch) or live from a
GPS receiver (for tracking against the current sky). A local web control
panel exposes the raw feed, the processed feed, and the solved-frame RSO
overlay, along with camera controls and system/GPS telemetry.

Confirmed RSOs are archived automatically: the raw frame and a JSON
sidecar (detection time, observer location, WCS center, matched stars, and
every RSO hit with its pixel position and velocity) are written to a
removable drive whenever a detection occurs.

## Requirements

- Python 3.10+
- [astrometry.net](https://astrometry.net) (`solve-field` on `PATH`) — this
  is a compiled binary, not a Python package, so it's the one dependency
  `setup.bash` cannot install via pip. See below for how to get it.
- `libturbojpeg` (native library backing the Python `turbojpeg` wrapper)
- A Basler USB3 camera + [pypylon](https://github.com/basler/pypylon) (only
  needed for the live camera; the fake camera replays saved PNGs and needs
  neither the camera nor pypylon's GenTL producer)
- A u-blox NMEA GPS receiver on a serial port (only needed for live-GPS
  observer mode; fixed-site mode needs no GPS)

### One-step setup

```bash
./setup.bash
```

This single script:

1. Creates a Python virtual environment at `./celestial_env` and installs
   `requirements.txt` into it via `pip`.
2. Downloads the astrometry.net 4100-series index files (Tycho-2 based,
   correct for wide fields, see [Astrometry Index Files](#astrometry-index-files)
   below if your lens has a different field of view) into `celestial_watcher/astrometry_indices/`, using a small stdlib-only
   Python downloader, no `wget`, no `apt`.
3. Generates `celestial_watcher/astrometry.cfg` pointing at that directory.

## Running

```bash
source celestial_env/bin/activate
python3 launch.py
```

Then open `http://<host>:5000/` for the control panel. Camera acquisition
starts from the panel's **Start Viewfinder** button

### Fake camera vs. real camera

`launch.py` has a `USE_FAKE_CAMERA` flag. The fake camera replays a
directory of PNG frames at a fixed rate — useful for testing the pipeline,
UI, and RSO correlation without hardware. The real camera drives the
Basler unit directly.

## Astrometry Index Files

`setup.bash` installs the **4100-series**, which astrometry.net recommends
for images wider than 1 degree. Check your lens's actual field of view
(focal length + sensor size) and pass matching `--scale-low`/
`--scale-high` bounds (in `PlateSolverStage.py`) — an incorrect scale hint
is the most common reason a frame with plenty of visible stars still fails
to solve. If your field of view is narrower than 1 degree, you'll also
need the 5200-series indexes; see <https://data.astrometry.net/> for the
full breakdown, and grab them the same way `setup.bash` grabs the
4100-series (adjust `INDEX_URL` to `http://data.astrometry.net/5200/`).

## Detection Archive

Confirmed detections are saved to the directory set by
`DETECTION_SAVE_DIRECTORY` in `RSOTrackerStage.py`. Each detection writes a
pair of files sharing a timestamped basename:

```
detection_20260725_055518_114733.png    # raw solved frame, unannotated
detection_20260725_055518_114733.json   # capture time, observer location,
                                         # WCS center, RSO hits, matched stars
```

Keeping the raw frame un-annotated means it stays usable as-is for
re-analysis; the JSON carries everything needed to re-render the overlay
or re-run correlation later.
