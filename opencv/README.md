# Intersection pedestrian counter

This application uses OpenCV for video handling and crosswalk selection, Ultralytics YOLO for pedestrian detection, and ByteTrack for maintaining pedestrian identities. It counts a tracked pedestrian once when they enter and then leave the selected crosswalk polygon. The CSV reports completed crossings by video minute, matching the human verification task described in the repository README.

## Setup

From the repository root, create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r opencv/requirements.txt
```

YOLO weights are downloaded by Ultralytics on first use if they are not present locally. OpenCV's GUI support is required for drawing the crosswalk.

## Run

```bash
python opencv/video_processor.py path/to/intersection.mp4 \
  --output results/pedestrian_counts.csv \
  --roi results/crosswalk_roi.json \
  --annotated results/annotated.mp4 \
  --preview
```

On first run, click at least three crosswalk boundary points in the first frame and press Enter. Press `u` to undo a point or Escape to cancel. The polygon is saved to the ROI JSON and reused on later runs. Delete that file or pass a new `--roi` path to draw another polygon. The ROI coordinates are tied to the video's frame dimensions and camera view.

The preview is optional; `q` stops processing early. The output CSV includes one row for each observed video minute (including zero counts), a total, and the source frame rate. Counts are attributed to the minute in which the track exits the crosswalk.

## Counting behavior and limits

- Only COCO's person class is tracked.
- A crossing means the track center enters the polygon and later leaves it. Someone who is inside at the end of the video is not counted.
- ByteTrack IDs can fragment during occlusion or merge nearby people. The total is therefore an estimate and should be compared with human labels.
- The default model is `yolo11n.pt`; pass `--model` with another Ultralytics model or local weights as needed.

```text
python opencv/video_processor.py --help
```
