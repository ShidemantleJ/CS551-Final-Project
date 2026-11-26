"""Count completed pedestrian crossings in an intersection video.

The crosswalk is a polygon drawn on the first video frame. A pedestrian is
counted once their tracked center enters and subsequently leaves that polygon.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "Select crosswalk"


def select_crosswalk(frame: np.ndarray) -> np.ndarray:
    """Let the user click polygon corners; Enter closes and accepts it."""
    points: list[tuple[int, int]] = []
    preview = frame.copy()

    def redraw() -> None:
        nonlocal preview
        preview = frame.copy()
        if points:
            for point in points:
                cv2.circle(preview, point, 5, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.polylines(preview, [np.asarray(points, np.int32)], False, (0, 255, 255), 2)
        cv2.putText(preview, "Click crosswalk corners | Enter: finish | u: undo | Esc: cancel",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()
    while True:
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) >= 3:
            break
        if key in (ord("u"), 8) and points:
            points.pop()
            redraw()
        if key == 27:
            cv2.destroyWindow(WINDOW_NAME)
            raise RuntimeError("Crosswalk selection cancelled")
    cv2.destroyWindow(WINDOW_NAME)
    return np.asarray(points, dtype=np.int32)


def load_or_select_polygon(frame: np.ndarray, roi_path: Path | None) -> np.ndarray:
    if roi_path and roi_path.exists():
        data = json.loads(roi_path.read_text(encoding="utf-8"))
        polygon = np.asarray(data["points"], dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"Invalid crosswalk polygon in {roi_path}")
        return polygon
    polygon = select_crosswalk(frame)
    if roi_path:
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        roi_path.write_text(json.dumps({"points": polygon.tolist()}, indent=2) + "\n", encoding="utf-8")
    return polygon


def write_csv(output: Path, counts: dict[int, int], observed_minutes: int,
              total_crossings: int, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["minute", "start_time_seconds", "end_time_seconds", "pedestrian_crossings"])
        for minute in range(observed_minutes):
            writer.writerow([minute + 1, minute * 60, (minute + 1) * 60, counts.get(minute, 0)])
        writer.writerow([])
        writer.writerow(["total_pedestrian_crossings", total_crossings])
        writer.writerow(["video_fps", round(fps, 3)])


def process_video(args: argparse.Namespace) -> None:
    # Import here so the ROI helper can be used without the Ultralytics package.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install with: pip install -r opencv/requirements.txt") from exc

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("The video contains no readable frames")
    polygon = load_or_select_polygon(first_frame, args.roi)

    model = YOLO(args.model)
    minute_counts: dict[int, int] = defaultdict(int)
    # Track IDs that have been inside the polygon and those already counted.
    entered: set[int] = set()
    counted: set[int] = set()
    frame_index = 0
    writer = None
    if args.annotated:
        height, width = first_frame.shape[:2]
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.annotated), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    try:
        while ok:
            result = model.track(
                first_frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                conf=args.confidence, iou=args.iou, verbose=False,
            )[0]
            canvas = first_frame.copy()
            cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2)
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box.astype(int)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    inside = cv2.pointPolygonTest(polygon, center, False) >= 0
                    if inside:
                        entered.add(track_id)
                    elif track_id in entered and track_id not in counted:
                        counted.add(track_id)
                        minute_counts[frame_index // max(1, int(fps * 60))] += 1
                    color = (0, 220, 0) if inside else (255, 120, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(canvas, f"ID {track_id}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(canvas, f"Completed crossings: {len(counted)}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if writer:
                writer.write(canvas)
            if args.preview:
                cv2.imshow("Pedestrian crossing counter", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
            ok, first_frame = capture.read()
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    duration = frame_index / fps
    observed_minutes = max(1, int(np.ceil(duration / 60)))
    write_csv(args.output, minute_counts, observed_minutes, len(counted), fps)
    print(f"Counted {len(counted)} completed crossings. CSV: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track pedestrians and count completed crosswalk crossings.")
    parser.add_argument("video", type=Path, help="Input intersection video")
    parser.add_argument("-o", "--output", type=Path, default=Path("pedestrian_counts.csv"), help="Output CSV path")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO model or weights path")
    parser.add_argument("--roi", type=Path, default=Path("crosswalk_roi.json"), help="Polygon JSON; created on first run")
    parser.add_argument("--annotated", type=Path, help="Optional path for an annotated MP4 output")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--preview", action="store_true", help="Display annotated video while processing; press q to stop")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        process_video(parse_args())
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
