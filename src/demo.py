"""Run the trained pipeline on a real courtship clip from dataset/videos."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import VIDEO_ROOT
from src.pipeline import FlyPipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
OUTPUT_DIR = ROOT / "outputs"


def load_pipeline() -> FlyPipeline:
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model at models/pipeline.joblib. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


def annotate(gray: np.ndarray, detections) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        color = (40, 90, 220) if detection.sex == "M" else (40, 160, 70)
        if detection.sex == "?":
            color = (160, 160, 160)
        cv2.drawContours(canvas, [detection.contour], -1, color, 2)
        radians = np.deg2rad(detection.heading)
        tip = (
            int(detection.cx + np.sin(radians) * 70),
            int(detection.cy - np.cos(radians) * 70),
        )
        cv2.arrowedLine(canvas, (int(detection.cx), int(detection.cy)), tip, color, 2, tipLength=0.25)
        label = detection.sex
        if detection.wing_left is not None:
            label = f"M  L{detection.wing_left:.0f} R{detection.wing_right:.0f}"
        cv2.putText(
            canvas,
            label,
            (int(detection.cx) + 10, int(detection.cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def run(video_path: Path, max_frames: int | None) -> dict:
    video_path = video_path if video_path.is_absolute() else ROOT / video_path
    pipeline = load_pipeline()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    fps_source = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "demo.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_source,
        (width, height),
    )
    traces = []
    preview = []
    n_frames = 0
    started = time.perf_counter()
    with (OUTPUT_DIR / "tracks.csv").open("w", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["frame", "sex", "x", "y", "heading_deg", "wing_left_deg", "wing_right_deg"])
        while True:
            if max_frames is not None and n_frames >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = pipeline.predict(gray)
            for detection in detections:
                csv_writer.writerow(
                    [
                        n_frames,
                        detection.sex,
                        f"{detection.cx:.2f}",
                        f"{detection.cy:.2f}",
                        f"{detection.heading:.2f}",
                        "" if detection.wing_left is None else f"{detection.wing_left:.2f}",
                        "" if detection.wing_right is None else f"{detection.wing_right:.2f}",
                    ]
                )
                if detection.sex == "M" and detection.wing_left is not None:
                    traces.append((n_frames / fps_source, detection.wing_left, detection.wing_right))
            annotated = annotate(gray, detections)
            writer.write(annotated)
            if n_frames % 80 == 0:
                preview.append(annotated)
            n_frames += 1
    capture.release()
    writer.release()
    elapsed = time.perf_counter() - started
    if preview:
        thumb = 280
        row = np.concatenate([cv2.resize(frame, (thumb, thumb)) for frame in preview[:6]], axis=1)
        cv2.imwrite(str(OUTPUT_DIR / "preview.jpg"), row)
    if traces:
        times, left, right = zip(*traces)
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        axis.plot(times, right, label="Right wing", color="#d35400")
        axis.plot(times, left, label="Left wing", color="#2980b9")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Wing angle (degrees)")
        axis.set_ylim(0, 90)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / "wing_trace.png", dpi=140)
        plt.close(figure)
    report = {
        "video": str(video_path.relative_to(ROOT)),
        "frames": n_frames,
        "fps": n_frames / elapsed if elapsed else 0.0,
        "source_fps": fps_source,
        "width": width,
        "height": height,
    }
    (OUTPUT_DIR / "demo_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a courtship clip with the trained pipeline.")
    parser.add_argument("--video", type=Path, default=VIDEO_ROOT / "test4.mp4")
    parser.add_argument("--frames", type=int, default=0, help="Stop after this many frames. 0 means the whole clip.")
    args = parser.parse_args()
    report = run(args.video, args.frames or None)
    print(
        f"{report['video']}  {report['frames']} frames at {report['width']}x{report['height']}  "
        f"{report['fps']:.1f} FPS processing"
    )
    print(f"Wrote {OUTPUT_DIR / 'demo.mp4'}, tracks.csv, preview.jpg, wing_trace.png")


if __name__ == "__main__":
    main()
