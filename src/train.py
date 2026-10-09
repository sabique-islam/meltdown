"""Train the four pipeline stages on dataset/images and write a metrics report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import export_text

from src.dataset import load_frames
from src.features import (
    angular_difference,
    body_mask,
    contour_features,
    fly_contours,
    hog_descriptor,
    orientation_patch,
    upright_fly,
    wing_patches,
)
from src.pipeline import FlyPipeline, count_flies_inside, match_fly, orientation_label

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
REPORT_PATH = ROOT / "outputs" / "train_metrics.json"


def _split(rng: np.random.Generator, n_frames: int) -> tuple[set[int], set[int]]:
    order = rng.permutation(n_frames)
    n_test = n_frames // 3
    return set(int(i) for i in order[n_test:]), set(int(i) for i in order[:n_test])


def _error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true != y_pred))


def _centroid_key(contour) -> tuple[float, float]:
    from src.features import centroid

    return centroid(contour)


def collect(frames, train_ids: set[int], test_ids: set[int]):
    buckets = {
        split: {key: [] for key in ("count_x", "count_y", "sex_x", "sex_y", "ori_x", "ori_y", "wing_x", "wing_y")}
        for split in ("train", "test")
    }
    for index, frame in enumerate(frames):
        split = "train" if index in train_ids else "test"
        bucket = buckets[split]
        gray = frame.gray()
        contours = fly_contours(body_mask(gray))
        singles = []
        for contour in contours:
            count = min(count_flies_inside(contour, frame.flies), 2)
            area, _aspect = contour_features(contour)
            bucket["count_x"].append([area])
            bucket["count_y"].append(count)
            if count == 1:
                singles.append(contour)
        if len(singles) == 2:
            ordered = sorted(singles, key=_centroid_key)
            matched = [match_fly(contour, frame.flies) for contour in ordered]
            if all(fly is not None for fly in matched):
                features = []
                for contour in ordered:
                    area, aspect = contour_features(contour)
                    features.extend([area, aspect])
                male_then_female = int(matched[0].sex == "M")
                bucket["sex_x"].append(features)
                bucket["sex_y"].append(male_then_female)
                if split == "train":
                    bucket["sex_x"].append(features[2:] + features[:2])
                    bucket["sex_y"].append(1 - male_then_female)
        for contour in singles:
            fly = match_fly(contour, frame.flies)
            if fly is None or fly.heading is None:
                continue
            patch, base_heading = orientation_patch(gray, contour)
            label = orientation_label(base_heading, fly.heading)
            bucket["ori_x"].append(hog_descriptor(patch, "orientation"))
            bucket["ori_y"].append(label)
            if split == "train":
                bucket["ori_x"].append(hog_descriptor(np.rot90(patch, 2), "orientation"))
                bucket["ori_y"].append(1 - label)
            if fly.sex != "M" or fly.wing_left is None or fly.wing_right is None:
                continue
            cx, cy = _centroid_key(contour)
            upright = upright_fly(gray, cx, cy, fly.heading)
            right_patch, left_patch = wing_patches(upright)
            bucket["wing_x"].append(hog_descriptor(right_patch, "wing"))
            bucket["wing_y"].append(fly.wing_right)
            bucket["wing_x"].append(hog_descriptor(left_patch, "wing"))
            bucket["wing_y"].append(fly.wing_left)
    return buckets


def _as_array(rows: list) -> np.ndarray:
    if not rows:
        return np.empty((0, 1), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _stage(train_error: float, test_error: float, n_train: int, n_test: int) -> dict:
    return {
        "train_error": train_error,
        "test_error": test_error,
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def cascade_on_split(pipeline: FlyPipeline, frames, test_ids: set[int]) -> dict:
    """Score the full pipeline on held-out labeled frames."""
    position, sex_hits, heading, wings = [], [], [], []
    for index in sorted(test_ids):
        frame = frames[index]
        detections = pipeline.predict(frame.gray())
        for fly in frame.flies:
            best, distance = None, 1e9
            for detection in detections:
                gap = (detection.cx - fly.center[0]) ** 2 + (detection.cy - fly.center[1]) ** 2
                if gap < distance:
                    best, distance = detection, gap
            distance = distance**0.5
            if best is None or distance > 80:
                continue
            position.append(distance)
            sex_hits.append(int(best.sex == fly.sex))
            if fly.heading is not None:
                heading.append(abs(angular_difference(best.heading, fly.heading)))
            if fly.sex == "M" and fly.wing_left is not None and best.wing_left is not None:
                wings.append(abs(best.wing_left - fly.wing_left))
                wings.append(abs(best.wing_right - fly.wing_right))
    return {
        "matched": len(position),
        "mean_position_error_px": float(np.mean(position)) if position else None,
        "sex_accuracy": float(np.mean(sex_hits)) if sex_hits else None,
        "mean_heading_error_deg": float(np.mean(heading)) if heading else None,
        "mean_wing_error_deg": float(np.mean(wings)) if wings else None,
    }


def train(seed: int) -> dict:
    started = time.perf_counter()
    frames = load_frames()
    rng = np.random.default_rng(seed)
    train_ids, test_ids = _split(rng, len(frames))
    buckets = collect(frames, train_ids, test_ids)
    pipeline = FlyPipeline()

    count_train_x = _as_array(buckets["train"]["count_x"])
    count_train_y = np.asarray(buckets["train"]["count_y"], dtype=int)
    count_test_x = _as_array(buckets["test"]["count_x"])
    count_test_y = np.asarray(buckets["test"]["count_y"], dtype=int)
    pipeline.fly_count.fit(count_train_x, count_train_y)

    sex_train_x = _as_array(buckets["train"]["sex_x"])
    sex_train_y = np.asarray(buckets["train"]["sex_y"], dtype=int)
    sex_test_x = _as_array(buckets["test"]["sex_x"])
    sex_test_y = np.asarray(buckets["test"]["sex_y"], dtype=int)
    pipeline.sex.fit(sex_train_x, sex_train_y)

    ori_train_x = _as_array(buckets["train"]["ori_x"])
    ori_train_y = np.asarray(buckets["train"]["ori_y"], dtype=int)
    ori_test_x = _as_array(buckets["test"]["ori_x"])
    ori_test_y = np.asarray(buckets["test"]["ori_y"], dtype=int)
    n_components = min(15, max(ori_train_x.shape[0] - 1, 1), ori_train_x.shape[1])
    pipeline.orientation.set_params(pca__n_components=n_components)
    pipeline.orientation.fit(ori_train_x, ori_train_y)

    wing_train_x = _as_array(buckets["train"]["wing_x"])
    wing_train_y = np.asarray(buckets["train"]["wing_y"], dtype=float)
    wing_test_x = _as_array(buckets["test"]["wing_x"])
    wing_test_y = np.asarray(buckets["test"]["wing_y"], dtype=float)
    wing_components = min(40, max(wing_train_x.shape[0] - 1, 1), wing_train_x.shape[1])
    pipeline.wing.set_params(pca__n_components=wing_components)
    pipeline.wing.fit(wing_train_x, wing_train_y)

    def regression_sigma(model, features, target) -> float:
        if len(target) == 0:
            return float("nan")
        return float(np.std(target - model.predict(features)))

    report = {
        "source": "dataset/images",
        "n_frames": len(frames),
        "n_train_frames": len(train_ids),
        "n_test_frames": len(test_ids),
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 2),
        "tree": export_text(pipeline.fly_count, feature_names=["area"]),
        "orientation_components": int(n_components),
        "orientation_variance": float(np.sum(pipeline.orientation.named_steps["pca"].explained_variance_ratio_)),
        "wing_components": int(wing_components),
        "wing_variance": float(np.sum(pipeline.wing.named_steps["pca"].explained_variance_ratio_)),
        "stages": {
            "FlyCount": _stage(
                _error(count_train_y, pipeline.fly_count.predict(count_train_x)),
                _error(count_test_y, pipeline.fly_count.predict(count_test_x)),
                len(count_train_y),
                len(count_test_y),
            ),
            "Sex": _stage(
                _error(sex_train_y, pipeline.sex.predict(sex_train_x)),
                _error(sex_test_y, pipeline.sex.predict(sex_test_x)),
                len(sex_train_y),
                len(sex_test_y),
            ),
            "Orientation": _stage(
                _error(ori_train_y, pipeline.orientation.predict(ori_train_x)),
                _error(ori_test_y, pipeline.orientation.predict(ori_test_x)),
                len(ori_train_y),
                len(ori_test_y),
            ),
            "WingAngle": {
                "train_sigma": regression_sigma(pipeline.wing, wing_train_x, wing_train_y),
                "test_sigma": regression_sigma(pipeline.wing, wing_test_x, wing_test_y),
                "n_train": int(len(wing_train_y)),
                "n_test": int(len(wing_test_y)),
            },
        },
    }
    report["cascade_test"] = cascade_on_split(pipeline, frames, test_ids)
    report["seconds"] = round(time.perf_counter() - started, 2)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on the labeled frames in dataset/images.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = train(args.seed)
    print(
        f"{report['n_frames']} labeled frames  "
        f"train {report['n_train_frames']}  test {report['n_test_frames']}  "
        f"fit in {report['seconds']}s"
    )
    print(report["tree"])
    print(
        f"Orientation PCA {report['orientation_components']} components, "
        f"{report['orientation_variance']:.1%} variance"
    )
    print(f"Wing PCA {report['wing_components']} components, {report['wing_variance']:.1%} variance")
    for name, stage in report["stages"].items():
        if "test_error" in stage:
            print(
                f"{name:12} train {stage['train_error']:.1%}  test {stage['test_error']:.1%}  "
                f"n={stage['n_train']}/{stage['n_test']}"
            )
        else:
            print(
                f"{name:12} train σ={stage['train_sigma']:.2f}°  test σ={stage['test_sigma']:.2f}°  "
                f"n={stage['n_train']}/{stage['n_test']}"
            )
    cascade = report["cascade_test"]
    print(
        f"Held-out cascade  sex {cascade['sex_accuracy']:.1%}  "
        f"heading {cascade['mean_heading_error_deg']:.1f}°  "
        f"wing {cascade['mean_wing_error_deg']:.1f}°  "
        f"position {cascade['mean_position_error_px']:.1f}px  "
        f"matched {cascade['matched']}"
    )
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
