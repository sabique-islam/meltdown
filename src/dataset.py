"""Labeled courtship frames from dataset/images.

Each JSON is a LabelMe file. Point names follow the CS229 annotation:

- fh, fp, fa: female head, a point in the body, abdomen
- mh, mp, mp2, ma: male head, two body points, abdomen
- mw: male wing tips (usually two)

fp and mp are the body centers used to match a contour to a fly.
Heading points from the abdomen toward the head, clockwise from image-up.
A wing angle is the opening from the tail axis, in [0, 90].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "dataset" / "images"
VIDEO_ROOT = ROOT / "dataset" / "videos"


@dataclass
class FlyAnn:
    sex: str
    center: tuple[float, float]
    head: tuple[float, float] | None
    abdomen: tuple[float, float] | None
    heading: float | None
    wing_left: float | None
    wing_right: float | None


@dataclass
class FrameAnn:
    image_path: Path
    session: str
    flies: list[FlyAnn]

    def gray(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.image_path)
        return image


def _point(raw) -> tuple[float, float]:
    return float(raw[0]), float(raw[1])


def heading_from_ends(head: tuple[float, float], abdomen: tuple[float, float]) -> float:
    dx = head[0] - abdomen[0]
    dy = head[1] - abdomen[1]
    return float(np.degrees(np.arctan2(dx, -dy)) % 360.0)


def _opening(head, abdomen, center, wing) -> float | None:
    tail = np.array(abdomen, dtype=float) - np.array(head, dtype=float)
    norm = np.linalg.norm(tail)
    if norm < 1:
        return None
    tail /= norm
    vector = np.array(wing, dtype=float) - np.array(center, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1:
        return None
    vector /= length
    cross = tail[0] * vector[1] - tail[1] * vector[0]
    dot = float(np.clip(np.dot(tail, vector), -1.0, 1.0))
    signed = float(np.degrees(np.arctan2(cross, dot)))
    # Negative signed angle is the fly's right once the head points up.
    opening = min(abs(signed), 180.0 - abs(signed))
    return signed, float(np.clip(opening, 0.0, 90.0))


def _fly_from_points(sex: str, points: dict[str, list]) -> FlyAnn | None:
    center_key = "fp" if sex == "F" else "mp"
    head_key = "fh" if sex == "F" else "mh"
    abdomen_key = "fa" if sex == "F" else "ma"
    if center_key not in points:
        return None
    head = _point(points[head_key][0]) if head_key in points else None
    abdomen = _point(points[abdomen_key][0]) if abdomen_key in points else None
    heading = heading_from_ends(head, abdomen) if head and abdomen else None
    wing_left = wing_right = None
    if sex == "M" and head and abdomen and "mw" in points:
        signed_open = []
        for raw in points["mw"]:
            opened = _opening(head, abdomen, _point(points["mp"][0]), raw)
            if opened is not None:
                signed_open.append(opened)
        rights = [opening for signed, opening in signed_open if signed < 0]
        lefts = [opening for signed, opening in signed_open if signed >= 0]
        if rights:
            wing_right = float(np.mean(rights))
        if lefts:
            wing_left = float(np.mean(lefts))
    return FlyAnn(sex, _point(points[center_key][0]), head, abdomen, heading, wing_left, wing_right)


def _image_path(json_path: Path, image_name: str) -> Path:
    candidate = json_path.with_name(image_name)
    if candidate.exists():
        return candidate
    matches = [path for path in json_path.parent.glob(json_path.stem + ".*") if path.suffix != ".json"]
    if not matches:
        raise FileNotFoundError(json_path)
    return matches[0]


def load_frames(root: Path = IMAGE_ROOT) -> list[FrameAnn]:
    frames = []
    for json_path in sorted(root.rglob("*.json")):
        payload = json.loads(json_path.read_text())
        points: dict[str, list] = {}
        for shape in payload["shapes"]:
            points.setdefault(shape["label"], []).append(shape["points"][0])
        flies = []
        for sex in ("F", "M"):
            fly = _fly_from_points(sex, points)
            if fly is not None:
                flies.append(fly)
        frames.append(
            FrameAnn(
                _image_path(json_path, payload.get("imagePath", "")),
                json_path.parent.name,
                flies,
            )
        )
    return frames
