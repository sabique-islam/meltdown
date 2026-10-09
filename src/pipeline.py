"""Four-stage fly tracker: count, sex, orientation, and male wing angle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import (
    angular_difference,
    body_mask,
    centroid,
    contour_features,
    fly_contours,
    hog_descriptor,
    orientation_patch,
    upright_fly,
    wing_patches,
)
@dataclass
class Detection:
    sex: str
    cx: float
    cy: float
    heading: float
    wing_left: float | None
    wing_right: float | None
    contour: np.ndarray


class FlyPipeline:
    def __init__(self) -> None:
        self.fly_count = DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=0)
        self.sex = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        self.orientation = Pipeline(
            [
                ("pca", PCA(n_components=15, random_state=0)),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        self.wing = Pipeline(
            [
                ("pca", PCA(n_components=40, random_state=0)),
                ("reg", LinearRegression()),
            ]
        )

    def predict(self, gray: np.ndarray) -> list[Detection]:
        mask = body_mask(gray)
        contours = fly_contours(mask)
        if not contours:
            return []
        areas = np.array([[contour_features(contour)[0]] for contour in contours])
        counts = self.fly_count.predict(areas)

        singles = [contour for contour, count in zip(contours, counts) if count == 1]
        sexes = ["?"] * len(singles)
        if len(singles) == 2:
            ordered = sorted(range(2), key=lambda index: centroid(singles[index]))
            features = []
            for index in ordered:
                area, aspect = contour_features(singles[index])
                features.extend([area, aspect])
            male_then_female = int(self.sex.predict(np.array(features, ndmin=2))[0])
            label = ["M", "F"] if male_then_female else ["F", "M"]
            for slot, index in enumerate(ordered):
                sexes[index] = label[slot]

        detections: list[Detection] = []
        for contour, sex in zip(singles, sexes):
            cx, cy = centroid(contour)
            patch, base_heading = orientation_patch(gray, contour)
            flip = int(self.orientation.predict(hog_descriptor(patch, "orientation").reshape(1, -1))[0])
            heading = (base_heading + 180.0) % 360.0 if flip else base_heading
            wing_left = wing_right = None
            if sex == "M":
                upright = upright_fly(gray, cx, cy, heading)
                right_patch, left_patch = wing_patches(upright)
                wing_right = _clip_angle(self.wing.predict(hog_descriptor(right_patch, "wing").reshape(1, -1))[0])
                wing_left = _clip_angle(self.wing.predict(hog_descriptor(left_patch, "wing").reshape(1, -1))[0])
            detections.append(Detection(sex, cx, cy, heading, wing_left, wing_right, contour))
        return detections


def match_fly(contour: np.ndarray, flies: list) -> object | None:
    matches = [fly for fly in flies if cv2_contains(contour, fly.center[0], fly.center[1])]
    if len(matches) == 1:
        return matches[0]
    return None


def cv2_contains(contour: np.ndarray, x: float, y: float) -> bool:
    import cv2

    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def count_flies_inside(contour: np.ndarray, flies: list) -> int:
    return sum(1 for fly in flies if cv2_contains(contour, fly.center[0], fly.center[1]))


def orientation_label(base_heading: float, true_heading: float) -> int:
    return int(abs(angular_difference(base_heading, true_heading)) > 90.0)


def _clip_angle(value: float) -> float:
    return float(np.clip(value, 0.0, 90.0))
