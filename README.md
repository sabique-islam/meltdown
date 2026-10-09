# Real-time Detailed Video Analysis of Fruit Flies

A four-stage tracker for a male and female fruit fly in a circular well. On each grayscale frame it estimates position, sex, heading, and the male's left and right wing angles.

The pipeline follows Steven Herbst's CS229 Fall 2018 project ([report](https://cs229.stanford.edu/proj2018/report/61.pdf)): a decision tree on contour area, logistic regression on area and aspect ratio, then HOG features reduced with PCA for heading and wing angle. Training uses the labeled frames in `dataset/`.

---

## Dataset

`dataset/` is in `.gitignore`. It is about 2 GB. Clone the code, then put the data at the repository root before training or running the demo:

```
dataset/
  images/    326 LabelMe frames, in the five folders below
  videos/    test1.mp4–test5.mp4
```

The files are the input bundle from [sgherbst/cs229-project](https://github.com/sgherbst/cs229-project). That repository links the [Dropbox download](https://www.dropbox.com/sh/78inyvw2ouut74a/AACc1DYrC1G0UxujwT-6ryRKa?dl=0).

`dataset/videos/` has five 1530×1530 clips (`test1.mp4`–`test5.mp4`) cropped from a Basler acA2440-35um courtship recording sent by Ryan York on 1 October 2018.

`dataset/images/` has the 326 LabelMe frames:

| Folder | What it is | Frames |
| --- | --- | --- |
| `12-04_17-54-43` | Sampled uniformly from the video | 100 |
| `12-05_12-43-00` | Flies in contact | 33 |
| `12-07_16-45-00` | Flies running along the wall | 50 |
| `12-08_11-15-00` | Wing-angle frames from `test4.mp4` | 102 |
| `12-08_22-00-00` | Wing-angle frames from `test1.mp4` | 41 |

Point names are `fh`, `fp`, `fa` (female head, body, abdomen) and `mh`, `mp`, `mp2`, `ma`, `mw` (male, including wing tips). `fp` and `mp` are the body points used to match a contour to a fly.

Heading is the direction from the abdomen to the head, measured clockwise from image-up. A wing angle is the opening of that wing off the tail axis, in degrees from 0 to 90. Zero lies along the abdomen. Ninety is straight out to the side. The left and right wings are the two sides of that tail axis.

---

## Held-out results

`python -m src.train` keeps 108 of the 326 frames out of training. On the flies it still separates, sex is correct, heading error is about 1°, and wing error is about 6°. The misses are the frames where the two bodies become one contour.

| Test frames | Flies | Missed | Heading (median) | Wing error (median) |
| --- | ---: | ---: | ---: | ---: |
| Uniform sample | 60 | 0 | 0.8° | 5.0° |
| On the wall | 36 | 0 | 1.1° | 5.3° |
| Wing frames from `test4.mp4` | 64 | 0 | 1.0° | 3.7° |
| Wing frames from `test1.mp4` | 34 | 8 | 1.1° | 8.5° |
| Flies in contact | 22 | 14 | 0.9° | 5.6° |

22 of 216 labeled flies are missed, 14 of them in the contact folder. A merged contour is classified as two flies and then dropped, because sex and heading run only when the bodies are separate. Wall-climbing flies are tracked. The 34 px gap between the contour centroid and `fp`/`mp` is the distance to that labeled interior point, not a missed detection.

Stage test error, before the stages are chained: fly count 0%, sex 0%, orientation 0%, wing angle σ = 7.26°. Herbst's wing-angle figure on his own split was σ = 2.92°.

---

## Setup

Use Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or:

```bash
make setup
```

---

## Train

From the repository root:

```bash
python -m src.train
```

This reads all 326 labeled frames, holds out one third, fits the four stages, and writes:

- `models/pipeline.joblib`
- `outputs/train_metrics.json`

---

## Run the demo

```bash
python -m src.demo
```

By default this annotates `dataset/videos/test4.mp4` (the 4:12–4:30 excerpt) and writes:

- `outputs/demo.mp4`
- `outputs/tracks.csv`
- `outputs/preview.jpg`
- `outputs/wing_trace.png`
- `outputs/demo_metrics.json`

Another clip:

```bash
python -m src.demo --video dataset/videos/test1.mp4
```

---