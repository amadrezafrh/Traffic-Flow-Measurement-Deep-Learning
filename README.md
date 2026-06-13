# Traffic Flow Measurement with Deep Learning

End-to-end pipeline that detects, tracks, and lane-classifies vehicles
from aerial intersection footage, then reports per-lane counts, average
speed, and immovable-vehicle counts in real time. Built for a national
traffic-monitoring deployment in Tehran.

<p align="center"><img src="data/helpers/classification.gif" width="640"/></p>

## Table of Contents

1. [Overview](#overview)
2. [Pipeline](#pipeline)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [Stages in Detail](#stages-in-detail)
6. [Repository Layout](#repository-layout)
7. [Sample Results](#sample-results)
8. [Disclaimers and Attribution](#disclaimers-and-attribution)

## Overview

A custom YOLOv4 detector trained on aerial imagery feeds vehicle
bounding boxes into a DeepSORT tracker. The resulting per-frame
tracklets are used to build a feature table (position and heading) over
~1000 frames, which is then clustered with Spectral Clustering to
discover lane structure without manual labels. An SVM with an RBF
kernel is trained on the clustered labels and used at inference time to
assign every new vehicle to a lane. The runtime loop in `main.py`
combines all of this with OpenCV overlays for live counting and speed
reporting.

## Pipeline

```
video -> YOLOv4 (TF SavedModel) -> DeepSORT tracker -> data.pkl
        -> feature build (x, y, angle) -> Spectral clustering
        -> SVM (RBF) trained on cluster labels -> classifier.pkl
        -> main.py: detect + track + classify + overlay
```

## Installation

Tested on Ubuntu 20.04 with Python 3.8.

```bash
pip3 install -r requirements.txt           # CPU
# or
pip3 install -r requirements-gpu.txt       # GPU (TensorFlow GPU build)
```

YOLOv4 weights are not in the repo. Drop your `.weights` file in
`./data/yolov4.weights` (or update `save_model.py`'s `--weights` flag).

## Quickstart

The pipeline runs in four stages. Steps 1 to 3 are one-off training
steps; step 4 is the real-time loop.

```bash
# 1. Convert YOLOv4 .weights to a TF SavedModel
python3 save_model.py --model yolov4

# 2. Run YOLO + DeepSORT on a video to produce data.pkl
python3 object_tracker.py --video ./data/video/test.mp4

# 3. Cluster lane structure and train the SVM lane classifier
python3 line_detection.py        # writes 4_Spectral_Clusters_Cars.csv
python3 line_classification.py   # writes classifier.pkl + scaler_loc.pkl

# 4. Run the real-time analyzer
python3 main.py --video ./example_2.mp4 --output ./out.mp4
```

## Stages in Detail

### 1. Vehicle detection and tracking

<p align="center"><img src="data/helpers/tracking.gif" width="560"/></p>


YOLOv4 (custom-trained on aerial vehicle imagery) produces
bounding boxes; DeepSORT (Kalman filter plus appearance metric)
links them into stable tracks. Tracker output is dumped to `data.pkl`
as `{frame_id: {track_id: {x, y}}}`.

### 2. Lane discovery via clustering

For each tracked vehicle we compute heading angle from its position
delta over `dist` frames. Features `(x, y, angle)` are standardized and
fed to five clustering algorithms (KMeans, Hierarchical, Birch,
MiniBatchKMeans, Spectral). Silhouette score on the sample dataset:

| Algorithm        | Silhouette |
| ---------------- | ---------- |
| KMeans           | 0.411      |
| Agglomerative    | 0.408      |
| Birch            | 0.409      |
| MiniBatchKMeans  | 0.369      |
| **Spectral**     | **0.432**  |

Spectral wins, so its labels become the pseudo-ground-truth for stage 3.

### 3. SVM lane classifier

An SVM with an RBF kernel is trained on the Spectral cluster labels.
This generalizes the clustering boundary so a single vehicle position at
inference time can be assigned to a lane without re-running the full
clustering pass. Model is persisted to `classifier.pkl` along with the
fitted `StandardScaler` for `(x, y)`.

### 4. Real-time analysis

`main.py` runs the full loop: YOLOv4 detection, DeepSORT tracking,
SVM lane assignment for every confirmed track, then per-lane
aggregations (count, average speed in pixels per second, immovable
vehicles). Results are rendered as OpenCV overlays on the video and
written to disk if `--output` is set.

## Repository Layout

```
.
|- core/                  YOLOv4 model + utils (forked, see attribution)
|- deep_sort/             DeepSORT tracker (forked, see attribution)
|- tools/                 DeepSORT feature encoder (forked)
|- model_data/            mars-small128.pb (DeepSORT appearance net)
|- data/                  helpers, anchors, classes, sample video
|- sample_csv/            example outputs of the clustering stage
|- save_model.py          .weights -> TF SavedModel
|- object_tracker.py      stage 1 (detection + tracking, dumps data.pkl)
|- line_detection.py      stage 2 (clustering)
|- line_classification.py stage 3 (SVM lane classifier)
|- main.py                stage 4 (real-time analyzer)
```

## Sample Results

Lane structure from Spectral Clustering:

<p align="center"><img src="data/helpers/spec_4.png" width="480"/></p>

SVM smoothing of the same lane structure:

<p align="center"><img src="data/helpers/spec_4_svm.png" width="480"/></p>

## Disclaimers and Attribution

This repository combines third-party components with original work.

**Third-party components:**

- `deep_sort/` is forked from
  [nwojke/deep_sort](https://github.com/nwojke/deep_sort) (Wojke,
  Bewley, Paulus, 2017). No structural changes; used as is for
  multi-target tracking with Kalman filter and deep appearance metric.
- `core/`, `tools/`, `save_model.py`, `convert_tflite.py`,
  `convert_trt.py`, and `object_tracker.py` are based on
  [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite),
  which provides the TensorFlow 2 YOLOv4 implementation and the
  YOLO-plus-DeepSORT scaffolding. The YOLOv4 weights are custom-trained
  on aerial vehicle imagery for this project; the inference and tracking
  glue code is reused from the upstream repo.

**Original contributions:**

- Custom YOLOv4 weights trained on aerial vehicle imagery.
- `line_detection.py`: lane discovery framed as an unsupervised
  clustering problem on `(x, y, heading)` features over ~1000 frames,
  with a five-algorithm comparison and Spectral Clustering selected by
  silhouette score.
- `line_classification.py`: SVM (RBF kernel) trained on Spectral cluster
  labels to produce a smooth, persistable lane classifier for inference.
- `main.py`: the real-time analysis loop that joins YOLOv4, DeepSORT,
  and the SVM lane classifier into a per-lane count, speed, and
  immovable-vehicle reporting pipeline with OpenCV overlays.
