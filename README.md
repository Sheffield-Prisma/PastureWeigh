# PastureWeigh

PastureWeigh estimates cattle live weight from full-resolution instance masks using torso geometry and lightweight neural regression. This repository packages the method in `pastureweigh_rgb`, including feature extraction, model definitions, identity-disjoint cross-validation, and inference.

The regressor consumes mask-derived measurements, not RGB tensors. Instance segmentation and image acquisition are upstream steps. No datasets, experimental results, pretrained weights, or checkpoints are distributed here.

## Installation

Use Python 3.10 or newer:

```bash
git clone git@github.com:Sheffield-Prisma/PastureWeigh.git
cd PastureWeigh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Runtime dependencies are NumPy, OpenCV, and PyTorch. For GPU training, install the appropriate CUDA-enabled PyTorch build for your system before installing this package. CPU execution is supported with `--device cpu`.

## Method

1. Fill holes and retain the largest connected foreground component.
2. Compute a distance transform, retain its core at `tau * maximum_distance` (default `tau = 0.25`), and fit an ellipse to the core contour.
3. Expand the ellipse semi-axes by the core threshold radius and intersect it with the cleaned mask to select the torso.
4. Measure either a longitudinal width profile or torso length and ordered segment widths, keeping their original pixel units.
5. Standardize features using training observations only, then regress log weight with a small neural network and Smooth L1 loss.

The default `torso_m4` model samples 32 widths in each of four longitudinal segments. A shared two-layer 1D convolutional encoder processes each segment; an ordered regression head predicts log weight. The `morph_m4` alternative uses `[length, width_1, ..., width_4]` and a small multilayer perceptron.

For directional features, training randomly reverses the longitudinal order. Inference averages the forward and reversed predictions **after** exponentiating to kilograms. Morphometry reversal preserves length and reverses only the segment widths.

Widths are not normalized by each animal's size, and masks are not resized. Pixel scale therefore affects predictions: use consistent acquisition scale and evaluate performance when camera geometry or resolution changes. The method does not perform camera calibration or infer metric dimensions.

### Available configurations

| Experiment | Features | Regressor |
| --- | --- | --- |
| `torso_m1`, `torso_m4`, `torso_m8`, `torso_m16` | Torso width profiles with 1, 4, 8, or 16 segments | Shared 1D encoder and ordered head |
| `fullmask_m4` | Four-segment full-mask width profile in the fitted ellipse coordinates | Shared 1D encoder and ordered head |
| `morph_m1`, `morph_m2`, `morph_m4`, `morph_m8` | Torso length and 1, 2, 4, or 8 segment widths | MLP |
| `torso_lw` | Torso length and width | MLP |
| `bbox` | Axis-aligned bounding-box width and height | MLP |

The training CLI fixes width-profile sampling at 32 points per segment. `--experiment` determines the segment count; the default is `torso_m4`. `morph_m1` uses the same geometric features as `torso_lw`.

For morphometry experiments, `--torso-mode pca_peak` selects an alternative torso extractor based on PCA and a peak-width interval. It uses a peak ratio of 0.68, requires at least 0.52 longitudinal retention, and explicitly falls back to the full cleaned mask when extraction is unreliable. The default `--torso-mode legacy` selects the ellipse method described above. The width-bottleneck extractor in `geometry.py` is a diagnostic utility, not a training configuration.

## Input format

Training expects Labelme JSON annotations containing cropped, base64-encoded PNG masks. Polygon annotations are not accepted. Supply your own images, annotations, positive weights in kilograms, and persistent cattle identities.

```text
data/annotations/
  raw-frames/
    session_a/
      frame_000001.jpg
      frame_000001.json
  split/                    # Created by pastureweigh-split
```

Each annotation declares `imageHeight`, `imageWidth`, `imagePath` (relative to the JSON file), and a `shapes` list. Each cattle shape contains:

| Field | Requirement |
| --- | --- |
| `label` | `"cattle"` |
| `shape_type` | `"mask"` |
| `group_id` | Persistent integer or string cattle identity, consistent across all images |
| `weight` | Positive finite weight in kilograms, identical for every observation of that identity |
| `points` | `[[x0, y0], [x1, y1]]`, integer **inclusive** corners inside the image |
| `mask` | Base64 PNG crop, single channel, binary values 0/1 or 0/255 |

The PNG crop must have shape `(y1 - y0 + 1, x1 - x0 + 1)`. It is placed back into the full-resolution image coordinates without resizing. Images must exist for training data validation; their pixels are read only when debug overlays are requested.

## Training and evaluation

Create deterministic five-fold splits with at least five distinct cattle identities:

```bash
pastureweigh-split --annotations-root data/annotations --seed 42
```

The default source is `<annotations-root>/raw-frames`; use `--source` to select another annotation directory within the annotations root. `--output-dir` changes the split destination, which must be outside the source and empty. Split files contain annotation paths relative to the annotations root, shape indices, identities, and weights.

Validate the data, geometry, forward/backward pass, and aggregation before training:

```bash
pastureweigh-train \
  --annotations-root data/annotations \
  --experiment torso_m4 \
  --device cpu \
  --smoke-test
```

The smoke test performs one optimizer step but does not run cross-validation or save a model. Run all five outer folds with:

```bash
pastureweigh-train \
  --annotations-root data/annotations \
  --experiment torso_m4 \
  --fold all \
  --seed 42 \
  --output-dir outputs/torso_m4_seed42
```

For the segment-morphometry model:

```bash
pastureweigh-train \
  --annotations-root data/annotations \
  --experiment morph_m4 \
  --torso-mode legacy \
  --output-dir outputs/morph_m4_seed42
```

Use `--split-dir` for an existing compatible split directory. Other controls include `--fold 1`, `--max-epochs`, `--patience`, `--batch-size`, `--lr`, and `--device`. Run any command with `--help` for all options. Default output paths are relative to the current working directory, and existing fold outputs are not overwritten.

The split field named `val` is the **outer held-out test partition**. Each outer fold selects its training duration using a separate identity-disjoint inner validation split of approximately 20% of the outer training identities. It then refits a fresh model on all outer training identities for that duration. Normalization statistics are recomputed from the relevant training partition. Outer test identities are excluded from fitting and epoch selection.

Evaluation reports MAE, RMSE, and MAPE at three levels:

- **Observation:** one prediction per annotated instance.
- **Cattle:** the median prediction across observations of each held-out identity.
- **Herd:** the sum of predictions for held-out instances in each annotated image. When an image also contains training identities, the reported total covers only its held-out subset.

Local training creates a checkpoint, normalization statistics, configuration, and evaluation files under the selected output directory. These generated files are excluded from version control. Five-fold summaries report the mean and sample standard deviation across folds.

To run and summarize the default ellipse-based configurations:

```bash
pastureweigh-train --annotations-root data/annotations --experiment all --output-dir outputs
pastureweigh-summarize --outputs-root outputs --seed 42
```

The summarizer reads completed five-fold runs in `<experiment>_seed<seed>` directories. Run PCA configurations individually and read their own summaries.

## Inference

Train a model locally, then pass full-resolution binary mask images to the predictor:

```bash
pastureweigh-predict \
  --checkpoint outputs/torso_m4_seed42/fold_1/model.pt \
  --device cpu \
  /path/to/cattle_mask.png
```

The command prints JSON with a weight in kilograms for each mask. Masks must be single-channel binary images at the same pixel scale as training. The checkpoint stores the feature configuration, normalization statistics, and model weights together. No pretrained checkpoint is included.

The Python API also accepts a two-dimensional NumPy mask:

```python
import cv2
from pastureweigh_rgb.inference import Predictor

predictor = Predictor("outputs/torso_m4_seed42/fold_1/model.pt", device="cpu")
mask = cv2.imread("/path/to/cattle_mask.png", cv2.IMREAD_GRAYSCALE)
weight_kg = predictor.predict(mask)
```

For feature extraction without training:

```python
from pastureweigh_rgb.geometry import extract_morphometry, extract_profile

profile, diagnostics, _ = extract_profile(mask, segments=4, samples_per_segment=32)
measurements, diagnostics = extract_morphometry(mask, segments=4)
```

## Repository layout

```text
src/pastureweigh_rgb/
  dataset.py                 Labelme cropped-mask decoding and validation
  geometry.py                Ellipse-based torso geometry and feature extraction
  pca_torso.py               Optional PCA/peak-width torso extraction
  model.py                   Profile, morphometry, and bounding-box regressors
  splits.py                  Identity-disjoint five-fold split generation
  train_test.py              Nested epoch selection, refitting, and evaluation
  inference.py               Checkpoint loading and mask-to-weight prediction
  summarize_experiments.py   Local cross-validation summary utility
tests/
  test_method.py             Synthetic-data validation and integration tests
```

## Validation

After installation, run:

```bash
python -m unittest discover -s tests -v
```

Tests generate their own masks and annotations in temporary directories. They check decoding, geometry, directional invariance, identity separation, checkpoint round trips, and the training/inference workflow. They do not require or distribute research data or trained artifacts.
