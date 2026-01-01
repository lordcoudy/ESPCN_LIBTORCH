# ESPCN_LIBTORCH

LibTorch/C++ implementation of the ESPCN_PYTORCH project (training, tuning, evaluation).

## Prerequisites
- CMake >= 3.22
- A C++17 compiler
- LibTorch (PyTorch C++ distribution)
- OpenCV (for image loading + Lanczos resize + augmentation)
- Optional: CUDA-enabled LibTorch for GPU training

## Build (macOS/Linux)
### Install OpenCV (macOS)
```bash
brew install opencv
```

1. Download LibTorch from https://pytorch.org (C++ distribution).
2. Configure:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/absolute/path/to/libtorch"
```

3. Build:

```bash
cmake --build build -j
```

## Dataset
This project expects the BSDS500 directory layout matching the Python repo:

```
dataset/BSDS500/images/{train,val,test}/*.jpg|png
```

You already have it in your existing `ESPCN_PYTORCH/dataset` folder; you can symlink:

```bash
ln -s ../ESPCN_PYTORCH/dataset ./dataset
```

## Usage
The main binary is `espcn_libtorch`.

### Train
```bash
./build/espcn_libtorch train \
  --dataset-root ./dataset/BSDS500/images \
  --upscale 2 \
  --epochs 5 \
  --batch 16 \
  --lr 1e-3 \
  --weight-decay 5e-5 \
  --separable 1 \
  --channels-last 1
```

### Train with OneCycleLR (cosine)
```bash
./build/espcn_libtorch train \
  --dataset-root ./dataset/BSDS500/images \
  --upscale 2 \
  --epochs 5 \
  --batch 16 \
  --lr 1e-3 \
  --scheduler 1 \
  --onecycle-pct-start 0.3 \
  --onecycle-div-factor 25 \
  --onecycle-final-div-factor 10000
```

### CUDA AMP
- Enabled only when `--cuda 1` and the LibTorch build has CUDA support.
- Turn on with `--mixed-precision 1`.

### Evaluate (PSNR)
```bash
./build/espcn_libtorch eval \
  --dataset-root ./dataset/BSDS500/images \
  --upscale 2 \
  --batch 8
```

### Tune (random search)
```bash
./build/espcn_libtorch tune \
  --dataset-root ./dataset/BSDS500/images \
  --upscale 2 \
  --trials 30
```

## Notes
- Mixed precision + GradScaler is enabled only on CUDA builds.
- On macOS CPU builds, training works but will be slower than the Python MPS path.
