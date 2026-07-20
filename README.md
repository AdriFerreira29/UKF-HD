# UKF-HD

Code for the paper **"UKF-HD: Unscented Kalman Filtering in Hyperdimensional Computing for Robust
On-Device IoT Forecasting"** — Adriano Ferreira, Leandro Santiago (Institute of Computing, UFF).

Built on top of the [KalmanHD](https://github.com/DarthIV02/KalmanHD) codebase
(Gomez, Yu and Rosing, ASP-DAC 2024).

## Contributions

- **UKF-HD** — an Unscented Kalman Filter gain inside HDC. The HDC prediction
  `y = phi(x) . alpha` is linear in the weight hypervector, so the nonlinearity worth capturing is
  the one in the encoder. The unscented transform is therefore applied to sigma-points on the
  *input window*, propagated through the nonlinear encoder. This replaces the `D x D` weight
  covariance with a `p x p` diagonal uncertainty over the input.
- **GradHD** — a gradient-trained HDC forecaster (Adam + MSE) whose cluster and regression
  hypervectors are learned end to end, with softmax-weighted contributions from all clusters.
- **GradHD+UKF** — GradHD fed by the denoised unscented encoding and trained with a
  heteroscedastic loss weighted by the innovation variance.

## Layout

```
KalmanHD-Final/        main codebase
  main.py              entry point
  models/              UKFHD, GradHD, and baselines (RegHD, KalmanHD, DNN, VAE, KalmanFilter)
  preprocessed_data/   time series datasets
  scripts/             experiment sweeps
  Stuff/               dataset loader
data/                  regression datasets
requirements.txt
```

## Setup

Tested with Python 3.13.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The `--extra-index-url` is needed because the pinned torch build is the CPU wheel (`+cpu`).

## Running

All commands are run from inside `KalmanHD-Final/`.

```bash
# UKF-HD (proposed), single-pass online
python main.py --model UKFHD --ukf_mode ukf --dataset MetroInterstateTrafficVolume --online 1

# Linear Kalman gain (KalmanHD-KF ablation, same encoder and harness)
python main.py --model UKFHD --ukf_mode kf --dataset MetroInterstateTrafficVolume --online 1

# GradHD (offline, gradient based)
python main.py --model GradHD --use_backprop --online 0 --epochs 30 --models 8 \
               --dataset MetroInterstateTrafficVolume

# GradHD+UKF
python main.py --model GradHD_UKF --use_backprop --online 0 --epochs 30 \
               --dataset MetroInterstateTrafficVolume
```

Sensor noise is injected into the input stream while targets stay clean:

| Flag | Meaning |
| --- | --- |
| `--gaussian_noise` | standard deviation of additive Gaussian noise |
| `--poisson_noise` | lambda of the Poisson noise |
| `--p` | probability of a missing-value segment (length `--s`) |

Full experiment sweep and aggregation:

```bash
python scripts/run_paper.py --full
python scripts/aggregate_paper.py
```

## Datasets

`SanFranciscoTraffic`, `MetroInterstateTrafficVolume`, `GuangzhouTraffic`,
`EnergyConsumptionFraunhofer`, `ElectricityLoadDiagrams`.

Each series is min-max normalized to `[0,1]`, windowed with `--size_of_sample` (default 20) and
split 70% training / 20% testing / 10% cross-validation. Accuracy is reported as MAE (and RMSE)
against the clean signal.
