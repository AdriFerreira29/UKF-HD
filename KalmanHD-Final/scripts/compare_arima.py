"""Rolling one-step-ahead ARIMA / SARIMA baselines (statsmodels), same eval protocol as main.py.

Classical models want history, not a fixed window, so they run separately from the HDC harness:
for each series we fit once on the training history and then walk the test region forecasting one
step at a time, appending each true (noisy) observation without refitting (Kalman filter update).
Predictions are scored against the CLEAN target, MAE and RMSE, exactly like the other models.

Cost is dominated by the initial fit per series, so --max-series caps how many series are used
(the multi-series datasets have hundreds). Noise injection replicates main.py.

Usage (run from the KalmanHD-Final directory):
  python scripts/compare_arima.py --dataset MetroInterstateTrafficVolume --model arima
  python scripts/compare_arima.py --dataset MetroInterstateTrafficVolume --model sarima --gaussian_noise 0.5
"""
import os
import sys
import csv
import random
import argparse
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)  # DatasetLoader uses relative paths

import logging
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from codecarbon import OfflineEmissionsTracker
from Stuff.DatasetLoader import DatasetLoader

warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").setLevel(logging.ERROR)

SIZE = 20  # matches --size_of_sample default, only used to align the train/test split with main.py

# seasonal period per dataset (hourly -> 24, daily -> 7; SFT is weekly, short fallback)
SEASONAL_S = {
    "MetroInterstateTrafficVolume": 24,
    "GuangzhouTraffic": 24,
    "SanFranciscoTraffic": 7,
    "EnergyConsumptionFraunhofer": 7,
    "ElectricityLoadDiagrams": 7,
}


def make_noisy(clean, trial, g=0.0, p=0.0, s=5, pois=0.0):
    """Replicates main.py: missing segments, then additive Gaussian, then Poisson. Seeded by trial."""
    random.seed(trial)
    np.random.seed(trial)
    m = np.copy(clean)
    if p > 0:
        for i in range(0, m.shape[1], s):
            for j in range(m.shape[0]):
                if random.random() < p:
                    m[j, i:i + s] = 0
    m = m + np.random.normal(0, g, size=m.shape)
    if pois > 0:
        m = m + np.random.poisson(lam=pois, size=m.shape)
    return m


def pick_series(n_series, max_series):
    if max_series <= 0 or max_series >= n_series:
        return list(range(n_series))
    # evenly spaced sample so we do not bias toward the first series only
    return list(np.linspace(0, n_series - 1, max_series).astype(int))


def rolling_forecast(noisy_s, clean_s, first, order, seasonal_order):
    """One-step-ahead forecasts over the test region.

    Fit once on history[:first], then filter the whole test block in a single pass
    (append with refit=False) and read the one-step-ahead in-sample predictions. Each
    prediction for position t uses only observations up to t-1, i.e. genuine rolling
    one-step-ahead, but computed vectorized instead of a per-step Python loop.
    """
    hist = noisy_s[:first]
    mod = SARIMAX(hist, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False, maxiter=50)
    T = len(noisy_s)
    res = res.append(noisy_s[first:T], refit=False)          # filter the test block at once
    yhat = np.asarray(res.predict(start=first, end=T - 1))    # one-step-ahead (dynamic=False)
    err = yhat - np.asarray(clean_s[first:T], dtype=float)
    return np.abs(err).tolist(), (err ** 2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MetroInterstateTrafficVolume", choices=list(SEASONAL_S))
    ap.add_argument("--model", default="arima", choices=["arima", "sarima"])
    ap.add_argument("--order", default="2,1,2", help="p,d,q")
    ap.add_argument("--seasonal", default="1,0,1", help="P,D,Q (s comes from the dataset)")
    ap.add_argument("--season_period", type=int, default=0, help="override seasonal s (0 = dataset default)")
    ap.add_argument("--max_series", type=int, default=20, help="cap number of series (cost control)")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--gaussian_noise", type=float, default=0.0)
    ap.add_argument("--poisson_noise", type=float, default=0.0)
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--s", type=int, default=5)
    ap.add_argument("--out", default="results_arima_ext.csv")
    args = ap.parse_args()

    order = tuple(int(v) for v in args.order.split(","))
    if args.model == "sarima":
        s = args.season_period or SEASONAL_S[args.dataset]
        P, D, Q = (int(v) for v in args.seasonal.split(","))
        seasonal_order = (P, D, Q, s)
    else:
        seasonal_order = (0, 0, 0, 0)

    clean = DatasetLoader(args.dataset).dataset_load_and_preprocess("normalized")
    noisy = make_noisy(clean, args.trial, g=args.gaussian_noise, p=args.p, s=args.s, pois=args.poisson_noise)

    T = clean.shape[1]
    i0 = int((T - SIZE) * 0.7)
    first = i0 + SIZE          # first test-target position (aligned with main.py's sets_testing)
    last = int((T - SIZE) * 0.9) + SIZE
    # trim series to the test region end so we only roll over the test targets
    clean = clean[:, :last]
    noisy = noisy[:, :last]

    series_ids = pick_series(clean.shape[0], args.max_series)
    # skip seasonal if history is too short to hold two seasons
    if seasonal_order[3] and first < 2 * seasonal_order[3]:
        print(f"[warn] history {first} < 2*s={2*seasonal_order[3]} -> dropping seasonal component")
        seasonal_order = (0, 0, 0, 0)

    tracker = None
    try:
        tracker = OfflineEmissionsTracker(country_iso_code="BRA", save_to_file=False,
                                          log_level="error", measure_power_secs=1)
        tracker.start()
    except Exception:
        tracker = None

    all_abs, all_sq, ok, fail = [], [], 0, 0
    for n in series_ids:
        try:
            a, sq = rolling_forecast(noisy[n], clean[n], first, order, seasonal_order)
            all_abs.extend(a)
            all_sq.extend(sq)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[skip] series {n}: {type(e).__name__}: {str(e)[:70]}")

    dur = energy = co2 = ""
    if tracker is not None:
        try:
            tracker.stop()
            d = tracker.final_emissions_data
            dur, energy, co2 = d.duration, d.energy_consumed, d.emissions
        except Exception:
            pass

    mae = float(np.mean(all_abs)) if all_abs else float("nan")
    rmse = float(np.sqrt(np.mean(all_sq))) if all_sq else float("nan")
    label = f"{args.model.upper()} order={order}" + (f" seasonal={seasonal_order}" if args.model == "sarima" else "")
    print(f"\n[{args.dataset}] {label} | series ok={ok} fail={fail} | "
          f"g={args.gaussian_noise} p={args.p} pois={args.poisson_noise} trial={args.trial}")
    print(f"Testing MAE {mae:.4f} | RMSE {rmse:.4f}")

    header = ["Dataset", "Model", "order", "seasonal", "Gaussian", "Poisson", "MissingP",
              "trial", "n_series", "device", "MAE", "RMSE", "Dur_s", "Energy_kWh", "CO2_kg"]
    row = [args.dataset, args.model, str(order), str(seasonal_order), args.gaussian_noise,
           args.poisson_noise, args.p, args.trial, ok, "cpu", mae, rmse, dur, energy, co2]
    exists = os.path.isfile(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


if __name__ == "__main__":
    main()
