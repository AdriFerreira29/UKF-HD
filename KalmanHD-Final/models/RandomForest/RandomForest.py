"""Random Forest baseline for single-step time series forecasting.

Windowed regression, same protocol as the HDC models: a window of `size` past values
predicts the next one. Trained on the (noisy) input stream, evaluated against the clean
target (matrix_1_norm_org). Metric: MAE (main) and RMSE.

Interface compatible with main.py: Return_Model(...) -> model.train(...) / model.test(...).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RF_Model:
    def __init__(self, size, opt):
        self.size = size
        self.opt = opt
        self.rf = RandomForestRegressor(
            n_estimators=int(getattr(opt, "rf_estimators", 100)),
            max_depth=getattr(opt, "rf_max_depth", None),
            random_state=int(getattr(opt, "trial", 0)),
            n_jobs=-1,
        )
        self.last_mae = ""
        self.last_rmse = ""

    # ---------- sliding windows over a [n_series, T] matrix ----------
    def _build_windows(self, sets, matrix):
        sets = list(sets)
        Xs, Ys = [], []
        for n in range(matrix.shape[0]):
            s = matrix[n]
            Xs.append(np.stack([s[i:i + self.size] for i in sets]))
            Ys.append(np.array([s[i + self.size] for i in sets]))
        return np.concatenate(Xs, 0), np.concatenate(Ys, 0)

    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):
        X, Y = self._build_windows(sets_training, matrix_1_norm)   # noisy input + noisy target
        self.rf.fit(X, Y)

    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False):
        Xt, _ = self._build_windows(sets_testing, matrix_1_norm)       # noisy input
        _, Yt = self._build_windows(sets_testing, matrix_1_norm_org)   # clean target
        pred = self.rf.predict(Xt)
        mae = float(np.abs(pred - Yt).mean())
        rmse = float(np.sqrt(((pred - Yt) ** 2).mean()))
        self.last_mae, self.last_rmse = mae, rmse
        if not cv:
            print(f"Testing MAE {mae:.4f} | RMSE {rmse:.4f}")
            return mae
        return pred.tolist(), Yt.tolist()


def Return_Model(size, d, models, number_ts, opt, dev):
    np.random.seed(int(getattr(opt, "trial", 0)))
    return RF_Model(size, opt)
