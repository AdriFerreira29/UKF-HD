"""LSTM / Bidirectional-LSTM baseline for single-step time series forecasting (PyTorch).

Windowed regression, same protocol as the HDC models: a window of `size` past values feeds
an LSTM whose last hidden state is mapped to the next value. Trained on the (noisy) input
stream, evaluated against the clean target. Metric: MAE (main) and RMSE.

BLSTM is the same module with a bidirectional LSTM (Return_Model reads opt.model). Training
follows the GradHD regime: single-pass online (opt.online=1) or offline multi-epoch
(opt.online=0, uses opt.epochs and opt.batch_size).

Interface compatible with main.py: Return_Model(...) -> model.train(...) / model.test(...).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMNet(nn.Module):
    def __init__(self, hidden, layers, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):              # x: [B, size, 1]
        out, _ = self.lstm(x)         # [B, size, out_dim]
        return self.fc(out[:, -1, :])  # last timestep -> [B, 1]


class LSTM_Model:
    def __init__(self, size, opt, dev, bidirectional=False):
        self.size = size
        self.opt = opt
        self.dev = torch.device(dev)
        self.lr = float(opt.learning_rate)
        self.batch_size = int(getattr(opt, "batch_size", 64))
        self.net = LSTMNet(
            hidden=int(getattr(opt, "lstm_hidden", 64)),
            layers=int(getattr(opt, "lstm_layers", 1)),
            bidirectional=bidirectional,
        ).to(self.dev)
        self.last_mae = ""
        self.last_rmse = ""

    # ---------- sliding windows over a [n_series, T] matrix ----------
    def _build_windows(self, sets, matrix_np):
        mt = torch.as_tensor(matrix_np, dtype=torch.float32, device=self.dev)
        sets_idx = torch.as_tensor(list(sets), dtype=torch.long, device=self.dev)
        Xs, Ys = [], []
        for n in range(mt.shape[0]):
            s = mt[n]
            w = s.unfold(0, self.size, 1)      # [T-size+1, size]
            Xs.append(w[sets_idx])
            Ys.append(s[sets_idx + self.size])
        return torch.cat(Xs, 0), torch.cat(Ys, 0)

    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):
        X, Y = self._build_windows(sets_training, matrix_1_norm)   # noisy input + noisy target
        opt_adam = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        online = bool(getattr(self.opt, "online", 1))
        batch = 1 if online else self.batch_size
        n_epochs = 1 if online else int(epochs)
        N = X.shape[0]

        self.net.train()
        for _ in range(n_epochs):
            order = torch.arange(N, device=self.dev) if online else torch.randperm(N, device=self.dev)
            for start in range(0, N, batch):
                idx = order[start:start + batch]
                xb = X[idx].unsqueeze(-1)              # [b, size, 1]
                yb = Y[idx].unsqueeze(1)               # [b, 1]
                opt_adam.zero_grad()
                loss = F.mse_loss(self.net(xb), yb)
                loss.backward()
                opt_adam.step()

    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False):
        Xt, _ = self._build_windows(sets_testing, matrix_1_norm)       # noisy input
        _, Yt = self._build_windows(sets_testing, matrix_1_norm_org)   # clean target

        preds = []
        bs = 256
        self.net.eval()
        with torch.no_grad():
            for start in range(0, Xt.shape[0], bs):
                xb = Xt[start:start + bs].unsqueeze(-1)
                preds.append(self.net(xb).squeeze(1))
        preds = torch.cat(preds, 0)

        mae = float((preds - Yt).abs().mean())
        rmse = float(torch.sqrt(((preds - Yt) ** 2).mean()))
        self.last_mae, self.last_rmse = mae, rmse
        if not cv:
            print(f"Testing MAE {mae:.4f} | RMSE {rmse:.4f}")
            return mae
        return preds.cpu().tolist(), Yt.cpu().tolist()


def Return_Model(size, d, models, number_ts, opt, dev):
    torch.random.manual_seed(int(getattr(opt, "trial", 0)))
    bidirectional = (getattr(opt, "model", "LSTM") == "BLSTM")
    return LSTM_Model(size, opt, dev, bidirectional=bidirectional)
