"""GradHD / RegHD time series forecasting models.

A single `MultiModel` covers:
  - RegHD  -> manual delta-rule training (use_backprop=False)
  - GradHD -> backpropagation training, Adam + MSE (use_backprop=True)
  - single model (models=1) or multi-cluster (models>1, softmax weighted)
  - `nonlinear` encoder (random projection cos/sin) or `temporal` (Level + bundle_sequence)
  - single-pass online regime (opt.online=1) or offline multi-epoch (opt.online=0)

Evaluation uses the normalized [0,1] scale against the clean targets, reporting MAE and RMSE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchhd import embeddings, bundle_sequence


# ===================== ENCODERS =====================
class NonlinearEncoder(nn.Module):
    """Random projection: enc = sum_size cos(Wx + b) * sin(Wx).  [B,size] -> [B,d]."""

    def __init__(self, size, d):
        super().__init__()
        w = torch.empty(d, size)
        nn.init.normal_(w, 0, 1)
        w = F.normalize(w, dim=1)
        self.register_buffer("weight", w)
        b = torch.empty(d, size).uniform_(0, 2 * math.pi)
        self.register_buffer("bias", b)

    def configure(self, low, high):  # no-op, keeps a uniform interface
        return self

    def forward(self, x):  # x: [B, size]
        enc = x.unsqueeze(1) * self.weight.unsqueeze(0)          # [B, d, size]
        enc = torch.cos(enc + self.bias.unsqueeze(0)) * torch.sin(enc)
        return enc.sum(dim=2)                                    # [B, d]


class TemporalEncoder(nn.Module):
    """Level embedding per timestep + bundle_sequence (order preserving). [B,size] -> [B,d]."""

    def __init__(self, size, d, levels):
        super().__init__()
        self.size = size
        self.d = d
        self.levels = levels
        self.low, self.high = 0.0, 1.0
        self._build()

    def _build(self):
        self.value = embeddings.Level(self.levels, self.d, low=self.low, high=self.high)
        self.value.weight.requires_grad_(False)

    def configure(self, low, high):
        self.low = float(low)
        self.high = float(high) if float(high) > float(low) else float(low) + 1.0
        self._build()
        return self

    def forward(self, x):  # x: [B, size]
        x = torch.clamp(x, self.low, self.high)
        hvs = self.value(x)                 # [B, size, d]
        seq = bundle_sequence(hvs)          # [B, d]
        return seq.as_subclass(torch.Tensor)


# ===================== MULTIMODEL (GradHD / RegHD regression head) =====================
class MultiModel(nn.Module):
    def __init__(self, size, d, models, opt, dev, encoder_type="nonlinear", use_backprop=False):
        super().__init__()
        self.size = size
        self.d = d
        self.k = models
        self.opt = opt
        self.dev = torch.device(dev)
        self.lr = float(opt.learning_rate)
        self.use_backprop = use_backprop
        self.encoder_type = encoder_type
        self.batch_size = int(getattr(opt, "batch_size", 64))

        if encoder_type == "temporal":
            self.encoder = TemporalEncoder(size, d, int(getattr(opt, "levels", 100)))
        else:
            self.encoder = NonlinearEncoder(size, d)

        # cluster and regression hypervectors, both learnable (clusters only used when k>1)
        clusters = F.normalize(torch.randn(self.k, d), p=2, dim=1)
        self.clusters = nn.Parameter(clusters)
        self.M = nn.Parameter(torch.zeros(self.k, d))

        self.opt_adam = None  # built in setup_optimizer(), after moving to device

        # ===== UKF front-end (GradHD+UKF): sigma-points on the input =====
        self.ukf = bool(getattr(opt, "ukf", False))
        if self.ukf:
            p = size
            alpha_ut, beta, kappa = 1e-3, 2.0, 0.0
            lam = alpha_ut ** 2 * (p + kappa) - p
            self.gamma_sp = (p + lam) ** 0.5
            wm = torch.full((2 * p + 1,), 1.0 / (2 * (p + lam)))
            wc = wm.clone()
            wm[0] = lam / (p + lam)
            wc[0] = lam / (p + lam) + (1 - alpha_ut ** 2 + beta)
            self.register_buffer("ut_wm", wm)
            self.register_buffer("ut_wc", wc)
            offsets = torch.cat([torch.zeros(1, p), torch.eye(p), -torch.eye(p)], dim=0)  # [2p+1, p]
            self.register_buffer("ut_offsets", offsets)

    # ---------- optimizer (backprop) ----------
    def setup_optimizer(self):
        if self.use_backprop:
            params = [p for p in self.parameters() if p.requires_grad]
            self.opt_adam = torch.optim.Adam(params, lr=self.lr)

    # ---------- encode / head ----------
    def encode(self, x):  # [B, size] -> [B, d]
        return self.encoder(x)

    def head(self, enc):  # [B, d] -> [B, 1]
        enc = F.normalize(enc, p=2, dim=1)
        if self.k == 1:
            return (enc * self.M[0]).sum(dim=1, keepdim=True)
        sims = enc @ F.normalize(self.clusters, p=2, dim=1).t()   # [B, k]
        confs = F.softmax(sims, dim=-1)
        dot = enc @ self.M.t()                                    # [B, k]
        return (confs * dot).sum(dim=1, keepdim=True)

    def forward(self, x):
        return self.head(self.encode(x))

    # ---------- UKF front-end: unscented encoding ----------
    def encode_unscented(self, X):
        """X: [B, size] -> (S_bar [B,d] denoised encoding, Pyy [B] innovation variance).

        Draws 2p+1 sigma-points per window (spread ~ sqrt(var(x))), pushes them through the
        nonlinear encoder and aggregates them, so the result accounts for the encoder nonlinearity.
        """
        B = X.shape[0]
        p = self.size
        var = X.var(dim=1)                                   # [B]  (R)
        spread = self.gamma_sp * torch.sqrt(torch.clamp(var, min=0.0) + 1e-12)  # [B]
        # sigma-points: [B, 2p+1, size]
        pts = X.unsqueeze(1) + spread.view(B, 1, 1) * self.ut_offsets.unsqueeze(0)
        Phi = self.encode(pts.reshape(B * (2 * p + 1), p)).reshape(B, 2 * p + 1, -1)  # [B,2p+1,d]
        S_bar = (self.ut_wm.view(1, -1, 1) * Phi).sum(dim=1)  # [B, d]
        # variance from the sigma-point predictions
        with torch.no_grad():
            Y = self.head(Phi.reshape(B * (2 * p + 1), -1)).reshape(B, 2 * p + 1)  # [B,2p+1]
            yhat = (self.ut_wm * Y).sum(dim=1, keepdim=True)  # [B,1]
            Pyy = (self.ut_wc * (Y - yhat) ** 2).sum(dim=1) + var * self.d + 1e-6  # [B]
        return S_bar, Pyy

    def predict(self, X):  # uses the UKF front-end when enabled
        if self.ukf:
            S_bar, _ = self.encode_unscented(X)
            return self.head(S_bar)
        return self.head(self.encode(X))

    # ---------- manual update (RegHD delta rule) ----------
    def manual_update(self, enc, y):  # enc: [B,d], y: [B,1]
        enc = F.normalize(enc, p=2, dim=1)
        b = enc.shape[0]
        if self.k == 1:
            pred = (enc * self.M[0]).sum(dim=1, keepdim=True)
            err = y - pred
            self.M.data[0] += self.lr * (err * enc).mean(dim=0)
            return
        sims = enc @ F.normalize(self.clusters, p=2, dim=1).t()
        confs = F.softmax(sims, dim=-1)
        dot = enc @ self.M.t()
        pred = (confs * dot).sum(dim=1, keepdim=True)
        err = y - pred
        self.M.data += self.lr * ((err * confs).t() @ enc) / b
        self.clusters.data += ((1.0 - confs).t() @ enc) * (self.lr / b)
        self.clusters.data = F.normalize(self.clusters.data, p=2, dim=1)

    # ---------- sliding windows ----------
    def _build_windows(self, sets, matrix_t):
        """matrix_t: [n_series, T] -> X [N, size], Y [N] (aligned with sets)."""
        sets_idx = torch.as_tensor(list(sets), dtype=torch.long, device=self.dev)
        Xs, Ys = [], []
        for n in range(matrix_t.shape[0]):
            s = matrix_t[n]
            w = s.unfold(0, self.size, 1)
            Xs.append(w[sets_idx])
            Ys.append(s[sets_idx + self.size])
        return torch.cat(Xs, 0), torch.cat(Ys, 0)

    # ---------- training ----------
    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):
        mt = torch.as_tensor(matrix_1_norm, dtype=torch.float32, device=self.dev)
        X, Y = self._build_windows(sets_training, mt)

        if self.encoder_type == "temporal":
            self.encoder.configure(float(X.min()), float(X.max()))
            self.encoder.to(self.dev)

        self.setup_optimizer()

        online = bool(getattr(self.opt, "online", 1))
        batch = 1 if online else self.batch_size
        n_epochs = 1 if online else int(epochs)
        N = X.shape[0]

        for _ in range(n_epochs):
            order = torch.arange(N, device=self.dev) if online else torch.randperm(N, device=self.dev)
            for start in range(0, N, batch):
                idx = order[start:start + batch]
                xb = X[idx]
                yb = Y[idx].unsqueeze(1)
                if self.use_backprop:
                    self.opt_adam.zero_grad()
                    if self.ukf:
                        S_bar, Pyy = self.encode_unscented(xb)
                        pred = self.head(S_bar)
                        loss = ((pred - yb) ** 2 / Pyy.unsqueeze(1)).mean()  # innovation weighted loss
                    else:
                        pred = self.head(self.encode(xb))
                        loss = F.mse_loss(pred, yb)
                    loss.backward()
                    self.opt_adam.step()
                    if self.k > 1:
                        with torch.no_grad():
                            self.clusters.data = F.normalize(self.clusters.data, p=2, dim=1)
                else:
                    with torch.no_grad():
                        enc = self.encode_unscented(xb)[0] if self.ukf else self.encode(xb)
                        self.manual_update(enc, yb)

    # ---------- testing ----------
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False):
        mt = torch.as_tensor(matrix_1_norm, dtype=torch.float32, device=self.dev)
        mo = torch.as_tensor(matrix_1_norm_org, dtype=torch.float32, device=self.dev)
        Xt, _ = self._build_windows(sets_testing, mt)
        _, Yt = self._build_windows(sets_testing, mo)   # clean targets

        preds = []
        bs = 256
        with torch.no_grad():
            for start in range(0, Xt.shape[0], bs):
                preds.append(self.predict(Xt[start:start + bs]).squeeze(1))
        preds = torch.cat(preds, 0)

        mae = float((preds - Yt).abs().mean())
        rmse = float(torch.sqrt(((preds - Yt) ** 2).mean()))
        self.last_mae, self.last_rmse = mae, rmse

        if not cv:
            print(f"Testing MAE {mae:.4f} | RMSE {rmse:.4f}")
            return mae
        return preds.cpu().tolist(), Yt.cpu().tolist()


def Return_Model(size, d, models, number_ts, opt, dev):
    torch.random.manual_seed(opt.trial)
    model = MultiModel(
        size, d, models, opt, dev,
        encoder_type=getattr(opt, "encoder", "nonlinear"),
        use_backprop=bool(getattr(opt, "use_backprop", False)),
    ).to(dev)
    return model
