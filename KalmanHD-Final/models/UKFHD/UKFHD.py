import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GradHD.GradHD import NonlinearEncoder, TemporalEncoder


class UKF_HD(nn.Module):
    def __init__(self, size, d, models, opt, dev, encoder_type="nonlinear", ukf_mode="ukf"):
        super().__init__()
        self.size = size
        self.d = d
        self.opt = opt
        self.dev = torch.device(dev)
        self.lr = float(opt.learning_rate)
        self.gamma = float(getattr(opt, "alpha", 0.3))   # moving average factor for the variance
        self.mode = ukf_mode
        self.encoder_type = encoder_type

        if encoder_type == "temporal":
            self.encoder = TemporalEncoder(size, d, int(getattr(opt, "levels", 100)))
        else:
            self.encoder = NonlinearEncoder(size, d)

        # state, updated manually without autograd
        self.alpha = torch.zeros(d, device=self.dev)
        self.var = torch.zeros((), device=self.dev)        # MA of window variance (signal+noise)
        self.noise_var = torch.zeros((), device=self.dev)  # MA of high-frequency (noise) variance

        # conditioning hyperparameters (see the UKF-HD improvement plan)
        self.var_floor = float(getattr(opt, "var_floor", 1e-4))    # floor on variance used for the spread
        self.pyy_floor = float(getattr(opt, "pyy_floor", 0.05))    # floor on innovation covariance (gain guard)
        self.meas_scale = float(getattr(opt, "meas_scale", 1.0))   # measurement-noise sensitivity to noise_var
        self.meas_r0 = float(getattr(opt, "meas_r0", 0.0))         # additive measurement-noise floor
        self.a_clip = float(getattr(opt, "a_clip", 1.0))           # innovation clip (anti-divergence)
        self.q_proc = float(getattr(opt, "q_proc", 0.0))           # KF process noise (0 = faithful linear KF)

        # d x d covariance, only needed in KF mode
        self.P = 0.1 * torch.eye(d, device=self.dev) if ukf_mode == "kf" else None

        # Unscented Transform weights (scaled UT; ut_alpha sets the sigma-point spread).
        # Note: ut_alpha must be moderate (~0.3); ut_alpha=1e-3 collapses the sigma-points onto the mean.
        p = size
        alpha_ut = float(getattr(opt, "ut_alpha", 0.3))
        beta, kappa = 2.0, 0.0
        lam = alpha_ut ** 2 * (p + kappa) - p
        self.p = p
        self.lam = lam
        self.gamma_sp = (p + lam) ** 0.5      # sigma-point spread factor
        wm = torch.full((2 * p + 1,), 1.0 / (2 * (p + lam)), device=self.dev)
        wc = wm.clone()
        wm[0] = lam / (p + lam)
        wc[0] = lam / (p + lam) + (1 - alpha_ut ** 2 + beta)
        self.wm = wm
        self.wc = wc

    # ---------- encode ----------
    def encode(self, x):  # [B, size] -> [B, d], normalized
        return F.normalize(self.encoder(x), p=2, dim=1)

    # ---------- sigma-points on the input ----------
    def _sigma_points(self, x):  # x: [size] -> [2p+1, size]; Px = noise_var * I_p (isotropic diagonal)
        spread = self.gamma_sp * torch.sqrt(torch.clamp(self.noise_var, min=self.var_floor))
        pts = x.unsqueeze(0).repeat(2 * self.p + 1, 1)
        eye = torch.eye(self.p, device=self.dev) * spread
        pts[1:self.p + 1] += eye
        pts[self.p + 1:] -= eye
        return pts

    # ---------- per sample update ----------
    def update(self, x, y):
        # moving averages: window variance (signal+noise) and NOISE variance (via lag-1 differences)
        self.var = self.gamma * self.var + (1 - self.gamma) * torch.var(x)
        nv = 0.5 * torch.var(x[1:] - x[:-1]) if x.numel() > 1 else torch.var(x)
        self.noise_var = self.gamma * self.noise_var + (1 - self.gamma) * nv
        # measurement noise from the estimated noise level (not Var(x)*D, which underfit and diverged)
        R = self.meas_scale * self.noise_var + self.meas_r0

        if self.mode == "ukf":
            X = self._sigma_points(x)                 # [2p+1, size]
            Phi = self.encode(X)                      # [2p+1, d]
            Y = Phi @ self.alpha                      # [2p+1]
            yhat = (self.wm * Y).sum()
            Pyy = (self.wc * (Y - yhat) ** 2).sum() + R
            Pyy = torch.clamp(Pyy, min=self.pyy_floor)     # gain guard: Pyy cannot collapse to ~0
            Phibar = (self.wm.unsqueeze(1) * Phi).sum(0)   # [d] denoised unscented encoding
            A = torch.clamp(y - yhat, -self.a_clip, self.a_clip)  # innovation guard
            G = Phibar / Pyy                          # unscented gain
            self.alpha += self.lr * G * A
        else:  # standard KF with a d x d covariance
            enc = self.encode(x.unsqueeze(0)).squeeze(0)   # [d]
            yhat = enc @ self.alpha
            Pphi = self.P @ enc                       # [d]
            S = torch.clamp(enc @ Pphi + R, min=self.pyy_floor)
            A = torch.clamp(y - yhat, -self.a_clip, self.a_clip)
            G = Pphi / S                              # Kalman gain
            self.alpha += self.lr * G * A
            self.P -= torch.outer(G, Pphi)
            self.P.diagonal().add_(self.q_proc)       # process noise Q: keeps P from collapsing

    # ---------- prediction ----------
    def predict_batch(self, X):  # [B, size] -> [B]
        return self.encode(X) @ self.alpha

    # ---------- sliding windows ----------
    def _build_windows(self, sets, matrix_t):
        sets_idx = torch.as_tensor(list(sets), dtype=torch.long, device=self.dev)
        Xs, Ys = [], []
        for n in range(matrix_t.shape[0]):
            s = matrix_t[n]
            w = s.unfold(0, self.size, 1)
            Xs.append(w[sets_idx])
            Ys.append(s[sets_idx + self.size])
        return torch.cat(Xs, 0), torch.cat(Ys, 0)

    # ---------- training (single-pass online, or epochs passes when offline) ----------
    def train(self, sets_training, matrix_1_norm, matrix_1_norm_org, y, epochs, sets_cv):
        mt = torch.as_tensor(matrix_1_norm, dtype=torch.float32, device=self.dev)
        X, Y = self._build_windows(sets_training, mt)
        if self.encoder_type == "temporal":
            self.encoder.configure(float(X.min()), float(X.max()))
            self.encoder.to(self.dev)

        online = bool(getattr(self.opt, "online", 1))
        n_epochs = 1 if online else int(epochs)
        N = X.shape[0]
        with torch.no_grad():
            for _ in range(n_epochs):
                for j in range(N):
                    self.update(X[j], Y[j])

    # ---------- testing ----------
    def test(self, sets_testing, matrix_1_norm, matrix_1_norm_org, y, cv=False):
        mt = torch.as_tensor(matrix_1_norm, dtype=torch.float32, device=self.dev)
        mo = torch.as_tensor(matrix_1_norm_org, dtype=torch.float32, device=self.dev)
        Xt, _ = self._build_windows(sets_testing, mt)
        _, Yt = self._build_windows(sets_testing, mo)

        preds = []
        bs = 256
        with torch.no_grad():
            for start in range(0, Xt.shape[0], bs):
                preds.append(self.predict_batch(Xt[start:start + bs]))
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
    mode = getattr(opt, "ukf_mode", "ukf")
    model = UKF_HD(
        size, d, models, opt, dev,
        encoder_type=getattr(opt, "encoder", "nonlinear"),
        ukf_mode=mode,
    ).to(dev)
    return model
