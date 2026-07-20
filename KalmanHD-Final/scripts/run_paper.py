"""Varredura de experimentos do novo paper (RegHD/GradHD-TS + UKF-HD + GradHD+UKF).

Roda a matriz (datasets x métodos x ruído x trials) chamando main.py e acumulando em
results_paper.csv. Use o aggregate_paper.py depois para obter média±desvio por trial.

Exemplos:
  python scripts/run_paper.py                 # matriz padrão (1 dataset, ruído gaussiano, 3 trials)
  python scripts/run_paper.py --full          # todos os datasets e todos os ruídos
Ajuste as listas abaixo conforme o orçamento de tempo.
"""
import argparse
import subprocess
import sys

PY = sys.executable

# Nº de clusters varridos para RegHD e GradHD
CLUSTERS = [1, 4, 8, 16]


def build_methods(clusters):
    """(rótulo, args extras para main.py) — implementações consistentes (MAE)."""
    methods = []
    for k in clusters:  # RegHD (delta manual, online)
        methods.append((f"RegHD-{k}", ["--model", "GradHD", "--models", str(k), "--online", "1"]))
    for k in clusters:  # GradHD (backprop, offline)
        methods.append((f"GradHD-{k}", ["--model", "GradHD", "--use_backprop",
                                        "--models", str(k), "--online", "0", "--epochs", "30"]))
    methods += [
        ("KalmanHD-KF", ["--model", "UKFHD", "--ukf_mode", "kf", "--online", "1"]),
        ("UKF-HD",      ["--model", "UKFHD", "--ukf_mode", "ukf", "--online", "1"]),
        ("GradHD+UKF",  ["--model", "GradHD_UKF", "--use_backprop", "--online", "0", "--epochs", "30"]),
    ]
    return methods

# hiperparâmetros por dataset (baseados nos scripts originais do KalmanHD)
DS_HP = {
    "SanFranciscoTraffic":          {"lr": "0.000001", "dim": "2000"},
    "MetroInterstateTrafficVolume": {"lr": "0.001",    "dim": "1000"},
    "GuangzhouTraffic":             {"lr": "0.000001", "dim": "1000"},
    "EnergyConsumptionFraunhofer":  {"lr": "0.000001", "dim": "1000"},
    "ElectricityLoadDiagrams":      {"lr": "0.00001",  "dim": "2000"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="todos os datasets e ruídos")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--encoder", default="nonlinear", choices=["nonlinear", "temporal"])
    ap.add_argument("--clusters", default="1,4,8,16",
                    help="nº de clusters para RegHD/GradHD (csv), ex.: 1,4,8,16")
    ap.add_argument("--lr", default="0.001",
                    help="learning rate dos nossos métodos (encoders contínuos; o 1e-6 do KalmanHD era p/ binário)")
    ap.add_argument("--methods", default=None,
                    help="rótulos exatos separados por vírgula p/ rodar só um subconjunto, "
                         "ex.: 'GradHD-1,GradHD+UKF'. Padrão: todos.")
    args = ap.parse_args()

    methods = build_methods([int(c) for c in args.clusters.split(",")])
    if args.methods:
        wanted = [m.strip() for m in args.methods.split(",")]
        methods = [(label, extra) for (label, extra) in methods if label in wanted]
        if not methods:
            raise SystemExit(f"Nenhum método casou com {wanted}. "
                             f"Opções: {[l for l, _ in build_methods([1,4,8,16])]}")

    # ordem fastest-first (por nº aprox. de janelas de treino) p/ resultados úteis cedo
    order = ["MetroInterstateTrafficVolume", "SanFranciscoTraffic", "EnergyConsumptionFraunhofer",
             "GuangzhouTraffic", "ElectricityLoadDiagrams"]

    if args.full:
        datasets = order
        noises = ([("--gaussian_noise", g) for g in ["0.0", "0.1", "0.2", "0.5", "1.0"]] +
                  [("--p", pp) for pp in ["0.1", "0.2", "0.5"]] +
                  [("--poisson_noise", pl) for pl in ["0.1", "0.2", "0.4", "0.5"]])
    else:
        datasets = ["MetroInterstateTrafficVolume"]
        noises = [("--gaussian_noise", g) for g in ["0.0", "0.5"]]

    for ds in datasets:
        hp = DS_HP[ds]
        for noise_flag, noise_val in noises:
            for label, extra in methods:
                for trial in range(args.trials):
                    cmd = [PY, "main.py", "--dataset", ds, "--device", args.device,
                           "--learning_rate", args.lr, "--dimension_hd", hp["dim"],
                           "--encoder", args.encoder, "--trial", str(trial),
                           noise_flag, noise_val] + extra
                    print(f"\n>>> {ds} | {label} | {noise_flag}={noise_val} | trial={trial}")
                    subprocess.call(cmd)


if __name__ == "__main__":
    main()
