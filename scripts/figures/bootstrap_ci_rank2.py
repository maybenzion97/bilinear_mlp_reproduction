#!/usr/bin/env python3
"""
Bootstrap 95% CIs for Table 3 (rank-2 correlation threshold sensitivity).

For each model, loads results/language/correlation_<model>.json, extracts the
rank-2 Pearson correlation per feature (per_feature[i].correlations["2"]),
drops None/NaN entries (matching the report's "features with valid Pearson r"
counts: ts-medium 2048, fw-small 3071, fw-medium 5278), and computes the
fraction of features above each threshold with a nonparametric bootstrap
over features (10,000 resamples, percentile CIs).

Input: results/language/correlation_{ts-medium,fw-small,fw-medium}.json
Output: results/language/bootstrap_ci_rank2.json
Env: conda from environment_cpu.yml (numpy only). Run from repo root.

Usage:
    python scripts/figures/bootstrap_ci_rank2.py
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "language"

MODELS = ["ts-medium", "fw-small", "fw-medium"]
THRESHOLDS = [0.40, 0.50, 0.60, 0.75]
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05


def load_rank2_correlations(model: str) -> np.ndarray:
    """Load per-feature rank-2 Pearson r, excluding None/NaN entries."""
    path = RESULTS_DIR / f"correlation_{model}.json"
    with open(path) as f:
        data = json.load(f)
    raw = [feat["correlations"].get("2") for feat in data["per_feature"]]
    vals = np.array([np.nan if v is None else v for v in raw], dtype=np.float64)
    return vals[~np.isnan(vals)]


def bootstrap_fraction_ci(vals: np.ndarray, thresholds: list, rng: np.random.Generator):
    """Bootstrap over features: resample indices once, evaluate all thresholds."""
    n = len(vals)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    resampled = vals[idx]  # (N_BOOT, n)
    out = {}
    for t in thresholds:
        point = float(np.mean(vals > t))
        boot_fracs = np.mean(resampled > t, axis=1)
        lo, hi = np.percentile(boot_fracs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out[f"{t:.2f}"] = {
            "fraction": point,
            "ci_low": float(lo),
            "ci_high": float(hi),
        }
    return out


def main():
    results = {
        "description": "Bootstrap 95% percentile CIs for fraction of features "
                       "with rank-2 correlation > threshold (Table 3)",
        "n_resamples": N_BOOT,
        "seed": SEED,
        "models": {},
    }
    for model in MODELS:
        vals = load_rank2_correlations(model)
        rng = np.random.default_rng(SEED)
        cis = bootstrap_fraction_ci(vals, THRESHOLDS, rng)
        results["models"][model] = {
            "n_features": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "thresholds": cis,
        }
        print(f"{model} (n={len(vals)}, mean={np.mean(vals):.3f}, "
              f"median={np.median(vals):.3f})")
        for t_key, c in cis.items():
            print(f"  >{t_key}: {100 * c['fraction']:.1f}% "
                  f"[{100 * c['ci_low']:.1f}, {100 * c['ci_high']:.1f}]")

    out_path = RESULTS_DIR / "bootstrap_ci_rank2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
