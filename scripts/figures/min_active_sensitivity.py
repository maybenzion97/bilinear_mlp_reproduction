#!/usr/bin/env python3
"""Activity-filtered sensitivity of the rank-2 correlation fractions (Table 3).

Recomputes, purely from the stored per-feature records, the fraction of
features with rank-2 Pearson r above each Table 3 threshold under minimum
activity filters n_active >= {1, 5, 50, 100, 500, 1000}. Backing the
Section 4.2 analysis: fw-medium's >0.75 fraction is criterion-dependent
(44.5% over all valid features vs 73.4% over well-sampled features), while
ts-medium and fw-small contain essentially no rarely-active features.

Input: results/language/correlation_{ts_medium,fw_small,fw_medium}.json
Output: results/language/min_active_sensitivity.json
Env: conda from environment_cpu.yml (stdlib json only; no numpy needed).

Usage:
    python scripts/figures/min_active_sensitivity.py
"""
import json
import math
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
THRESHOLDS = (0.40, 0.50, 0.60, 0.75)
MIN_ACTIVE = (1, 5, 50, 100, 500, 1000)
MODELS = ("ts-medium", "fw-small", "fw-medium")

out = {}
for model in MODELS:
    path = REPO / f"results/language/correlation_{model}.json"
    per_feature = json.load(open(path))["per_feature"]
    feats = []
    for f in per_feature:
        r = f["correlations"].get("2")
        if r is None or (isinstance(r, float) and math.isnan(r)):
            continue
        feats.append((int(f["n_active"]), float(r)))
    n_actives = sorted(n for n, _ in feats)
    res = {"n_valid": len(feats),
           "n_active_min": n_actives[0],
           "n_active_p01": n_actives[max(0, len(n_actives) // 100 - 1)]}
    for m in MIN_ACTIVE:
        sub = [r for n, r in feats if n >= m]
        res[f"min_active_{m}"] = {
            "n": len(sub),
            **{f"frac_gt_{t}": (100.0 * sum(r > t for r in sub) / len(sub))
               if sub else None for t in THRESHOLDS},
        }
    out[model] = res

print(json.dumps(out, indent=1))
dest = REPO / "results/language/min_active_sensitivity.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", dest)
