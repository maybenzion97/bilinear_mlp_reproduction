#!/usr/bin/env python3
"""Render the Section 6.4 QFS bridge figure from stored results (no recomputation).

Two panels: (top) seed-averaged 10x10 QFS matrix on a diverging colormap scaled
to the data (+/-0.25) so class structure is visible; (bottom) a 1-D anchor scale
on the full [0,1] QFS axis placing the diagonal mean against the budget-matched
control and the same-family anchors, which span too wide a range to read off
the heatmap colormap.

Inputs: results/extension2/qfs_cp_bridge.json, results/extension2/qfs_budget_control.json
Output: bilinear_mlp_reproduction_report/figures/extension_cp/qfs_cp_bridge.pdf
Env: conda from environment_cpu.yml (numpy + matplotlib only).

Usage:
    python scripts/figures/render_qfs_bridge_figure.py
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
bridge = json.load(open(REPO / "results/extension2/qfs_cp_bridge.json"))
control = json.load(open(REPO / "results/extension2/qfs_budget_control.json"))

grid = np.array(bridge["confirmatory_mean_grid"], dtype=float)
anchors = bridge["anchors"]
diag_mean = float(np.mean(np.diag(grid)))
offdiag_mean = float(np.mean(grid[~np.eye(10, dtype=bool)]))
ctrl = control["control_fb_x_pub"]["mean"]

fig = plt.figure(figsize=(6.4, 6.9))
gs = fig.add_gridspec(2, 1, height_ratios=[10, 2.0], hspace=0.35)

# --- heatmap, diverging scale matched to the data ---
ax = fig.add_subplot(gs[0])
vmax = 0.25
im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
for i in range(10):
    for j in range(10):
        v = grid[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.13 else "black")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel(r"$\Lambda$-CP ($R$=256) class")
ax.set_ylabel("dense-wd class")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("QFS ($k$=20)", fontsize=8)

# --- anchor scale on the full [0,1] axis ---
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(-0.06, 1.0)
ax2.set_ylim(0, 1)
ax2.axhline(0.5, color="0.6", lw=1, zorder=1)
# (x, label, color, above?) -- left cluster staggered to avoid overlap
marks = [
    (offdiag_mean, "off-diag $-0.02$", "0.4", False),
    (ctrl, "budget-matched dense $0.13$", "#d62728", True),
    (diag_mean, "dense$\\times$CP diag $0.18$", "#1f77b4", False),
    (anchors["cp_lambda_cross_seed_diag"], "CP$\\times$CP $0.69$", "0.2", True),
    (anchors["dense_wd_cross_seed_diag"], "dense$\\times$dense $0.86$", "0.2", False),
]
for x, label, color, above in marks:
    ax2.plot([x], [0.5], marker="v" if not above else "^", ms=8, color=color, zorder=3)
    if above:
        ax2.annotate(label, (x, 0.5), xytext=(0, 11), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.5, color=color)
    else:
        ax2.annotate(label, (x, 0.5), xytext=(0, -13), textcoords="offset points",
                     ha="center", va="top", fontsize=6.5, color=color)
ax2.set_yticks([])
ax2.set_xticks([0.25, 0.5, 0.75, 1.0])
ax2.tick_params(labelsize=7)
ax2.set_xlabel("same-class QFS, full scale", fontsize=8)
for s in ("left", "right", "top"):
    ax2.spines[s].set_visible(False)

dest = REPO.parent / "bilinear_mlp_reproduction_report/figures/extension_cp/qfs_cp_bridge.pdf"
fig.savefig(dest, bbox_inches="tight")
print("wrote", dest)
print(f"diag {diag_mean:.4f} offdiag {offdiag_mean:.4f} ctrl {ctrl:.4f}")
