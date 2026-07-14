#!/usr/bin/env python3
"""Verify the negation-feature geometry claims of the Figure 8 case study.

Computes, directly from the released fw-medium weights and layer-7 mlp-out SAE
(tdooms/fw-medium, tdooms/fw-medium-scope 7-mlp-out-x8-k30), the cosine
similarities between features 3834 and 751 backing the paper text:
decoder cosine (-0.73, near anti-parallel), encoder cosine, top-eigenvector
cosines (-0.996 / -0.990), and the opposing top-2 eigenvalue signs.

Weights are fetched via huggingface_hub (cached after first use).
Output: results/language/negation_cosine_check.json
Env: conda from environment_cpu.yml (torch, safetensors, huggingface_hub).

Usage:
    python scripts/figures/verify_negation_cosine.py
"""
import json
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

REPO = pathlib.Path(__file__).resolve().parents[2]
FEATS = (3834, 751)
LAYER = 7

model_path = hf_hub_download("tdooms/fw-medium", "model.safetensors")
sae_path = hf_hub_download("tdooms/fw-medium-scope",
                           f"{LAYER}-mlp-out-x8-k30/model.safetensors")

mw = load_file(model_path)
sw = load_file(sae_path)

w = mw[f"transformer.h.{LAYER}.mlp.w.weight"].float()   # [2*d_hidden, d_model]
w_l, w_r = w.chunk(2, dim=0)
w_p = mw[f"transformer.h.{LAYER}.mlp.p.weight"].float()  # [d_model, d_hidden]
w_enc = sw["w_enc.weight"].float()                       # [n_feat, d_model]
w_dec = sw["w_dec.weight"].float()                       # [d_model, n_feat]


def cos(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0),
                               b.flatten().unsqueeze(0)).item()


def q_for(direction):
    proj = direction @ w_p
    Q = (proj.unsqueeze(1) * w_l).T @ w_r
    return 0.5 * (Q + Q.T)


eig = {}
for f in FEATS:
    ev, evec = torch.linalg.eigh(q_for(w_enc[f]))
    order = ev.abs().argsort(descending=True)
    eig[f] = (ev[order], evec[:, order])

a, b = eig[FEATS[0]], eig[FEATS[1]]
out = {
    "features": list(FEATS),
    "decoder_cosine": cos(w_dec[:, FEATS[0]], w_dec[:, FEATS[1]]),
    "encoder_cosine": cos(w_enc[FEATS[0]], w_enc[FEATS[1]]),
    "top_eigvec_cosines": [cos(a[1][:, i], b[1][:, i]) for i in range(2)],
    "top2_eigenvalues": {str(f): [float(v) for v in eig[f][0][:2]] for f in FEATS},
}
print(json.dumps(out, indent=2))
dest = REPO / "results/language/negation_cosine_check.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote", dest)
