import os, sys
import numpy as np, torch
from types import SimpleNamespace
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.batch import sample_drafts
from tensorfold.families.glm5_next.cuda.decode import sample_rows
print("GPU_ROUND", os.environ.get("GLM53_TF_GPU_ROUND"))
rng = np.random.default_rng(1)
V = 300
w = SimpleNamespace(comm=None, world=1, vocab_offset=17)
bad = tot = pbad = 0
for it in range(200):
    logits = torch.tensor(rng.normal(size=(5, V)) * 3, dtype=torch.float32).to(torch.bfloat16).cuda()
    logits[2, 7] = logits[2, 9] = logits[2].max() + 1
    specs = []
    for r in range(5):
        samp = None if rng.random() < 0.4 else Sampling(int(rng.integers(0, 1 << 40)), float(rng.uniform(0.3, 1.5)), int(rng.integers(1, 60)), float(rng.uniform(0.5, 1.0)))
        specs.append((r, int(rng.integers(0, 5000)), samp, bool(rng.random() < 0.6)))
    got = sample_drafts(w, logits, specs)
    for (r, pos, samp, want), (tok, p) in zip(specs, got):
        probs = []
        wt = sample_rows(w, logits[r:r + 1], [pos], samp, None, probs if want else None)[0]
        tot += 1
        if tok != wt:
            bad += 1
            if bad < 4: print("tok diff", it, r, samp, tok, wt)
        if want and p != probs[0]:
            pbad += 1
            if pbad < 4: print("p diff", it, r, samp, p, probs[0], tok, wt)
print(f"rows {tot} token diffs {bad} prob diffs {pbad}")
import json
out = []
rng = np.random.default_rng(7)
for it in range(300):
    logits = torch.tensor(rng.normal(size=(5, V)) * 3, dtype=torch.float32).to(torch.bfloat16).cuda()
    specs = []
    for r in range(5):
        samp = None if rng.random() < 0.3 else Sampling(int(rng.integers(0, 1 << 40)), float(rng.uniform(0.3, 1.5)), int(rng.integers(0, 60)), float(rng.uniform(0.5, 1.0)))
        specs.append((r, int(rng.integers(0, 5000)), samp, bool(rng.random() < 0.6)))
    out.append([[int(t), p] for t, p in sample_drafts(w, logits, specs)])
    probs = []
    out.append(sample_rows(w, logits, [s[1] for s in specs], specs[0][2], None, None))
json.dump(out, open(sys.argv[1], "w"))
