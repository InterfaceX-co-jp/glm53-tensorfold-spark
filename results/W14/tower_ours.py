#!/usr/bin/env python3
"""W14: our Tower (patches/0500, in the image) on hf_tower.py's inputs, bf16 and fp32, then the comparison table.
tower_ours.py SNAPSHOT OUTDIR [screenshot.png]"""
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from tensorfold.families.glm5_next.cuda import vision, vision_prep as vp

snap, out = sys.argv[1], Path(sys.argv[2])
s = vp.Settings.read(snap)
imgs = {"rand1024x768": Image.fromarray(np.random.default_rng(0).integers(0, 256, (768, 1024, 3), dtype=np.uint8))}
if len(sys.argv) > 3:
    imgs["desk1920x1080"] = Image.open(sys.argv[3]).convert("RGB")
for dt in (torch.bfloat16, torch.float32):
    tw = vision.Tower.load(snap, dtype=dt)
    for name, im in imgs.items():
        prep = vp.preprocess(im, s)
        torch.save(tw.encode(prep.pixels, prep.grid).float().cpu(), out / f"ours-{name}-{str(dt).split('.')[-1]}.pt")
    del tw; torch.cuda.empty_cache()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
for name in imgs:
    t = {f"{k}-{d}": torch.load(out / f"{k}-{name}-{d}.pt") for k in ("ours", "hf") for d in ("bfloat16", "float32")
         if (out / f"{k}-{name}-{d}.pt").exists()}
    print(name, {k: tuple(v.shape) for k, v in t.items()})
    for a, b in [("ours-bfloat16", "ours-float32"), ("hf-bfloat16", "hf-float32"), ("ours-float32", "hf-float32"),
                 ("ours-bfloat16", "hf-bfloat16"), ("ours-bfloat16", "hf-float32")]:
        if a in t and b in t:
            cos = torch.nn.functional.cosine_similarity(t[a], t[b], dim=-1)
            print(f"  {a} vs {b}: rel {rel(t[a], t[b]):.4f}, row cosine min {cos.min().item():.5f} mean {cos.mean().item():.6f}")
