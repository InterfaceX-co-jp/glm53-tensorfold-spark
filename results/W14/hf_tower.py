#!/usr/bin/env python3
"""W14: transformers' Glm5NextVisionModel (the reference; head venvs/vllm-exl3, transformers 5.17) on the same inputs
as tests/cuda/test_vision_patches.py's real-tower test (random 1024 x 768, rng 0) and on a real screenshot: bf16 and
fp32 outputs saved, so ours (tower_ours.py, in the image) can be compared with both.   hf_tower.py SNAPSHOT OUTDIR"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

snap, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cfg = Glm5NextConfig.from_pretrained(snap)
vc = cfg.vision_config
idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
state = {}
for f in sorted({v for k, v in idx.items() if k.startswith("model.visual.")}):
    with safe_open(snap / f, "pt") as s:
        for k in s.keys():
            if k.startswith("model.visual."):
                state[k[len("model.visual."):]] = s.get_tensor(k)
proc = AutoImageProcessor.from_pretrained(snap)
imgs = {"rand1024x768": Image.fromarray(np.random.default_rng(0).integers(0, 256, (768, 1024, 3), dtype=np.uint8)),
        "desk1920x1080": Image.open(sys.argv[3]).convert("RGB") if len(sys.argv) > 3 else None}
for dt in (torch.bfloat16, torch.float32):
    vc._attn_implementation = "sdpa"
    m = Glm5NextVisionModel._from_config(vc, torch_dtype=dt)
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    m = m.to("cuda", dt).eval()
    for name, im in imgs.items():
        if im is None:
            continue
        p = proc(images=im, return_tensors="pt")
        with torch.no_grad():
            y = m(p["pixel_values"].to("cuda", dt), grid_thw=p["image_grid_thw"].to("cuda")).pooler_output
        torch.save(y.float().cpu(), out / f"hf-{name}-{str(dt).split('.')[-1]}.pt")
        print(name, str(dt), tuple(y.shape), tuple(p["image_grid_thw"][0].tolist()), flush=True)
    del m; torch.cuda.empty_cache()
