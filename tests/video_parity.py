"""Parity check: patches/0670's video preprocessing against the model's own Glm5NextVideoProcessor.

Runs inside the image (torch, torchvision, transformers). Every value is compared; the sampling, the canvas and the
patched pixels must be identical, because they are what the tower sees.
"""

import sys

import numpy as np
import torch
from transformers.models.glm5_next.video_processing_glm5_next import Glm5NextVideoProcessor
from transformers.image_utils import PILImageResampling

from tensorfold.families.glm5_next.cuda import video as V
from tensorfold.families.glm5_next.cuda.vision_prep import Settings

PROC = Glm5NextVideoProcessor()
S = Settings.read(None)
LIM = V.VideoLimits()

FAILS = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


print("== processor config ==")
print("  patch", PROC.patch_size, "temporal", PROC.temporal_patch_size, "merge", PROC.merge_size,
      "min_tokens", PROC.min_image_tokens, "max_tokens", PROC.max_image_tokens, "fps", PROC.fps,
      "max_frames", PROC.max_frames)
check("Settings matches the processor's geometry",
      (S.patch, S.temporal, S.merge) == (PROC.patch_size, PROC.temporal_patch_size, PROC.merge_size),
      f"{S.patch}/{S.temporal}/{S.merge}")
check("our video budget is the video processor's", LIM.max_tokens == PROC.max_image_tokens,
      f"{LIM.max_tokens} vs {PROC.max_image_tokens}")
check("our frame rate is the video processor's", LIM.fps == float(PROC.fps), f"{LIM.fps} vs {PROC.fps}")
check("our frame ceiling is the video processor's", LIM.max_frames == PROC.max_frames,
      f"{LIM.max_frames} vs {PROC.max_frames}")

print("== sample_frames ==")
for total, fps in ((48, 24.0), (300, 30.0), (8, 24.0), (16, 2.0), (1000, 25.0)):
    ours = V.sample_indices(total, fps, LIM)
    try:
        import dataclasses

        from transformers.video_utils import VideoMetadata

        fields = {f.name for f in dataclasses.fields(VideoMetadata)}
        kwargs = {"fps": fps, "duration": total / fps, "total_num_frames": total}
        if "timestamps" in fields:
            kwargs["timestamps"] = [i / fps for i in range(total)]
        md = VideoMetadata(**kwargs)
        ref = [int(i) for i in PROC.sample_frames(md, fps=LIM.fps)]
    except Exception as exc:                                     # noqa: BLE001 - the metadata API may differ
        print(f"  skip  reference sample_frames ({type(exc).__name__}: {exc})")
        check(f"sampling {total}@{fps}: even, in range, deduped",
              len(ours) % 2 == 0 and ours == sorted(set(ours)) and 0 <= ours[0] and ours[-1] < total, str(ours[:6]))
        continue
    check(f"sampling {total} frames @{fps} fps -> {len(ref)} indices match", list(ours) == list(ref),
          f"ours {ours[:5]}... ref {ref[:5]}...")

print("== canvas and patched pixels ==")
CASES = ((32, 48, 8), (64, 64, 8), (90, 160, 6), (14, 14, 4), (1024, 768, 4), (48, 48, 100))
for height, width, frames in CASES:
    generator = torch.Generator().manual_seed(hash((height, width, frames)) & 0xFFFF)
    stack = torch.randint(0, 256, (frames, 3, height, width), dtype=torch.uint8, generator=generator)
    want_h, want_w, content_h, content_w = V.video_canvas(height, width, frames, S, LIM.max_tokens)
    ref = PROC._preprocess(                                          # noqa: SLF001 - the reference path itself
        videos=[stack],
        do_convert_rgb=True,
        do_resize=True,
        resample=PILImageResampling.BICUBIC,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=tuple(PROC.image_mean),
        image_std=tuple(PROC.image_std),
        patch_expand_factor=PROC.patch_expand_factor,
        patch_size=PROC.patch_size,
        temporal_patch_size=PROC.temporal_patch_size,
        merge_size=PROC.merge_size,
        min_image_tokens=PROC.min_image_tokens,
        max_image_tokens=PROC.max_image_tokens,
        return_tensors="pt",
    )
    ref_pixels = ref["pixel_values_videos"]
    grid_t, grid_h, grid_w = (int(v) for v in ref["video_grid_thw"][0])
    own_h, own_w, own_ch, own_cw = want_h, want_w, content_h, content_w
    check(f"{frames}f {height}x{width}: canvas {own_h}x{own_w} == reference {grid_h * S.patch}x{grid_w * S.patch}",
          (own_h // S.patch, own_w // S.patch) == (grid_h, grid_w),
          f"content {own_ch}x{own_cw}")
    groups = V.preprocess_groups(V.fit_frames(stack, own_h, own_w, own_ch, own_cw), S)
    ours = torch.cat([g.pixels for g in groups])
    check(f"{frames}f {height}x{width}: {len(groups)} groups == grid_t {grid_t}", len(groups) == grid_t)
    check(f"{frames}f {height}x{width}: pixels {tuple(ours.shape)} == {tuple(ref_pixels.shape)}",
          tuple(ours.shape) == tuple(ref_pixels.shape))
    if tuple(ours.shape) == tuple(ref_pixels.shape):
        diff = (ours - ref_pixels).abs().max().item()
        check(f"{frames}f {height}x{width}: pixels identical", diff == 0.0, f"max |diff| {diff:.3e}")
    check(f"{frames}f {height}x{width}: tokens per group == replace_video_token",
          all(g.tokens == (grid_t * grid_h * grid_w) // (S.merge ** 2) // grid_t for g in groups),
          f"{groups[0].tokens}")
    check(f"{frames}f {height}x{width}: grids are (1, gh, gw)",
          all(g.grid == (1, grid_h, grid_w) for g in groups), str(groups[0].grid))

print("== prompt structure ==")
stamps = [0.0, 1.0, 2.5]
text = V.video_structure(stamps)
check("video tokens frame the runs", text.startswith(V.VIDEO_BEGIN) and text.endswith(V.VIDEO_END), text[:60])
check("one image run per group", text.count("<|begin_of_image|>") == len(stamps))
check("stamps are one decimal", all(f"{s:.1f} seconds" in text for s in stamps))

print()
if FAILS:
    print(f"FAILED: {len(FAILS)}")
    for name in FAILS:
        print("  -", name)
    sys.exit(1)
print("ALL PARITY CHECKS PASSED")
