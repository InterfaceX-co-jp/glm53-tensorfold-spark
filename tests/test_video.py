"""patches/0670: the video content parts, the prompt's frame structure, and the expand hand-off.

Host only (no GPU, no PyAV, no torch): the video is decoded for real in ``video_parity.py`` (against the model's
own processor) and in ``tests/test_vision_server.py``. What is checked here is the wiring around it -- which parts
are videos, what the prompt looks like once they are, and that the rows the prompt claims are the rows the engine
gets. Run against the patched tree:

    PYTHONPATH=<tree>/src pytest -q tests/test_video.py
"""

from __future__ import annotations

import dataclasses

import pytest

vision_prep = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision_prep")
from tensorfold.families.glm5_next.cuda import video  # noqa: E402
from tensorfold.families.glm5_next.cuda.vision_prep import (  # noqa: E402
    VBASE, Host, Prepared, Ref, count_images, expand, part_ref)

IMAGE_RUN = "<|begin_of_image|><|image|><|end_of_image|>"
MARKERS = {"<|begin_of_video|>": "video_start_token", "<|end_of_video|>": "video_end_token",
           "<|begin_of_image|>": "begin_token", "<|image|>": "image_token", "<|end_of_image|>": "end_token"}


class EchoTemplate:
    """A chat template reduced to what the host needs of it: the parts' text, in order."""

    def render(self, messages, *, tools, enable_thinking, extra=None) -> str:
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                out.extend(p.get("text", "") for p in content if isinstance(p, dict))
            elif isinstance(content, str):
                out.append(content)
        return "".join(out)


def groups(tokens: int = 4, count: int = 2) -> list[Prepared]:
    """``count`` temporal groups of ``tokens`` rows each, with vids as the registry would hand them out."""

    return [Prepared(None, (1, 4, 4), tokens, bytes([i]), (56, 56),
                     [VBASE + 1000 * i + k for k in range(tokens)]) for i in range(count)]


def ids_of(text: str, s) -> list[int]:
    """The text as token ids: the media markers by their real ids, anything else a filler."""

    import re

    out = []
    for chunk in re.split(r"(<\|begin_of_video\|>|<\|end_of_video\|>|<\|begin_of_image\|>|<\|image\|>"
                          r"|<\|end_of_image\|>)", text):
        if not chunk:
            continue
        name = MARKERS.get(chunk)
        out.append(getattr(s, name) if name else 1)
    return out


def a_video(url: str = "data:video/mp4;base64,AAAA") -> dict:
    return {"type": "video_url", "video_url": {"url": url}}


@pytest.mark.parametrize("part,url", [
    ({"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}, "https://x/v.mp4"),
    ({"type": "video_url", "video_url": "https://x/v.mp4"}, "https://x/v.mp4"),
    ({"type": "video", "video": {"url": "u"}}, "u"),
    ({"type": "video", "url": "u"}, "u"),
    ({"type": "input_video", "url": "u"}, "u"),
    ({"type": "input_video", "video_url": "u"}, "u"),
])
def test_every_video_part_form_reads_as_a_video(part, url):
    ref = part_ref(part)
    assert ref is not None and ref.url == url and ref.kind == "video"
    assert ref.groups == [] and ref.stamps == []          # filled in by render, not by the part


def test_an_image_part_is_still_an_image():
    ref = part_ref({"type": "image_url", "image_url": {"url": "u"}})
    assert ref.kind == "image"
    assert part_ref({"type": "text", "text": "hi"}) is None


def test_a_video_without_a_url_is_a_400():
    with pytest.raises(vision_prep.VisionError):
        part_ref({"type": "video_url", "video_url": {}})
    with pytest.raises(vision_prep.VisionError):
        part_ref({"type": "video", "url": ""})


def test_videos_count_as_media():
    messages = [{"role": "user", "content": [{"type": "text", "text": "x"}, a_video()]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]
    assert count_images(messages) == 2


def test_video_structure_frames_each_group_with_its_stamp():
    assert video.video_structure([0.0, 1.5]) == (
        "<|begin_of_video|>"
        f"{IMAGE_RUN}0.0 seconds"
        f"{IMAGE_RUN}1.5 seconds"
        "<|end_of_video|>")


def test_a_video_renders_one_run_per_group_and_expands(monkeypatch):
    host = Host(None)
    made = groups()
    monkeypatch.setattr(host, "prepare_video", lambda ref, deadline=None: (list(made), [0.0, 1.5]))
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "watch "}, a_video(), {"type": "text", "text": " ok"}]}]
    text, refs = host.render(EchoTemplate(), messages, tools=[], enable_thinking=False, extra=None)
    assert text == ("watch <|begin_of_video|>"
                    f"{IMAGE_RUN}0.0 seconds"
                    f"{IMAGE_RUN}1.5 seconds"
                    "<|end_of_video|> ok")
    assert [r.kind for r in refs] == ["video"]
    assert refs[0].stamps == [0.0, 1.5]

    request = host.request(refs)                              # render's groups, not decoded again
    assert request.images == made
    assert request.extra == 2 * (4 - 1)                       # each group adds tokens - 1 rows

    out = expand(ids_of(text, host.s), request.images, host.s)
    assert host.s.image_token not in out                      # every one-row placeholder became rows
    assert sum(1 for t in out if t >= VBASE) == 8             # 2 groups x 4 rows
    assert len(out) == len(ids_of(text, host.s)) - 2 + 8


def test_an_image_and_a_video_in_one_prompt_keep_their_order(monkeypatch):
    host = Host(None)
    made = groups(tokens=4, count=1)
    monkeypatch.setattr(host, "prepare_video", lambda ref, deadline=None: (list(made), [0.0]))
    images = [Prepared(None, (1, 2, 2), 1, b"img", (28, 28), [VBASE + 7])]
    monkeypatch.setattr(host, "prepare_ref", lambda ref, deadline=None: images[0])
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "i"}}, a_video(), {"type": "text", "text": "?"}]}]
    text, refs = host.render(EchoTemplate(), messages, tools=[], enable_thinking=False, extra=None)
    assert [r.kind for r in refs] == ["image", "video"]
    assert text.count(IMAGE_RUN) == 2                         # the image, then the video's one group
    request = host.request(refs)
    assert request.images == [images[0], made[0]]
    out = expand(ids_of(text, host.s), request.images, host.s)
    assert sum(1 for t in out if t >= VBASE) == 1 + 4
    assert out.index(VBASE + 7) < out.index(VBASE + 0)        # the image's row comes before the video's rows


def test_request_refuses_a_video_render_did_not_prepare():
    host = Host(None)
    with pytest.raises(vision_prep.VisionError, match="not prepared"):
        host.request([Ref("data:video/mp4;base64,AAAA", None, "video")])


def test_too_many_videos_is_a_400(monkeypatch):
    host = Host(None)
    monkeypatch.setattr(host, "video", dataclasses.replace(host.video, max_videos=1))
    messages = [{"role": "user", "content": [a_video("u1"), a_video("u2")]}]
    with pytest.raises(vision_prep.VisionError, match="at most 1"):
        host.render(EchoTemplate(), messages, tools=[], enable_thinking=False, extra=None)


def test_video_limits_are_validated(monkeypatch):
    monkeypatch.delenv("GLM53_TF_VIDEO_MAX_VIDEOS", raising=False)
    assert video.VideoLimits.read().max_videos >= 1
    monkeypatch.setenv("GLM53_TF_VIDEO_MAX_VIDEOS", "0")
    with pytest.raises(ValueError):
        video.VideoLimits.read()
    monkeypatch.setenv("GLM53_TF_VIDEO_MAX_VIDEOS", "2")
    monkeypatch.setenv("GLM53_TF_VIDEO_FPS", "-1")
    with pytest.raises(ValueError):
        video.VideoLimits.read()


def test_sample_indices_are_glms_own_choice():
    pytest.importorskip("numpy")
    lim = video.VideoLimits()
    indices = video.sample_indices(300, 30.0, lim)             # 10 s at 2 fps
    assert indices == sorted(set(indices)) and len(indices) % 2 == 0 and indices[-1] < 300
    assert video.sample_indices(8, 24.0, lim) == []             # under a second: the processor's own answer


def test_video_budget_is_the_processors():
    s = vision_prep.Settings.read(None)
    assert (s.video_token, s.video_start_token, s.video_end_token) == (154855, 154832, 154833)
    assert video.VIDEO_MAX_TOKENS == 240000                     # the video processor's max_image_tokens
    assert video.VIDEO_MAX_FRAMES == 2048 and video.VIDEO_FPS == 2.0
