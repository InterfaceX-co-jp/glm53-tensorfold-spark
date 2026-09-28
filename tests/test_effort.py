"""patches/0150: OpenAI ``reasoning_effort`` (GLM53_TF_EFFORT_FIELD) and a server default effort
(GLM53_TF_DEFAULT_EFFORT) mapped onto the GLM-5.3 chat template's ``enable_thinking`` / ``reasoning_effort``.

Host only. Run against the patched tree: PYTHONPATH=<tree>/src pytest -q tests/test_effort.py
"""

from __future__ import annotations

import pytest

app = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")
if not hasattr(app, "apply_effort"):
    pytest.skip("patches/0150 not applied", allow_module_level=True)


def run(body, field=True, default=None, thinking=True):
    err = app.apply_effort(body, field=field, default=default, default_thinking=thinking)
    return err, body.get("chat_template_kwargs")


def test_off_by_default_changes_nothing(monkeypatch):
    monkeypatch.delenv("GLM53_TF_EFFORT_FIELD", raising=False)
    monkeypatch.delenv("GLM53_TF_DEFAULT_EFFORT", raising=False)
    field, default = app.effort_env()
    assert (field, default) == (False, None)
    body = {"messages": [], "reasoning_effort": "low"}
    assert run(body, field=field, default=default) == (None, None)
    assert "chat_template_kwargs" not in body
    body = {"messages": [], "chat_template_kwargs": {"enable_thinking": False}}
    assert run(body, field=field, default=default) == (None, {"enable_thinking": False})


@pytest.mark.parametrize("effort,kwargs", [
    ("low", {"enable_thinking": True, "reasoning_effort": "low"}),
    ("LOW", {"enable_thinking": True, "reasoning_effort": "low"}),
    ("medium", {"enable_thinking": True, "reasoning_effort": "high"}),
    ("high", {"enable_thinking": True, "reasoning_effort": "high"}),
    ("max", {"enable_thinking": True}),
    ("xhigh", {"enable_thinking": True}),
    ("none", {"enable_thinking": False}),
    ("minimal", {"enable_thinking": False}),
])
def test_field_mapping(effort, kwargs):
    assert run({"reasoning_effort": effort}) == (None, kwargs)


def test_request_kwargs_win_and_idempotent():
    body = {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False, "x": 1}}
    assert run(body) == (None, {"enable_thinking": False, "reasoning_effort": "low", "x": 1})
    assert run(body) == (None, {"enable_thinking": False, "reasoning_effort": "low", "x": 1})
    body = {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}
    assert run(body)[1] == {"enable_thinking": True, "reasoning_effort": "high"}


def test_bad_effort_is_an_error():
    err, _ = run({"reasoning_effort": "turbo"})
    assert err and "reasoning_effort must be one of" in err


def test_default_effort_only_for_thinking_requests():
    assert run({}, field=False, default="low", thinking=True) == (None, {"reasoning_effort": "low"})
    assert run({}, field=False, default="low", thinking=False) == (None, None)
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    assert run(body, default="low") == (None, {"enable_thinking": False})
    body = {"chat_template_kwargs": {"enable_thinking": True}}
    assert run(body, default="low", thinking=False) == (None, {"enable_thinking": True, "reasoning_effort": "low"})
    # the field wins over the default
    assert run({"reasoning_effort": "high"}, default="low")[1] == {"enable_thinking": True, "reasoning_effort": "high"}


def test_env_validation(monkeypatch):
    monkeypatch.setenv("GLM53_TF_EFFORT_FIELD", "yes")
    with pytest.raises(ValueError):
        app.effort_env()
    monkeypatch.setenv("GLM53_TF_EFFORT_FIELD", "1")
    monkeypatch.setenv("GLM53_TF_DEFAULT_EFFORT", "medium")
    with pytest.raises(ValueError):
        app.effort_env()
    monkeypatch.setenv("GLM53_TF_DEFAULT_EFFORT", "max")
    assert app.effort_env() == (True, None)
    monkeypatch.setenv("GLM53_TF_DEFAULT_EFFORT", "Low")
    assert app.effort_env() == (True, "low")


class _Inner:
    def render(self, messages, *, tools, enable_thinking, extra=None):
        # the checkpoint template's first two lines: the effort line, then the turn
        effort = (extra or {}).get("reasoning_effort")
        effort = effort if effort in ("low", "high") else "max"
        text = f"<|system|>Reasoning Effort: {effort.capitalize()}<|user|>hi<|assistant|><think>"
        return text


def test_rendered_effort_line():
    tpl = app.ThinkingOffTemplate(_Inner())
    body = {"reasoning_effort": "low"}
    run(body)
    kw = dict(body["chat_template_kwargs"])
    thinking = kw.pop("enable_thinking")
    assert tpl.render([], tools=[], enable_thinking=thinking, extra=kw).startswith("<|system|>Reasoning Effort: Low")
    body = {"reasoning_effort": "none"}
    run(body)
    kw = dict(body["chat_template_kwargs"])
    thinking = kw.pop("enable_thinking")
    assert tpl.render([], tools=[], enable_thinking=thinking, extra=kw) == "<|user|>hi<|assistant|><think></think>"
