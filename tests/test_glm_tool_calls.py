"""GLM tool-call parsing (patches/0002). Run against the patched tree: PYTHONPATH=<tree>/src pytest tests/."""

import json

from tensorfold.cuda.server import parse_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "replace_all": {"type": "boolean"}, "content": {"type": "string"}}}}},
]


def args(call):
    return json.loads(call["function"]["arguments"])


def test_glm_single_call_types_follow_schema():
    text = "Running it.<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la 42</arg_value>" \
           "<arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>"
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "Running it."
    assert calls[0]["function"]["name"] == "bash"
    assert args(calls[0]) == {"command": "ls -la 42", "timeout": 30}


def test_glm_string_that_looks_like_json_stays_text():
    text = "<tool_call>edit\n<arg_key>path</arg_key>\n<arg_value>a.json</arg_value>\n<arg_key>content</arg_key>\n" \
           "<arg_value>{\"a\": 1}\n</arg_value>\n<arg_key>replace_all</arg_key>\n<arg_value>true</arg_value>\n</tool_call>"
    _, calls = parse_tool_calls(text, TOOLS)
    assert args(calls[0]) == {"path": "a.json", "content": "{\"a\": 1}\n", "replace_all": True}


def test_glm_two_calls_and_unknown_tool_left_as_text():
    text = "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>" \
           "<tool_call>nope<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>" \
           "<tool_call>BASH<arg_key>command</arg_key><arg_value>id</arg_value></tool_call>"
    content, calls = parse_tool_calls(text, TOOLS)
    assert [c["function"]["name"] for c in calls] == ["bash", "bash"]
    assert "<tool_call>nope" in content


def test_glm_call_without_arguments():
    _, calls = parse_tool_calls("<tool_call>bash</tool_call>", TOOLS)
    assert args(calls[0]) == {}


def test_qwen_format_still_parses():
    text = "<tool_call><function=bash><parameter=command>\nls\n</parameter></function></tool_call>"
    _, calls = parse_tool_calls(text, TOOLS)
    assert args(calls[0]) == {"command": "ls"}
