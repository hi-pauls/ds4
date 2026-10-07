#!/usr/bin/env python3
"""Kolibri-1 real-artifact Responses/Anthropic server smoke.

Spawns ds4-server on a real Kolibri GGUF (Metal on this Mac), then drives the
two API surfaces the agent traffic does not usually hit:

  * /v1/responses: plain, streamed, tool call, tool-output-only continuation
  * /v1/messages:  plain, streamed, tool_use, tool_result continuation

The continuations are the live tool-tail paths (the client replays only the
tool output), which the OpenAI smoke in the handover does not cover. Run
manually; this is not part of make test.

  python3 tests/test_kolibri1_server_smoke.py --model gguf/Kolibri-1-Q8.gguf
"""
import argparse
import json
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def post(base, path, body, stream=False, timeout=900):
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read().decode()
    except urllib.error.HTTPError as error:
        raise AssertionError(f"{path}: HTTP {error.code}: {error.read().decode()}") from error
    return data if stream else json.loads(data)


def sse_events(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload != "[DONE]":
                events.append(json.loads(payload))
    return events


def message_text(items, key="content"):
    return "".join(c.get("text", "") for item in items for c in item.get(key, []))


def responses_tool_schema():
    return [{
        "type": "function",
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "strict": True,
    }]


def anthropic_tool_schema():
    return [{
        "name": "add",
        "description": "Add two integers and return the sum.",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }]


def smoke(base, model):
    ids = [m["id"] for m in post(base, "/v1/models", None)["data"]]
    assert "kolibri-1" in ids, ids
    model = model or ids[0]
    print(f"models: {ids}; using {model}")

    response = post(base, "/v1/responses", {
        "model": model,
        "input": "What is 2+2? Answer with just the number.",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 16,
        "temperature": 0,
    })
    assert response["object"] == "response", response
    text = message_text([o for o in response.get("output", [])
                         if o.get("type") == "message"])
    assert text.strip(), response
    print("responses plain:", repr(text.strip()))

    raw = post(base, "/v1/responses", {
        "model": model,
        "input": "What is 3+4? Answer with just the number.",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 16,
        "temperature": 0,
        "stream": True,
    }, stream=True)
    types = {e.get("type") for e in sse_events(raw)}
    assert "response.completed" in types, types
    print("responses stream events:", len(types))

    response = post(base, "/v1/responses", {
        "model": model,
        "input": "Use the add tool to compute 2+3. Do not answer from memory.",
        "reasoning": {"effort": "none"},
        "tools": responses_tool_schema(),
        "tool_choice": "auto",
        "max_output_tokens": 128,
        "temperature": 0,
    })
    calls = [o for o in response.get("output", []) if o.get("type") == "function_call"]
    assert calls, f"expected a function_call: {response}"
    call = calls[0]
    assert call.get("name") == "add", call
    print("responses tool:", call.get("arguments"))

    continuation = post(base, "/v1/responses", {
        "model": model,
        "input": [{"type": "function_call_output",
                   "call_id": call["call_id"], "output": "5"}],
        "tools": responses_tool_schema(),
        "reasoning": {"effort": "none"},
        "max_output_tokens": 64,
        "temperature": 0,
    })
    text = message_text([o for o in continuation.get("output", [])
                         if o.get("type") == "message"])
    assert text.strip(), continuation
    print("responses continuation:", repr(text.strip()))

    message = post(base, "/v1/messages", {
        "model": model,
        "system": "Be precise.",
        "messages": [{"role": "user",
                      "content": "What is 5+5? Answer with just the number."}],
        "thinking": {"type": "disabled"},
        "max_tokens": 16,
        "temperature": 0,
    })
    text = "".join(b.get("text", "") for b in message.get("content", [])
                   if b.get("type") == "text")
    assert text.strip(), message
    print("anthropic plain:", repr(text.strip()))

    raw = post(base, "/v1/messages", {
        "model": model,
        "messages": [{"role": "user",
                      "content": "What is 6+6? Answer with just the number."}],
        "thinking": {"type": "disabled"},
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
    }, stream=True)
    types = {e.get("type") for e in sse_events(raw)}
    assert "message_stop" in types, types
    print("anthropic stream events:", len(types))

    message = post(base, "/v1/messages", {
        "model": model,
        "messages": [{"role": "user", "content": "Use the add tool to compute 4+5."}],
        "tools": anthropic_tool_schema(),
        "thinking": {"type": "disabled"},
        "max_tokens": 128,
        "temperature": 0,
    })
    uses = [b for b in message.get("content", []) if b.get("type") == "tool_use"]
    assert uses, f"expected a tool_use block: {message}"
    use = uses[0]
    assert use.get("name") == "add", use
    print("anthropic tool:", use.get("input"))

    continuation = post(base, "/v1/messages", {
        "model": model,
        "messages": [
            {"role": "user", "content": "Use the add tool to compute 4+5."},
            {"role": "assistant", "content": message["content"]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": use["id"], "content": "9"},
            ]},
        ],
        "tools": anthropic_tool_schema(),
        "thinking": {"type": "disabled"},
        "max_tokens": 64,
        "temperature": 0,
    })
    text = "".join(b.get("text", "") for b in continuation.get("content", [])
                   if b.get("type") == "text")
    assert text.strip(), continuation
    print("anthropic continuation:", repr(text.strip()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="real Kolibri-1 GGUF")
    parser.add_argument("--ctx", default=32768, type=int)
    parser.add_argument("--output", type=Path, default=Path(tempfile.mkdtemp(
        prefix="kolibri1-server-smoke-")))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    with (args.output / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [str(ROOT / "ds4-server"), "-m", args.model, "--ctx", str(args.ctx),
             "--host", "127.0.0.1", "--port", str(port),
             "--trace", str(args.output / "trace.txt")],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        try:
            deadline = time.monotonic() + 600
            while True:
                assert proc.poll() is None, "server exited during startup"
                try:
                    post(base, "/v1/models", None, timeout=5)
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    assert time.monotonic() < deadline, "server startup timed out"
                    time.sleep(1.0)
            smoke(base, None)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    assert proc.returncode == 0, f"server exit status {proc.returncode}"
    print("PASS: Kolibri-1 Responses/Anthropic real-artifact smoke")


if __name__ == "__main__":
    sys.exit(main())
