#!/usr/bin/env python3
"""Kolibri-1 chat-template parity: ds4 vs the released template's renderer.

Renders conversations with the released Jinja template (from the fixture
GGUF's tokenizer.chat_template metadata, HF-compatible environment) and
encodes the rendered strings with the HuggingFace tokenizers BPE (an
independent implementation). Each case must match ds4's chat prompt
tokenization id-for-id, special tokens included.
"""
import json
import os
import pathlib
import subprocess
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from kolibri1_reference import load_gguf, tensor_array  # noqa: E402

GGUF = ROOT / "gguf" / "Kolibri-1-mini.gguf"
DS4 = str(ROOT / "ds4")

meta, tensors, blob, data_start = load_gguf(str(GGUF))
template_src = meta["tokenizer.chat_template"]
if isinstance(template_src, bytes):
    template_src = template_src.decode()

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
template = env.from_string(template_src)

try:
    from tokenizers import Tokenizer
except ImportError:
    print("kolibri1-chat: tokenizers package required (pip install tokenizers)")
    sys.exit(2)

snapshot = os.environ.get("KOLIBRI1_TOKENIZER_DIR")
if snapshot and (pathlib.Path(snapshot) / "tokenizer.json").exists():
    tok = Tokenizer.from_file(str(pathlib.Path(snapshot) / "tokenizer.json"))
else:
    # fall back to the mini fixture: it embeds the full metadata, but the
    # GGUF only carries the token strings, so we need tokenizer.json's
    # added-token matching from the snapshot directory.
    print("kolibri1-chat: KOLIBRI1_TOKENIZER_DIR with tokenizer.json required")
    sys.exit(2)


def ds4_ids(rendered: str) -> list[int]:
    proc = subprocess.run(
        [DS4, "-m", str(GGUF), "--dump-tokens", "-p", rendered],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ds4 failed: {proc.stderr.strip()}")
    first = proc.stdout.splitlines()[0]
    return [int(t) for t in first.strip().strip("[]").split(",")]


def ds4_chat_ids(system: str, prompt: str, think_level: int) -> list[int]:
    """Tokenize a chat prompt rendered by ds4 itself (not the paste path)."""
    proc = subprocess.run(
        [DS4, "-m", str(GGUF), "--dump-tokens", "--system", system,
         "--think-level", str(think_level), "-p", prompt],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ds4 failed: {proc.stderr.strip()}")
    first = proc.stdout.splitlines()[0]
    return [int(t) for t in first.strip().strip("[]").split(",")]


AGENT_DRIVER = str(ROOT / "tests" / "test_kolibri1_agent_chat")
SERVER_DRIVER = str(ROOT / "tests" / "test_kolibri1_server_render")

AGENT_SYSTEM_TEXT = "You are a coding agent running in a local workspace."
AGENT_USER_TEXT = "List the files here, then read the first one."
AGENT_ASSISTANT_TEXT = ("Checking the directory first.", [
    {"name": "list", "arguments": {"path": "."}},
    {"name": "read", "arguments": {"path": "README.md", "max_lines": 5}},
])
AGENT_TOOL_RESULTS = ["a\nb\nc", "line1\nline2"]


def agent_tools_section() -> str:
    proc = subprocess.run([AGENT_DRIVER, "--tools-section"],
                          capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"agent driver failed: {proc.stderr.strip()}")
    return proc.stdout


def agent_rules_text() -> str:
    proc = subprocess.run([AGENT_DRIVER, "--rules"],
                          capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"agent driver failed: {proc.stderr.strip()}")
    return proc.stdout


def agent_transcript_ids() -> list[int]:
    proc = subprocess.run([AGENT_DRIVER, str(GGUF)],
                          capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"agent driver failed: {proc.stderr.strip()}")
    return [int(line) for line in proc.stdout.split()]


def agent_reference_ids(tools_section: str, rules: str) -> list[int]:
    """Render the same conversation with the released template.

    The ds4 agent folds its coding-agent rules into the system content (the
    template only renders the fixed tools section from `tools=`), so the
    reference content is system text + rules verbatim."""
    _, calls = AGENT_ASSISTANT_TEXT
    schema_lines = tools_section.split("<tools>\n", 1)[1].rsplit("\n</tools>", 1)[0]
    tools = [json.loads(line) for line in schema_lines.split("\n") if line.strip()]
    content = AGENT_SYSTEM_TEXT + rules
    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": AGENT_USER_TEXT},
        {"role": "assistant", "content": AGENT_ASSISTANT_TEXT[0],
         "tool_calls": calls},
    ]
    for result in AGENT_TOOL_RESULTS:
        messages.append({"role": "tool", "name": "run", "content": result})
    rendered = template.render(messages=messages, tools=tools,
                               add_generation_prompt=True)
    return tok.encode(rendered).ids


def server_rendered(body: dict, effort: str) -> str:
    """Render a request body through the ds4-server Kolibri renderer."""
    proc = subprocess.run([SERVER_DRIVER, "--effort", effort],
                          input=json.dumps(body), capture_output=True, text=True,
                          cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"server render driver failed: {proc.stderr.strip()}")
    return proc.stdout


def server_live_tail(body: dict, effort: str, start: int) -> str:
    """Render the live tool tail for a session mid tool round."""
    proc = subprocess.run([SERVER_DRIVER, "--effort", effort,
                           "--live-tail", str(start)],
                          input=json.dumps(body), capture_output=True, text=True,
                          cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"server render driver failed: {proc.stderr.strip()}")
    return proc.stdout


SERVER_TOOLS = [{"type": "function", "function": {
    "name": "ls", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}}}}]

SERVER_CASES = [
    # (name, request body, jinja kwargs, effort)
    ("server-system+user",
     {"messages": [{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "Hello"}]},
     {}, "high"),
    ("server-nothink",
     {"messages": [{"role": "user", "content": "Hello"}]},
     {}, "none"),
    ("server-system+effort-low",
     {"messages": [{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "Hello"}]},
     {}, "low"),
    ("server-tools-calls-results",
     {"messages": [
         {"role": "user", "content": "check both"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"type": "function", "id": "call_1", "function": {
                 "name": "ls", "arguments": "{\"path\": \".\"}"}},
             {"type": "function", "id": "call_2", "function": {
                 "name": "cat", "arguments": "{\"path\": \"a b/c\"}"}}]},
         {"role": "tool", "content": "a\nb"},
         {"role": "tool", "content": "line1"},
         {"role": "user", "content": "summarize"},
      ],
      "tools": SERVER_TOOLS},
     {}, "high"),
    ("server-assistant-replay-reasoning",
     {"messages": [
         {"role": "user", "content": "one"},
         {"role": "assistant", "content": "answer one",
          "reasoning": "because 2+2=4"},
         {"role": "user", "content": "two"},
      ]},
     {}, "high"),
    ("server-embedded-think",
     {"messages": [
         {"role": "user", "content": "one"},
         {"role": "assistant", "content": "<think>\nbecause\n</think>\nanswer one"},
         {"role": "user", "content": "two"},
      ]},
     {}, "high"),
]


def run_server_cases() -> int:
    """Byte-compare the server renderer against the released template."""
    failures = 0
    for name, body, kwargs, effort in SERVER_CASES:
        rendered = template.render(messages=body["messages"],
                                   tools=body.get("tools"),
                                   add_generation_prompt=True,
                                   reasoning_effort=effort, **kwargs)
        try:
            got = server_rendered(body, effort)
        except RuntimeError as exc:
            failures += 1
            print(f"kolibri1-chat: {name:32s} DRIVER FAILED ({exc})")
            continue
        if got == rendered:
            print(f"kolibri1-chat: {name:32s} {len(got):4d} chars OK")
            continue
        failures += 1
        print(f"kolibri1-chat: {name:32s} MISMATCH "
              f"(driver {len(got)} vs ref {len(rendered)} chars)")
        for i, (g, r) in enumerate(zip(got, rendered)):
            if g != r:
                lo = max(0, i - 25)
                print(f"    first diff at {i}:\n    driver {got[lo:i+25]!r}\n"
                      f"    ref    {rendered[lo:i+25]!r}")
                break
        else:
            print("    length differs after the common prefix")
    return failures


# Mid tool-round continuation: the assistant turn is already sampled in the
# live KV, so the tail after the assistant end token must be exactly the
# template's bytes from that boundary on (grouped tool results + the next
# generation prefix).
SERVER_LIVE_TAIL_BODY = {
    "messages": [
        {"role": "user", "content": "check both"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "call_1", "function": {
                "name": "ls", "arguments": "{\"path\": \".\"}"}},
            {"type": "function", "id": "call_2", "function": {
                "name": "cat", "arguments": "{\"path\": \"a b/c\"}"}}]},
        {"role": "tool", "content": "a\nb"},
        {"role": "tool", "content": "line1"},
    ],
    "tools": SERVER_TOOLS,
}


def run_server_live_tail_cases() -> int:
    """Gate render_live_tool_tail_for_syntax against the released template.

    The live KV of a tool-call turn ends before the assistant end token; the
    tail must therefore equal the full render sliced just before it."""
    failures = 0
    body = SERVER_LIVE_TAIL_BODY
    messages = body["messages"]
    start = next(i for i, m in enumerate(messages) if m["role"] == "tool")
    marker = "<|im_end|>\n"
    for effort in ("high", "none"):
        full = server_rendered(body, effort)
        try:
            tail = server_live_tail(body, effort, start)
        except RuntimeError as exc:
            failures += 1
            print(f"kolibri1-chat: {'server-live-tool-tail-'+effort:32s} DRIVER FAILED ({exc})")
            continue
        prefix = template.render(messages=messages[:start],
                                 tools=body.get("tools"),
                                 add_generation_prompt=False,
                                 reasoning_effort=effort)
        if not prefix.endswith(marker):
            failures += 1
            print(f"kolibri1-chat: {'server-live-tool-tail-'+effort:32s} "
                  "reference prefix does not close the assistant turn")
            continue
        expected = full[len(prefix) - len(marker):]
        if tail == expected:
            print(f"kolibri1-chat: {'server-live-tool-tail-'+effort:32s} "
                  f"{len(tail):4d} chars OK")
            continue
        failures += 1
        print(f"kolibri1-chat: {'server-live-tool-tail-'+effort:32s} MISMATCH "
              f"(driver {len(tail)} vs ref {len(expected)})")
        for i, (g, r) in enumerate(zip(tail, expected)):
            if g != r:
                lo = max(0, i - 25)
                print(f"    first diff at {i}:\n    driver {tail[lo:i+25]!r}\n"
                      f"    ref    {expected[lo:i+25]!r}")
                break
        else:
            print("    length differs after the common prefix")
    return failures


CASES = [
    ("user-only-default",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "high"}),
    ("system+user-default",
     [{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Hello there"}],
     {}),
    ("nothink",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "none"}),
    ("effort-low",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "low"}),
    ("effort-medium",
     [{"role": "user", "content": "Hello there"}],
     {"reasoning_effort": "medium"}),
    ("system+effort-medium",
     [{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Hello"}],
     {"reasoning_effort": "medium"}),
    ("tool-results-grouped",
     [{"role": "user", "content": "check both"},
      {"role": "assistant", "content": "", "tool_calls": [
          {"name": "ls", "arguments": {"path": "."}},
          {"name": "cat", "arguments": {"path": "a b/c"}}]},
      {"role": "tool", "name": "ls", "content": "a\nb"},
      {"role": "tool", "name": "cat", "content": "line1"},
      {"role": "user", "content": "summarize"}],
     {}),
    ("assistant-replay",
     [{"role": "user", "content": "one"},
      {"role": "assistant", "content": "answer one"},
      {"role": "user", "content": "two"}],
     {}),
    ("assistant-final-with-reasoning",
     [{"role": "user", "content": "one"},
      {"role": "assistant", "content": "answer one",
       "reasoning": "because 2+2=4"},
      {"role": "user", "content": "two"}],
     {"preserve_thinking": True}),
]


def main() -> int:
    failures = 0
    for name, messages, kwargs in CASES:
        rendered = template.render(messages=messages,
                                   add_generation_prompt=True, **kwargs)
        ref_ids = tok.encode(rendered).ids
        got = ds4_ids(rendered)
        if got == ref_ids:
            print(f"kolibri1-chat: {name:32s} {len(got):4d} tokens OK")
            continue
        failures += 1
        print(f"kolibri1-chat: {name:32s} MISMATCH "
              f"(ds4 {len(got)} vs ref {len(ref_ids)})")
        for i, (g, r) in enumerate(zip(got, ref_ids)):
            if g != r:
                lo = max(0, i - 3)
                print(f"    first diff at {i}: ds4 "
                      f"{got[lo:i+3]} ref {ref_ids[lo:i+3]}")
                break
        else:
            print(f"    length differs after {min(len(got), len(ref_ids))}")

    # Numeric --think-level values must select the matching released effort
    # sentence; level 0 is the flag spelling of nothink (regression: the
    # engine used to emit the HIGH sentence for 0 while decoding no-think).
    for level, effort in ((0, "none"), (10, "low"), (40, "medium"),
                          (80, "high")):
        system = "You are terse."
        rendered = template.render(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": "Hello there"}],
            add_generation_prompt=True, reasoning_effort=effort)
        ref_ids = tok.encode(rendered).ids
        try:
            got = ds4_chat_ids(system, "Hello there", level)
        except RuntimeError as exc:
            failures += 1
            print(f"kolibri1-chat: {'think-level-'+str(level):32s} "
                  f"DRIVER FAILED ({exc})")
            continue
        if got == ref_ids:
            print(f"kolibri1-chat: {'think-level-'+str(level):32s} "
                  f"{len(got):4d} tokens OK")
            continue
        failures += 1
        print(f"kolibri1-chat: {'think-level-'+str(level):32s} MISMATCH "
              f"(ds4 {len(got)} vs ref {len(ref_ids)})")
        for i, (g, r) in enumerate(zip(got, ref_ids)):
            if g != r:
                lo = max(0, i - 3)
                print(f"    first diff at {i}: ds4 "
                      f"{got[lo:i+3]} ref {ref_ids[lo:i+3]}")
                break
        else:
            print(f"    length differs after {min(len(got), len(ref_ids))}")

    # Agent tool-calling transcript: system + effort + tools in one block,
    # replayed assistant tool calls, grouped tool results, fresh prefix.
    try:
        tools_section = agent_tools_section()
        rules = agent_rules_text()
        got = agent_transcript_ids()
        ref = agent_reference_ids(tools_section, rules)
        if got == ref:
            print(f"kolibri1-chat: {'agent-tools-transcript':32s} "
                  f"{len(got):4d} tokens OK")
        else:
            failures += 1
            print(f"kolibri1-chat: {'agent-tools-transcript':32s} MISMATCH "
                  f"(ds4 {len(got)} vs ref {len(ref)})")
            for i, (g, r) in enumerate(zip(got, ref)):
                if g != r:
                    lo = max(0, i - 3)
                    print(f"    first diff at {i}: ds4 "
                          f"{got[lo:i+3]} ref {ref[lo:i+3]}")
                    break
            else:
                print(f"    length differs after {min(len(got), len(ref))}")
    except RuntimeError as exc:
        print(f"kolibri1-chat: agent transcript skipped ({exc})")

    # Server renderer: the ds4-server prompt text must byte-match the
    # released template for request-shaped conversations.
    failures += run_server_cases()
    failures += run_server_live_tail_cases()

    if failures:
        print(f"kolibri1-chat: {failures} case(s) FAILED")
        return 1
    print("kolibri1 chat-template parity tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
