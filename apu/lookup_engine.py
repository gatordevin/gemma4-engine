"""
Lookup Decoding Engine — 230 t/s with tool calling

Uses llama-lookup as a subprocess for n-gram speculative decoding,
with tool call detection and execution in the loop.

Architecture:
  1. Format conversation into Gemma 4 chat template
  2. Run llama-lookup subprocess with the formatted prompt
  3. Parse output for tool call JSON patterns
  4. If tool call found: execute tool, append result, restart generation
  5. Lookup cache (-lcd) persists across calls, building patterns over the session
"""

import os
import json
import subprocess
import tempfile
import time
import re
import signal
from pathlib import Path

LLAMA_CPP = os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp"))
MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/gemma-4-26B-A4B-it-Q4_K_M.gguf"))
LOOKUP_CACHE = os.environ.get("LOOKUP_CACHE", os.path.expanduser("~/models/lookup_session.bin"))
WORKDIR = os.environ.get("WORKDIR", os.path.expanduser("~/projects"))

LOOKUP_BIN = os.path.join(LLAMA_CPP, "build-hip-fast/bin/llama-lookup")
if not os.path.exists(LOOKUP_BIN):
    LOOKUP_BIN = os.path.join(LLAMA_CPP, "build/bin/llama-lookup")

TOOL_DEFS = """You have access to these tools:
- bash(command: str): Execute a shell command in the project directory and return output.
- read_file(path: str): Read a file and return its contents.
- write_file(path: str, content: str): Write content to a file. Creates parent directories.
- edit_file(path: str, old_string: str, new_string: str): Replace old_string with new_string in a file.
- list_files(path: str, depth: int): List files in a directory tree.

Use tools by calling them with the appropriate arguments.
Wait for the tool result before continuing. You can chain multiple tool calls.
Always use tools to actually create/modify files when asked to write code."""

# Detect Gemma 4 native tool call: <|tool_call>call:name(...)<tool_call|>
# Greedy match for args, anchored by <tool_call|> end marker
NATIVE_TOOL_RE = re.compile(r'<\|tool_call\>call:(\w+)[\(\{](.+)[\)\}]<tool_call\|>')
# Fallback: JSON format
JSON_TOOL_RE = re.compile(r'\{"tool":\s*"(\w+)",\s*"args":\s*(\{[^}]*\})\}')


def format_chat(messages, system_extra=""):
    """Format messages into Gemma 4 chat template."""
    system_content = f"You are a helpful coding assistant. Your working directory is {WORKDIR}.\n{TOOL_DEFS}"
    if system_extra:
        system_content += "\n" + system_extra

    parts = [f"<bos><|turn>system\n{system_content}<turn|>"]

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|turn>user\n{content}<turn|>")
        elif role == "assistant":
            parts.append(f"<|turn>model\n{content}<turn|>")
        elif role == "tool_result":
            parts.append(f"<|turn>user\n[Tool Result]\n{content}<turn|>")

    # Add generation prompt
    parts.append("<|turn>model\n")
    return "\n".join(parts)


def exec_tool(name, args):
    """Execute a tool and return result string."""
    try:
        if name == "bash":
            cmd = args.get("command", "")
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=WORKDIR
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr.strip()
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output[:4000] or "(no output)"

        elif name == "read_file":
            fpath = args.get("path", "")
            if not os.path.isabs(fpath):
                fpath = os.path.join(WORKDIR, fpath)
            with open(fpath, "r") as f:
                return f.read(50000) or "(empty file)"

        elif name == "write_file":
            fpath = args.get("path", "")
            if not os.path.isabs(fpath):
                fpath = os.path.join(WORKDIR, fpath)
            Path(fpath).parent.mkdir(parents=True, exist_ok=True)
            content = args.get("content", "")
            with open(fpath, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} bytes to {fpath}"

        elif name == "edit_file":
            fpath = args.get("path", "")
            if not os.path.isabs(fpath):
                fpath = os.path.join(WORKDIR, fpath)
            with open(fpath, "r") as f:
                content = f.read()
            old = args.get("old_string", "")
            new = args.get("new_string", "")
            if old not in content:
                return f"Error: string not found in {fpath}"
            content = content.replace(old, new, 1)
            with open(fpath, "w") as f:
                f.write(content)
            return f"Edited {fpath}"

        elif name == "list_files":
            dpath = args.get("path", ".")
            if not os.path.isabs(dpath):
                dpath = os.path.join(WORKDIR, dpath)
            depth = min(int(args.get("depth", 3)), 5)
            lines = []
            for root, dirs, files in os.walk(dpath):
                level = root.replace(dpath, "").count(os.sep)
                if level >= depth:
                    dirs.clear()
                    continue
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d not in
                          ("node_modules", "__pycache__", ".git", "venv", ".venv")]
                indent = "  " * level
                lines.append(f"{indent}{os.path.basename(root)}/")
                for f in sorted(files)[:50]:
                    if not f.startswith("."):
                        lines.append(f"{indent}  {f}")
            return "\n".join(lines[:200]) or "(empty)"

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


def run_lookup(prompt_text, max_tokens=4096, on_token=None):
    """Run llama-lookup subprocess. Returns (output, stats)."""
    prompt_len = len(prompt_text)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt_text)
        prompt_file = f.name

    try:
        cmd = [
            LOOKUP_BIN,
            "-m", MODEL,
            "-ngl", "99",
            "-t", "16",
            "-lcd", LOOKUP_CACHE,
            "--draft", "16",
            "--temp", "0.3",
            "-f", prompt_file,
            "-n", str(max_tokens),
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )

        raw = []
        skip_chars = prompt_len  # Skip the prompt echo
        char_count = 0

        # Read character by character for real-time streaming
        while True:
            char = proc.stdout.read(1)
            if not char:
                break

            char_count += 1

            # Skip prompt echo (llama-lookup prints the prompt first)
            if char_count <= skip_chars:
                continue

            raw.append(char)
            if on_token:
                on_token(char)

        # Read stderr for stats
        stderr = proc.stderr.read()
        proc.wait()

        stats = {}
        for line in stderr.split("\n"):
            if "prompt eval" in line and "tokens per second" in line:
                m = re.search(r"([\d.]+) tokens per second", line)
                if m:
                    stats["prompt_tps"] = float(m.group(1))
            if "eval time" in line and "tokens per second" in line:
                m = re.search(r"([\d.]+) tokens per second", line)
                if m:
                    stats["gen_tps"] = float(m.group(1))
            if "total time" in line:
                m = re.search(r"total time =\s+([\d.]+) ms /\s+(\d+)", line)
                if m:
                    stats["total_ms"] = float(m.group(1))
                    stats["total_tokens"] = int(m.group(2))
            if "accept" in line:
                m = re.search(r"accept\s+=\s+([\d.]+)", line)
                if m:
                    stats["accept_rate"] = float(m.group(1))

        return "".join(raw), stats

    finally:
        os.unlink(prompt_file)


def generate_with_tools(messages, on_token=None, on_tool=None, on_stats=None, max_iterations=8):
    """
    Generate a response with tool calling support using lookup decoding.

    Args:
        messages: List of {role, content} dicts
        on_token: Callback(char) for each generated character
        on_tool: Callback(tool_name, args, result) for each tool call
        on_stats: Callback(stats_dict) with generation statistics
        max_iterations: Max tool call rounds
    """
    all_tool_calls = []
    conversation = list(messages)

    for iteration in range(max_iterations):
        prompt = format_chat(conversation)

        # Run lookup decoding
        raw_output, stats = run_lookup(
            prompt,
            max_tokens=4096,
            on_token=on_token if iteration == max_iterations - 1 or True else None
        )

        # Clean up output — remove special tokens and thinking blocks
        output = raw_output
        # Remove Gemma special tokens
        for tok in ['<bos>', '<turn|>', '<|turn>', '<channel|>', '<|channel>', '<|channel',
                     '<tool_call|>', '<|tool_call>', '<|tool_response>', '<tool_response>',
                     'channel|>', '|>']:
            output = output.replace(tok, '')
        # Remove thinking blocks (various formats)
        think_match = re.search(r'(?:\[Start thinking\]|thought\n|lthought\n)(.*?)(?:\[End thinking\]|(?=<\|tool_call>)|$)', output, re.DOTALL)
        thinking = ""
        if think_match:
            thinking = think_match.group(1).strip()
            output = output[:think_match.start()] + output[think_match.end():]
        # Clean leading/trailing whitespace and special chars
        output = re.sub(r'^[\s\n|>l]+', '', output)
        output = re.sub(r'<turn\|?\>.*$', '', output, flags=re.DOTALL)
        output = output.strip()

        # Check for tool calls — try native format first, then JSON
        tool_name = None
        tool_args = {}
        # Search raw output (before cleaning) for tool calls
        tool_match = NATIVE_TOOL_RE.search(raw_output)

        if tool_match:
            tool_name = tool_match.group(1)
            # Parse Gemma native args: key: "value" or key="value" or key: value
            args_str = tool_match.group(2)
            try:
                tool_args = json.loads("{" + args_str + "}")
            except json.JSONDecodeError:
                # Parse key=val, key: val, key="val", key: "val" patterns
                for kv in re.finditer(r'(\w+)\s*[=:]\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?)(?:\s*[,\)]|$))', args_str):
                    val = kv.group(2) or kv.group(3) or kv.group(4) or ""
                    tool_args[kv.group(1)] = val
        else:
            tool_match = JSON_TOOL_RE.search(raw_output)
            if tool_match:
                tool_name = tool_match.group(1)
                try:
                    tool_args = json.loads(tool_match.group(2))
                except json.JSONDecodeError:
                    tool_args = {}

        if tool_name:
            tool_result = exec_tool(tool_name, tool_args)
            all_tool_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result[:2000]
            })

            if on_tool:
                on_tool(tool_name, tool_args, tool_result)

            # Get text before the tool call
            pre_tool_text = output[:tool_match.start()].strip()

            # Add to conversation for next iteration
            conversation.append({"role": "assistant", "content": pre_tool_text + f"\n[Called {tool_name}]"})
            conversation.append({"role": "tool_result", "content": tool_result})

            continue  # Next iteration

        # No tool call — this is the final response
        if on_stats:
            on_stats(stats)

        return {
            "content": output.strip(),
            "reasoning": thinking,
            "tool_calls": all_tool_calls,
            "stats": stats
        }

    # Max iterations reached
    return {
        "content": "(max tool iterations reached)",
        "reasoning": "",
        "tool_calls": all_tool_calls,
        "stats": {}
    }


# CLI test
if __name__ == "__main__":
    import sys

    print(f"Lookup engine: {LOOKUP_BIN}")
    print(f"Model: {MODEL}")
    print(f"Workdir: {WORKDIR}")
    print()

    def on_tok(c):
        sys.stdout.write(c)
        sys.stdout.flush()

    def on_tool(name, args, result):
        print(f"\n[TOOL] {name}({args}) → {result[:200]}")

    messages = [{"role": "user", "content": " ".join(sys.argv[1:]) or "List the files in this directory and tell me what you see"}]

    print("Generating...\n")
    t0 = time.time()
    result = generate_with_tools(messages, on_token=on_tok, on_tool=on_tool)
    elapsed = time.time() - t0

    print(f"\n\n--- Stats ---")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tools: {len(result['tool_calls'])}")
    if result['stats']:
        print(f"Accept rate: {result['stats'].get('accept_rate', '?')}%")
        print(f"Gen speed: {result['stats'].get('gen_tps', '?')} t/s")
        print(f"Effective: {result['stats'].get('total_tokens', 0) / (result['stats'].get('total_ms', 1) / 1000):.0f} t/s")
