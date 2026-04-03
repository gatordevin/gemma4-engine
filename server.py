"""
Gemma4 ↔ Anthropic API translation server.
Translates between Anthropic Messages API (what claw-code speaks) and
Gemma4's native tool-calling format (<|tool_call>call:name{...}<tool_call|>).

Usage:
  python gemma4_server.py

Then:
  ANTHROPIC_BASE_URL=http://localhost:8642 ANTHROPIC_API_KEY=local claw "your prompt"
"""

import json
import time
import uuid
import re
import sys
from pathlib import Path

import torch
from flask import Flask, request, Response
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from engine_turbo import Gemma4TurboEngine, DEVICE

PORT = 8642
app = Flask(__name__)
engine = None
processor = None  # HF processor for chat template (handles tool format natively)

# Gemma4 special token IDs
TOK_TOOL_CALL_START = 48   # <|tool_call>
TOK_TOOL_CALL_END = 49     # <tool_call|>
TOK_TOOL_RESP_START = 50   # <|tool_response>
TOK_TOOL_RESP_END = 51     # <tool_response|>
TOK_TURN_END = 106         # <turn|>  (also EOS)
TOK_EOS = 1


# ── Anthropic → Gemma4 format translation ──────────────────────────────────────

def anthropic_tools_to_gemma(tools):
    """Convert Anthropic tool definitions to Gemma4 processor format."""
    gemma_tools = []
    for t in tools:
        # Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
        gemma_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
        })
    return gemma_tools


def anthropic_messages_to_gemma(messages, system_text=""):
    """Convert Anthropic message history to Gemma4 chat format.

    Anthropic format:
      {"role": "user", "content": "..."}
      {"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]}
      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}

    Gemma4 format:
      {"role": "user", "content": "..."}
      {"role": "assistant", "tool_calls": [...], "tool_responses": [...]}
    """
    gemma_msgs = []

    # System instructions as first user+assistant exchange to establish context
    if system_text:
        # Truncate to avoid overwhelming the model
        if len(system_text) > 1500:
            system_text = system_text[:800] + "\n...(truncated)...\n" + system_text[-400:]
        gemma_msgs.append({"role": "user", "content": "System instructions: " + system_text})
        gemma_msgs.append({"role": "assistant", "content": "Understood. I will follow these instructions and use the available tools when needed. What would you like me to do?"})

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                gemma_msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Could contain tool_result blocks
                text_parts = []
                tool_results_for_prev = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_result":
                            # This is a tool result — attach to previous assistant message
                            result_content = block.get("content", "")
                            if isinstance(result_content, list):
                                result_content = "\n".join(
                                    b.get("text", "") for b in result_content if isinstance(b, dict)
                                )
                            tool_name = block.get("_tool_name", "unknown")
                            # Find matching tool_use in previous assistant message
                            tool_use_id = block.get("tool_use_id", "")
                            if gemma_msgs and gemma_msgs[-1].get("role") == "assistant":
                                prev = gemma_msgs[-1]
                                # Find the tool call name by ID
                                for tc in prev.get("_anthropic_tool_uses", []):
                                    if tc.get("id") == tool_use_id:
                                        tool_name = tc["name"]
                                        break
                                # Add tool response to previous assistant message
                                if "tool_responses" not in prev:
                                    prev["tool_responses"] = []
                                prev["tool_responses"].append({
                                    "name": tool_name,
                                    "response": result_content,
                                })

                if text_parts:
                    gemma_msgs.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == "assistant":
            if isinstance(content, str):
                gemma_msgs.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                anthropic_tool_uses = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "function": {
                                    "name": block["name"],
                                    "arguments": block.get("input", {}),
                                }
                            })
                            anthropic_tool_uses.append({
                                "id": block.get("id", ""),
                                "name": block["name"],
                            })

                entry = {"role": "assistant"}
                if text_parts:
                    entry["content"] = "\n".join(text_parts)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                    entry["_anthropic_tool_uses"] = anthropic_tool_uses
                gemma_msgs.append(entry)

        i += 1

    return gemma_msgs


# ── Small-model compaction ─────────────────────────────────────────────────────

# Tools the model actually needs for code editing / command execution
CORE_TOOLS = {
    "bash", "read_file", "write_file", "edit_file",
    "glob_search", "grep_search",
}

# Compact system prompt optimized for a 4B model
# KEY: Must override the model's instinct to say "I can't execute code"
COMPACT_SYSTEM = """\
You are a coding agent running on the user's machine. You HAVE full access to their filesystem and can run commands.

CRITICAL: You MUST use tools to do work. NEVER say "I cannot execute code" or "I don't have access" — you DO have access. Use the tools.

To run commands: use the bash tool. To read files: use read_file. To write files: use write_file.

When asked to create a project or run code:
1. Use write_file to create the files
2. Use bash to run them (e.g. bash with command "python3 filename.py")
3. Read the output, fix errors if any

Keep responses short. Show results, not explanations."""


def compact_for_small_model(gemma_msgs, gemma_tools, original_system):
    """Replace bloated system prompt and 48 tools with focused versions for 4B model."""

    # 1. Replace system prompt
    # Find and replace the system instruction in gemma_msgs
    new_msgs = []
    replaced_system = False
    for m in gemma_msgs:
        if m.get("role") == "user" and not replaced_system and "System instructions:" in m.get("content", ""):
            # Extract just the environment context from original
            env_lines = []
            for line in original_system.split("\n"):
                if any(k in line.lower() for k in ["working directory:", "date:", "platform:", "today"]):
                    env_lines.append(line.strip())

            env_context = "\n".join(env_lines) if env_lines else ""
            compact = COMPACT_SYSTEM
            if env_context:
                compact += f"\n\nEnvironment:\n{env_context}"

            new_msgs.append({"role": "user", "content": "System instructions: " + compact})
            replaced_system = True
        elif m.get("role") == "assistant" and m.get("content") == "Understood. I will follow these instructions and use the available tools when needed. What would you like me to do?":
            new_msgs.append({"role": "assistant", "content": "Ready. I'll use the available tools to help you."})
        else:
            new_msgs.append(m)

    # 2. Filter tools to core set the model can reliably use
    if gemma_tools:
        core = []
        for t in gemma_tools:
            name = t.get("function", {}).get("name", "")
            if name.lower().replace("_", "") in {n.lower().replace("_", "") for n in CORE_TOOLS}:
                # Also shorten descriptions
                func = t["function"]
                short_desc = func.get("description", "")
                if len(short_desc) > 60:
                    short_desc = short_desc[:60].rsplit(" ", 1)[0]
                core.append({
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": short_desc,
                        "parameters": func.get("parameters", {}),
                    }
                })
        gemma_tools = core if core else gemma_tools

    return new_msgs, gemma_tools


# ── Gemma4 → Anthropic format translation ──────────────────────────────────────

def parse_gemma_tool_calls(raw_text):
    """Parse Gemma4's native tool call format from raw output (with special tokens).

    Gemma4 outputs: call:func_name{arg:<|"|>value<|"|>,...}
    With special tokens stripped, it looks like: call:func_name{arg:value,...}

    We need to parse the raw token IDs to detect tool calls properly.
    """
    # Pattern for the text representation (after decode with skip_special_tokens=False)
    # The model outputs: <|tool_call>call:NAME{key:<|"|>val<|"|>,...}<tool_call|>
    # With skip_special_tokens=True, we get: call:NAME{key:val,...}

    calls = []
    # Try to match: call:FUNCNAME{...}
    pattern = r'call:(\w+)\{(.*?)\}'
    for m in re.finditer(pattern, raw_text, re.DOTALL):
        func_name = m.group(1)
        args_str = m.group(2)

        # Parse args: key:value pairs (values may be quoted or not)
        args = {}
        # Handle both <|"|> delimited and plain values
        arg_pattern = r'(\w+):\s*(?:<\|\"\|>(.+?)<\|\"\|>|"([^"]*?)"|([^,}]+))'
        for am in re.finditer(arg_pattern, args_str, re.DOTALL):
            key = am.group(1)
            val = am.group(2) or am.group(3) or am.group(4) or ""
            args[key] = val.strip()

        if not args and args_str.strip():
            # Fallback: try simple key:value parsing
            for part in args_str.split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    args[k.strip()] = v.strip().strip('"')

        calls.append({"name": func_name, "input": args})

    return calls


def gemma_output_to_anthropic_content(raw_text, token_ids):
    """Convert Gemma4 output to Anthropic content blocks.

    Returns list of content blocks:
      [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]
    """
    content_blocks = []

    # Check if output contains tool calls (token ID 48 = <|tool_call>)
    has_tool_call = TOK_TOOL_CALL_START in token_ids

    if has_tool_call:
        # Parse tool calls from the raw text
        calls = parse_gemma_tool_calls(raw_text)

        # Extract any text before the tool call (strip special tokens)
        text_before = raw_text.split("call:")[0].strip()
        # Remove special token artifacts
        for tok in ["<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|turn>", "<turn|>"]:
            text_before = text_before.replace(tok, "")
        text_before = text_before.strip()
        if text_before:
            content_blocks.append({"type": "text", "text": text_before})

        for call in calls:
            content_blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": call["name"],
                "input": call["input"],
            })

        return content_blocks, "tool_use"
    else:
        # Plain text response
        # Clean up any leftover special token artifacts
        clean = raw_text.strip()
        if clean:
            content_blocks.append({"type": "text", "text": clean})
        return content_blocks, "end_turn"


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_with_tools(gemma_messages, gemma_tools, max_tokens=2048, temperature=0.7):
    """Generate using Gemma4's native tool-calling format via the processor."""
    global engine, processor

    engine.cache.reset()

    # Use processor's chat template to render tools natively
    text = processor.apply_chat_template(
        gemma_messages,
        tools=gemma_tools if gemma_tools else None,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = processor.tokenizer.encode(text, return_tensors="pt").to(DEVICE)

    # Truncate if needed
    max_input = engine.max_seq_len - max_tokens
    if input_ids.shape[1] > max_input:
        input_ids = input_ids[:, -max_input:]

    B, S = input_ids.shape
    pos_ids = torch.arange(S, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits = engine.forward(input_ids, pos_ids)

    next_logits = logits[:, -1:, :].squeeze(0)
    del logits
    torch.cuda.empty_cache()

    token_ids = []
    for _ in range(max_tokens):
        if temperature > 0:
            probs = torch.nn.functional.softmax(next_logits / temperature, dim=-1)
            v, idx = torch.topk(probs, 50)
            probs = torch.zeros_like(probs).scatter_(1, idx, v)
            probs /= probs.sum(dim=-1, keepdim=True)
            nt = torch.multinomial(probs, 1)
        else:
            nt = next_logits.argmax(dim=-1, keepdim=True)

        tid = nt.item()
        token_ids.append(tid)

        # Stop conditions
        if tid == TOK_EOS:
            break
        if tid == TOK_TURN_END:  # <turn|> = end of model turn
            break
        if tid == TOK_TOOL_RESP_START:  # <|tool_response> = model wants tool result
            break

        with torch.no_grad():
            next_logits = engine.decode_one_token(
                torch.tensor([tid], device=DEVICE),
                torch.tensor([engine.cache.seq_len], device=DEVICE),
            )

    # Decode with special tokens visible for tool call parsing
    raw_text = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
    clean_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)

    return raw_text, clean_text, token_ids, S, len(token_ids)


# ── API endpoint ───────────────────────────────────────────────────────────────

@app.route("/v1/messages", methods=["POST"])
def messages_endpoint():
    body = request.get_json(force=True)
    system = body.get("system", "")
    msgs = body.get("messages", [])
    tools = body.get("tools")
    max_tokens = min(body.get("max_tokens", 2048), 2048)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    # Translate Anthropic → Gemma4 format
    system_text = ""
    if system:
        if isinstance(system, list):
            system_text = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
        else:
            system_text = system

    gemma_msgs = anthropic_messages_to_gemma(msgs, system_text)
    gemma_tools = anthropic_tools_to_gemma(tools) if tools else None

    # For small models: replace bloated system prompt with a focused one
    gemma_msgs, gemma_tools = compact_for_small_model(gemma_msgs, gemma_tools, system_text)

    n_tools = len(tools) if tools else 0
    print(f"  Request: {len(msgs)} msgs, {n_tools} tools, stream={stream}")

    # Dump real requests (not health pings) for debugging
    if tools and not hasattr(messages_endpoint, '_dumped'):
        messages_endpoint._dumped = True
        with open("/tmp/claw_request_dump.json", "w") as f:
            json.dump(body, f, indent=2, default=str)
        print(f"  [Dumped request: {len(tools)} tools, {len(system_text)} char system]")
        print(f"  Tool names: {[t['name'] for t in tools]}")

    # Generate
    t0 = time.time()
    raw_text, clean_text, token_ids, in_tok, out_tok = generate_with_tools(
        gemma_msgs, gemma_tools, max_tokens, temperature
    )
    dt = time.time() - t0

    # Translate Gemma4 → Anthropic format
    content_blocks, stop_reason = gemma_output_to_anthropic_content(raw_text, token_ids)

    has_tools = any(b.get("type") == "tool_use" for b in content_blocks)
    tool_names = [b["name"] for b in content_blocks if b.get("type") == "tool_use"]
    print(f"  Response: {out_tok} tok, {dt:.1f}s, {out_tok/max(dt,0.01):.0f} tok/s"
          f"{' TOOL_USE: ' + ','.join(tool_names) if has_tools else ''}")
    if not has_tools:
        print(f"    text: {clean_text[:100]}...")

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage = {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    if stream:
        return Response(
            _sse_stream(msg_id, content_blocks, stop_reason, usage),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    else:
        return {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": content_blocks, "model": "gemma-4-e4b",
            "stop_reason": stop_reason, "stop_sequence": None, "usage": usage,
        }


def _sse_stream(msg_id, content_blocks, stop_reason, usage):
    """Generate SSE events in Anthropic streaming format."""
    # message_start
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':'gemma-4-e4b','stop_reason':None,'stop_sequence':None,'usage':usage}})}\n\n"

    for idx, block in enumerate(content_blocks):
        if block["type"] == "text":
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'text','text':''}})}\n\n"
            text = block["text"]
            for i in range(0, len(text), 30):
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'text_delta','text':text[i:i+30]}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"

        elif block["type"] == "tool_use":
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'tool_use','id':block['id'],'name':block['name'],'input':{}}})}\n\n"
            # Send input as a single JSON delta
            input_json = json.dumps(block["input"])
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'input_json_delta','partial_json':input_json}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"

    # message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':usage})}\n\n"

    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"


if __name__ == "__main__":
    print("=== Gemma4 ↔ Anthropic Translation Server ===")
    print("Loading Gemma4 engine + processor...")
    engine = Gemma4TurboEngine(max_seq_len=4096)
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
    print(f"\nStarting on http://localhost:{PORT}")
    print(f"Connect claw-code:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PORT} ANTHROPIC_API_KEY=local claw \"your prompt\"")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
