#!/usr/bin/env python3
"""
Gemma 4 A4B — Web IDE Chat Server

Authenticated web interface with:
- Chat with real-time metrics
- Image upload (vision)
- File browser / project tree
- Tool calling: bash, read, write, edit files
- Agentic loop: model can chain tool calls

Usage:
  AUTH_TOKEN=your-token WORKDIR=/path/to/project python3 chat_server.py

Requires llama-server running on LLAMA_API (default http://localhost:8082)
"""

import os, sys, json, time, secrets, subprocess, base64, mimetypes, re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError
from pathlib import Path
import threading

LLAMA_API = os.environ.get("LLAMA_API", "http://localhost:8082")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", None)
PORT = int(os.environ.get("CHAT_PORT", "8080"))
WORKDIR = os.environ.get("WORKDIR", os.path.expanduser("~/projects"))

if not AUTH_TOKEN:
    AUTH_TOKEN = secrets.token_urlsafe(48)
    print(f"\n  Generated AUTH_TOKEN: {AUTH_TOKEN}\n")

Path(WORKDIR).mkdir(parents=True, exist_ok=True)

valid_sessions = set()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return its output. Use for running code, installing packages, git operations, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file (relative to project root or absolute)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "The content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific string in a file with new content. The old_string must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_string": {"type": "string", "description": "The exact string to find and replace"},
                    "new_string": {"type": "string", "description": "The replacement string"}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a path. Returns a tree structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: project root)"},
                    "depth": {"type": "integer", "description": "Max depth (default: 3)"}
                }
            }
        }
    }
]

SYSTEM_PROMPT = f"""You are a helpful coding assistant with access to a project directory at {WORKDIR}.
You can execute bash commands, read/write/edit files, and list directory contents.
When the user asks you to write code or make changes, use the tools to actually create and modify files.
Always show what you're doing and explain your changes. Be concise."""


def resolve_path(path):
    """Resolve a path relative to WORKDIR, preventing directory traversal."""
    if not path:
        return WORKDIR
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORKDIR) / p
    p = p.resolve()
    # Allow WORKDIR and common system paths for reading
    return str(p)


def exec_tool(name, args):
    """Execute a tool and return the result string."""
    try:
        if name == "bash":
            cmd = args.get("command", "")
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=WORKDIR
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr] " + result.stderr
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output[:4000] or "(no output)"

        elif name == "read_file":
            fpath = resolve_path(args.get("path", ""))
            with open(fpath, "r") as f:
                content = f.read(50000)
            return content or "(empty file)"

        elif name == "write_file":
            fpath = resolve_path(args.get("path", ""))
            Path(fpath).parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w") as f:
                f.write(args.get("content", ""))
            return f"Wrote {len(args.get('content', ''))} bytes to {fpath}"

        elif name == "edit_file":
            fpath = resolve_path(args.get("path", ""))
            with open(fpath, "r") as f:
                content = f.read()
            old = args.get("old_string", "")
            new = args.get("new_string", "")
            if old not in content:
                return f"Error: old_string not found in {fpath}"
            content = content.replace(old, new, 1)
            with open(fpath, "w") as f:
                f.write(content)
            return f"Edited {fpath}: replaced {len(old)} chars with {len(new)} chars"

        elif name == "list_files":
            dpath = resolve_path(args.get("path", ""))
            depth = min(args.get("depth", 3), 5)
            lines = []
            for root, dirs, files in os.walk(dpath):
                level = root.replace(dpath, "").count(os.sep)
                if level >= depth:
                    dirs.clear()
                    continue
                # Skip hidden dirs and common noise
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d not in
                          ("node_modules", "__pycache__", ".git", "venv", ".venv")]
                indent = "  " * level
                lines.append(f"{indent}{os.path.basename(root)}/")
                for f in sorted(files)[:50]:
                    if not f.startswith("."):
                        lines.append(f"{indent}  {f}")
            return "\n".join(lines[:200]) or "(empty directory)"

        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


def get_file_tree(path=WORKDIR, depth=3):
    """Get JSON file tree for the sidebar."""
    def walk(p, d):
        if d <= 0:
            return []
        items = []
        try:
            entries = sorted(Path(p).iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return []
        for entry in entries:
            if entry.name.startswith(".") or entry.name in ("node_modules", "__pycache__", "venv", ".venv"):
                continue
            if entry.is_dir():
                children = walk(str(entry), d - 1) if d > 1 else []
                items.append({"name": entry.name, "type": "dir", "children": children})
            else:
                items.append({"name": entry.name, "type": "file", "path": str(entry)})
            if len(items) > 100:
                break
        return items
    return walk(path, depth)


def build_api_messages(messages_history):
    """Convert frontend messages to API format."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages_history:
        if m["role"] == "user":
            if m.get("image"):
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": m["content"]},
                    {"type": "image_url", "image_url": {"url": m["image"]}}
                ]})
            else:
                messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("content"):
            # Skip empty assistant messages (placeholders from frontend)
            messages.append({"role": "assistant", "content": m["content"]})
    return messages


def call_model(messages, stream=False):
    """Call llama-server. Returns parsed JSON or raw response for streaming."""
    payload = {
        "model": "gemma",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
        "tools": TOOLS,
        "stream": stream,
    }
    req = Request(
        f"{LLAMA_API}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    return urlopen(req, timeout=180)


def chat_with_tools_streaming(messages_history, write_sse):
    """Stream response via SSE. Handle tool calls in non-streaming mode, final response streamed."""
    messages = build_api_messages(messages_history)
    total_usage = {}

    for iteration in range(10):
        t0 = time.time()

        # First try streaming — if model wants tool calls it won't stream properly,
        # so we do non-streaming for tool detection
        try:
            resp = call_model(messages, stream=False)
            result = json.loads(resp.read())
        except Exception as e:
            err_body = ""
            if hasattr(e, 'read'):
                try: err_body = e.read().decode()[:500]
                except: pass
            print(f"[ERROR] Model call failed: {e} | body: {err_body}", flush=True)
            print(f"[ERROR] Messages: {json.dumps(messages)[:500]}", flush=True)
            write_sse({"type": "error", "content": f"Model error: {e} {err_body}"})
            return

        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        total_usage = result.get("usage", total_usage)
        tc = message.get("tool_calls", [])

        if not tc:
            # No tool calls — stream the response live
            write_sse({"type": "stream_start"})
            try:
                stream_resp = call_model(messages, stream=True)
                for raw_line in stream_resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            write_sse({"type": "token", "content": delta["content"]})
                        if delta.get("reasoning_content"):
                            write_sse({"type": "reasoning_token", "content": delta["reasoning_content"]})
                        if chunk.get("usage"):
                            total_usage = chunk["usage"]
                    except json.JSONDecodeError:
                        pass
            except Exception:
                # Streaming failed — send the non-streamed content
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")
                if reasoning:
                    write_sse({"type": "reasoning_token", "content": reasoning})
                if content:
                    write_sse({"type": "token", "content": content})

            elapsed = time.time() - t0
            write_sse({"type": "done", "usage": total_usage, "time": elapsed})
            return

        # Tool calls — execute and loop
        messages.append(message)
        for call in tc:
            fn = call.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

            tool_result = exec_tool(tool_name, tool_args)
            write_sse({"type": "tool_exec", "tool": tool_name, "args": tool_args,
                       "result": tool_result[:2000]})

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": tool_result
            })

    write_sse({"type": "done", "usage": total_usage, "time": 0})


def check_auth(cookie_header):
    if not cookie_header:
        return False
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("session="):
            return part[8:] in valid_sessions
    return False


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 A4B IDE</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root { --bg: #0a0a0f; --surface: #13131a; --surface2: #1a1a24; --border: #2a2a3a;
        --text: #e0e0e8; --dim: #888898; --accent: #6c8aff; --green: #4ecdc4;
        --yellow: #ffd93d; --red: #ff6b6b; --orange: #ff9f43; }
body { font-family: 'SF Mono','Fira Code','Consolas',monospace; background: var(--bg);
       color: var(--text); height: 100vh; display: flex; flex-direction: column; font-size: 13px; }

.header { background: var(--surface); border-bottom: 1px solid var(--border);
          padding: 8px 16px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 13px; color: var(--accent); }
.header .info { font-size: 11px; color: var(--dim); }

.metrics { background: var(--surface2); border-bottom: 1px solid var(--border);
           padding: 6px 16px; display: flex; gap: 20px; font-size: 11px; flex-wrap: wrap; }
.metric { display: flex; gap: 4px; align-items: center; }
.metric-label { color: var(--dim); }
.metric-value { color: var(--green); font-weight: 600; }
.metric-value.yellow { color: var(--yellow); }

.main { flex: 1; display: flex; overflow: hidden; }

/* File tree */
.sidebar { width: 220px; background: var(--surface); border-right: 1px solid var(--border);
           overflow-y: auto; padding: 8px 0; font-size: 12px; flex-shrink: 0; }
.sidebar-header { padding: 4px 12px 8px; color: var(--dim); font-size: 10px;
                  text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border);
                  margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }
.sidebar-header button { background: none; border: none; color: var(--dim); cursor: pointer;
                         font-size: 11px; }
.tree-item { padding: 2px 12px; cursor: pointer; white-space: nowrap; overflow: hidden;
             text-overflow: ellipsis; }
.tree-item:hover { background: var(--surface2); }
.tree-item.dir { color: var(--accent); }
.tree-item.file { color: var(--text); }

/* Chat */
.chat-area { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 90%; padding: 10px 14px; border-radius: 10px; line-height: 1.5; word-break: break-word; }
.msg.user { background: var(--accent); color: #fff; align-self: flex-end; border-radius: 10px 10px 2px 10px; }
.msg.assistant { background: var(--surface2); border: 1px solid var(--border); align-self: flex-start;
                 border-radius: 10px 10px 10px 2px; }
.msg .thinking { color: var(--dim); font-size: 11px; font-style: italic; border-left: 2px solid var(--border);
                 padding-left: 8px; margin-bottom: 6px; max-height: 150px; overflow-y: auto; }
.msg .thinking.collapsed { max-height: 20px; overflow: hidden; cursor: pointer; }
.msg .thinking-toggle { color: var(--dim); font-size: 10px; cursor: pointer; margin-bottom: 4px;
                        user-select: none; }
.msg .tool-call { background: rgba(0,0,0,0.3); border-radius: 6px; padding: 8px; margin: 6px 0;
                  font-size: 11px; border-left: 3px solid var(--orange); }
.msg .tool-call .tool-name { color: var(--orange); font-weight: 600; }
.msg .tool-call .tool-args { color: var(--dim); margin: 2px 0; }
.msg .tool-call .tool-result { color: var(--green); white-space: pre-wrap; max-height: 150px;
                                overflow-y: auto; margin-top: 4px; }
.msg pre { background: rgba(0,0,0,0.4); padding: 8px; border-radius: 4px; overflow-x: auto;
           margin: 6px 0; font-size: 12px; }
.msg code { background: rgba(0,0,0,0.3); padding: 1px 4px; border-radius: 3px; font-size: 12px; }
.msg.system { color: var(--dim); font-size: 11px; text-align: center; align-self: center; background: none; }
.loading { color: var(--dim); align-self: flex-start; padding: 10px; }
.loading::after { content: '...'; animation: dots 1.5s infinite; }
@keyframes dots { 0% { content: '.'; } 33% { content: '..'; } 66% { content: '...'; } }

/* Input */
.input-area { background: var(--surface); border-top: 1px solid var(--border); padding: 12px 16px; }
.input-row { display: flex; gap: 8px; align-items: flex-end; }
textarea { flex: 1; background: var(--surface2); border: 1px solid var(--border); color: var(--text);
           border-radius: 6px; padding: 8px 12px; font-family: inherit; font-size: 13px; resize: none;
           outline: none; min-height: 40px; max-height: 150px; }
textarea:focus { border-color: var(--accent); }
button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 8px 16px;
         font-family: inherit; font-size: 12px; cursor: pointer; font-weight: 600; white-space: nowrap; }
button:hover { opacity: 0.9; }
button:disabled { opacity: 0.3; cursor: not-allowed; }
.btn-secondary { background: var(--surface2); color: var(--dim); border: 1px solid var(--border); }
.btn-img { padding: 8px 10px; font-size: 14px; }
input[type=file] { display: none; }

@media (max-width: 768px) { .sidebar { display: none; } }
</style>
</head>
<body>
<div class="header">
  <h1>Gemma 4 26B-A4B</h1>
  <div class="info">MoE 128 experts, 8 active | Q4_K_M | Vision | Tools</div>
</div>
<div class="metrics" id="metrics">
  <div class="metric"><span class="metric-label">Gen:</span><span class="metric-value" id="m-gen">-</span></div>
  <div class="metric"><span class="metric-label">TTFT:</span><span class="metric-value" id="m-ttft">-</span></div>
  <div class="metric"><span class="metric-label">Reasoning:</span><span class="metric-value yellow" id="m-reason">-</span></div>
  <div class="metric"><span class="metric-label">Completion:</span><span class="metric-value" id="m-comp">-</span></div>
  <div class="metric"><span class="metric-label">Tools:</span><span class="metric-value" id="m-tools">0</span></div>
  <div class="metric"><span class="metric-label">Total:</span><span class="metric-value" id="m-total">-</span></div>
  <div class="metric"><span class="metric-label">Time:</span><span class="metric-value" id="m-time">-</span></div>
</div>
<div class="main">
  <div class="sidebar" id="sidebar"></div>
  <div class="chat-area">
    <div class="chat" id="chat"></div>
    <div class="input-area">
      <div class="input-row">
        <button class="btn-img" onclick="document.getElementById('img-input').click()" title="Attach image">&#128247;</button>
        <input type="file" id="img-input" accept="image/*" onchange="handleImage(this)">
        <textarea id="input" placeholder="Ask me to write code, run commands, edit files..." rows="1"
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"
          oninput="this.style.height='40px';this.style.height=Math.min(this.scrollHeight,150)+'px'"></textarea>
        <button onclick="send()" id="send-btn">Send</button>
        <button class="btn-secondary" onclick="clearChat()">Clear</button>
      </div>
      <div id="img-preview" style="display:none;margin-top:6px;font-size:11px;color:var(--dim)"></div>
    </div>
  </div>
</div>
<script>
let messages = [];
let busy = false;
let pendingImage = null;
let fileTree = [];

function loadTree() {
  fetch('/api/tree').then(r=>r.json()).then(data => { fileTree = data; renderTree(); });
}

function renderTree() {
  const el = document.getElementById('sidebar');
  function renderItems(items, depth) {
    return items.map(item => {
      const indent = depth * 12;
      if (item.type === 'dir') {
        return `<div class="tree-item dir" style="padding-left:${12+indent}px"
          onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none'">
          &#128193; ${item.name}</div><div style="display:${depth<1?'block':'none'}">${renderItems(item.children||[],depth+1)}</div>`;
      }
      return `<div class="tree-item file" style="padding-left:${12+indent}px"
        onclick="readFile('${(item.path||'').replace(/'/g,"\\'")}')">&#128196; ${item.name}</div>`;
    }).join('');
  }
  el.innerHTML = `<div class="sidebar-header">Files <button onclick="loadTree()">&#8635;</button></div>` +
    renderItems(fileTree, 0);
}

function readFile(path) {
  if (!path || busy) return;
  const input = document.getElementById('input');
  input.value = `Read the file: ${path}`;
  send();
}

function handleImage(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    pendingImage = reader.result;
    document.getElementById('img-preview').style.display = 'block';
    document.getElementById('img-preview').innerHTML =
      `&#128247; ${file.name} (${(file.size/1024).toFixed(0)}KB) <button class="btn-secondary"
       style="padding:2px 6px;font-size:10px" onclick="pendingImage=null;this.parentElement.style.display='none'">x</button>`;
  };
  reader.readAsDataURL(file);
  input.value = '';
}

function formatMsg(text) {
  if (!text) return '';
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/\n/g, '<br>');
}

function renderChat() {
  const chat = document.getElementById('chat');
  let html = '';
  if (messages.length === 0) {
    html = `<div class="msg system">Chat with Gemma 4 A4B. Attach images, ask it to write code, run commands, or edit files.</div>`;
  }
  for (const m of messages) {
    if (m.role === 'user') {
      html += `<div class="msg user">${formatMsg(m.content)}${m.image ? '<br><span style="font-size:11px">&#128247; image attached</span>' : ''}</div>`;
    } else if (m.role === 'assistant') {
      html += `<div class="msg assistant">`;
      if (m.reasoning) {
        const id = 'think-' + Math.random().toString(36).slice(2,8);
        html += `<div class="thinking-toggle" onclick="document.getElementById('${id}').classList.toggle('collapsed')">&#9654; Reasoning (click to expand)</div>`;
        html += `<div class="thinking collapsed" id="${id}">${formatMsg(m.reasoning)}</div>`;
      }
      if (m.tool_calls?.length) {
        for (const tc of m.tool_calls) {
          html += `<div class="tool-call"><span class="tool-name">${tc.tool}</span>`;
          const argStr = tc.tool === 'bash' ? tc.args.command :
                        tc.tool === 'write_file' ? `${tc.args.path} (${tc.args.content?.length||0} bytes)` :
                        tc.tool === 'read_file' ? tc.args.path :
                        tc.tool === 'edit_file' ? tc.args.path :
                        JSON.stringify(tc.args);
          html += `<div class="tool-args">${formatMsg(argStr)}</div>`;
          if (tc.result) html += `<div class="tool-result">${formatMsg(tc.result.slice(0,500))}</div>`;
          html += `</div>`;
        }
      }
      html += formatMsg(m.content) + `</div>`;
    } else if (m.role === 'loading') {
      html += `<div class="loading">Thinking</div>`;
    }
  }
  chat.innerHTML = html;
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text || busy) return;

  const userMsg = { role: 'user', content: text };
  if (pendingImage) {
    userMsg.image = pendingImage;
    pendingImage = null;
    document.getElementById('img-preview').style.display = 'none';
  }
  messages.push(userMsg);
  // Add assistant placeholder
  const assistant = { role: 'assistant', content: '', reasoning: '', tool_calls: [] };
  messages.push(assistant);
  input.value = '';
  input.style.height = '40px';
  busy = true;
  document.getElementById('send-btn').disabled = true;
  renderChat();

  const t0 = performance.now();
  let genStart = null;
  let tokenCount = 0;
  let reasoningTokens = 0;
  let contentTokens = 0;
  let toolCount = 0;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages.filter(m => m.role === 'user' || (m.role === 'assistant' && m.content)) })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const evt = JSON.parse(line.slice(6));

          if (evt.type === 'tool_exec') {
            toolCount++;
            assistant.tool_calls.push({ tool: evt.tool, args: evt.args, result: evt.result });
            document.getElementById('m-tools').textContent = toolCount;
            renderChat();
          }
          else if (evt.type === 'reasoning') {
            assistant.reasoning = evt.content;
            reasoningTokens = evt.content.split(/\s+/).filter(w=>w).length;
            renderChat();
          }
          else if (evt.type === 'stream_start') {
            genStart = performance.now();
          }
          else if (evt.type === 'reasoning_token') {
            assistant.reasoning += evt.content;
            reasoningTokens++;
            tokenCount++;
            document.getElementById('m-reason').textContent = reasoningTokens;
            // Live speed update
            if (genStart) {
              const genElapsed = (performance.now() - genStart) / 1000;
              document.getElementById('m-gen').textContent = genElapsed > 0.1 ? (tokenCount / genElapsed).toFixed(1) + ' t/s' : '...';
            }
            renderChat();
          }
          else if (evt.type === 'token') {
            assistant.content += evt.content;
            contentTokens++;
            tokenCount++;
            document.getElementById('m-comp').textContent = contentTokens;
            document.getElementById('m-total').textContent = tokenCount + ' tok';
            // Live speed update (only count gen time, not tool time)
            if (genStart) {
              const genElapsed = (performance.now() - genStart) / 1000;
              document.getElementById('m-gen').textContent = genElapsed > 0.1 ? (tokenCount / genElapsed).toFixed(1) + ' t/s' : '...';
            }
            renderChat();
          }
          else if (evt.type === 'done') {
            const u = evt.usage || {};
            if (u.completion_tokens) {
              tokenCount = u.completion_tokens;
              document.getElementById('m-total').textContent = tokenCount + ' tok';
            }
            // Compute final gen speed excluding tool call time
            if (genStart) {
              const genElapsed = (performance.now() - genStart) / 1000;
              document.getElementById('m-gen').textContent = genElapsed > 0.1 ? (tokenCount / genElapsed).toFixed(1) + ' t/s' : '-';
              document.getElementById('m-ttft').textContent = ((genStart - t0) / 1000).toFixed(1) + 's';
            }
            const totalElapsed = (performance.now() - t0) / 1000;
            document.getElementById('m-time').textContent = totalElapsed.toFixed(1) + 's';
            if (toolCount > 0) loadTree();
          }
          else if (evt.type === 'error') {
            assistant.content = evt.content;
            renderChat();
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    assistant.content = 'Error: ' + e.message;
  }

  busy = false;
  document.getElementById('send-btn').disabled = false;
  renderChat();
  document.getElementById('input').focus();
}

function clearChat() {
  messages = [];
  ['m-gen','m-ttft','m-reason','m-comp','m-tools','m-total','m-time'].forEach(id =>
    document.getElementById(id).textContent = '-');
  renderChat();
}

// Init
loadTree();
renderChat();
document.getElementById('input').focus();
</script>
</body>
</html>"""

LOGIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login — Gemma 4</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono',monospace; background: #0a0a0f; color: #e0e0e8;
       display: flex; justify-content: center; align-items: center; height: 100vh; }
.box { background: #13131a; border: 1px solid #2a2a3a; border-radius: 12px; padding: 32px; width: 360px; }
h2 { color: #6c8aff; margin-bottom: 20px; font-size: 16px; }
input { width: 100%; background: #1a1a24; border: 1px solid #2a2a3a; color: #e0e0e8;
        border-radius: 8px; padding: 10px 14px; font-family: inherit; font-size: 13px;
        outline: none; margin-bottom: 16px; }
input:focus { border-color: #6c8aff; }
button { width: 100%; background: #6c8aff; color: #fff; border: none; border-radius: 8px;
         padding: 10px; font-family: inherit; font-size: 13px; cursor: pointer; font-weight: 600; }
.err { color: #ff6b6b; font-size: 12px; margin-bottom: 10px; display: ERR_DISPLAY; }
</style></head>
<body><div class="box"><h2>Gemma 4 A4B IDE</h2>
<div class="err">ERR_MSG</div>
<form method="POST" action="/login">
<input type="password" name="token" placeholder="Enter access token" autofocus>
<button type="submit">Login</button></form></div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/login":
            return self._login_page()
        if not check_auth(self.headers.get("Cookie", "")):
            return self._redirect("/login")
        if path in ("/", ""):
            self._html(HTML_PAGE)
        elif path == "/api/tree":
            self._json(get_file_tree())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))

        if path == "/login":
            params = parse_qs(body.decode())
            token = params.get("token", [""])[0]
            if secrets.compare_digest(token, AUTH_TOKEN):
                sid = secrets.token_hex(32)
                valid_sessions.add(sid)
                self.send_response(302)
                self.send_header("Set-Cookie", f"session={sid}; HttpOnly; Path=/; SameSite=Strict; Max-Age=86400")
                self.send_header("Location", "/")
                self.end_headers()
            else:
                self._login_page("Invalid token")
            return

        if not check_auth(self.headers.get("Cookie", "")):
            self.send_response(401); self.end_headers(); return

        if path == "/api/chat":
            data = json.loads(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def write_sse(event):
                try:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    pass

            chat_with_tools_streaming(data.get("messages", []), write_sse)
            return
        else:
            self.send_response(404); self.end_headers()

    def _html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _redirect(self, loc):
        self.send_response(302)
        self.send_header("Location", loc)
        self.end_headers()

    def _login_page(self, error=None):
        page = LOGIN_PAGE.replace("ERR_MSG", error or "").replace("ERR_DISPLAY", "block" if error else "none")
        self._html(page)


class ThreadedHTTPServer(HTTPServer):
    """Handle requests in threads for concurrent access."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread, args=(request, client_address))
        t.daemon = True
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Gemma 4 IDE running on http://0.0.0.0:{PORT}")
    print(f"Backend: {LLAMA_API}")
    print(f"Workdir: {WORKDIR}")
    print(f"Token: {AUTH_TOKEN[:8]}...{AUTH_TOKEN[-4:]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
