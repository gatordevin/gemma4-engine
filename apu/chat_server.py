#!/usr/bin/env python3
"""
Gemma 4 A4B — Web Chat Server with Real-Time Metrics

Authenticated web interface for chatting with the local Gemma 4 model.
Shows live speed metrics: prompt eval, reasoning tokens, generation speed,
cache utilization, and token breakdown.

Usage:
  AUTH_TOKEN=your-long-secure-password python3 chat_server.py
  # Then open http://localhost:8080 in your browser

Requires the llama-server running on LLAMA_API (default http://localhost:8082)
"""

import os
import json
import time
import hashlib
import secrets
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError
import ssl
import threading

# Config
LLAMA_API = os.environ.get("LLAMA_API", "http://localhost:8082")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", None)
PORT = int(os.environ.get("CHAT_PORT", "8080"))
SESSION_SECRET = secrets.token_hex(32)

if not AUTH_TOKEN:
    AUTH_TOKEN = secrets.token_urlsafe(48)
    print(f"\n{'='*60}")
    print(f"  No AUTH_TOKEN set. Generated one:")
    print(f"  {AUTH_TOKEN}")
    print(f"{'='*60}\n")

# Simple session store
valid_sessions = set()


def check_auth(cookie_header):
    if not cookie_header:
        return False
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("session="):
            return part[8:] in valid_sessions
    return False


def create_session():
    sid = secrets.token_hex(32)
    valid_sessions.add(sid)
    return sid


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 A4B Chat</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root { --bg: #0a0a0f; --surface: #13131a; --surface2: #1a1a24; --border: #2a2a3a;
        --text: #e0e0e8; --dim: #888898; --accent: #6c8aff; --accent2: #4ecdc4;
        --green: #4ecdc4; --yellow: #ffd93d; --red: #ff6b6b; }
body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: var(--bg);
       color: var(--text); height: 100vh; display: flex; flex-direction: column; }

/* Header */
.header { background: var(--surface); border-bottom: 1px solid var(--border);
          padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 14px; font-weight: 600; color: var(--accent); }
.header .model-info { font-size: 11px; color: var(--dim); }

/* Metrics bar */
.metrics { background: var(--surface2); border-bottom: 1px solid var(--border);
           padding: 8px 20px; display: flex; gap: 24px; font-size: 11px; overflow-x: auto; }
.metric { display: flex; flex-direction: column; gap: 2px; min-width: 100px; }
.metric-label { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { color: var(--green); font-weight: 600; font-size: 13px; }
.metric-value.yellow { color: var(--yellow); }
.metric-value.red { color: var(--red); }

/* Chat area */
.chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 85%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 13px;
       white-space: pre-wrap; word-break: break-word; }
.msg.user { background: var(--accent); color: #fff; align-self: flex-end; border-radius: 12px 12px 4px 12px; }
.msg.assistant { background: var(--surface2); border: 1px solid var(--border); align-self: flex-start;
                 border-radius: 12px 12px 12px 4px; }
.msg.assistant .thinking { color: var(--dim); font-style: italic; font-size: 12px;
                           border-left: 2px solid var(--border); padding-left: 10px; margin-bottom: 8px; }
.msg.system { background: transparent; color: var(--dim); font-size: 11px; text-align: center;
              align-self: center; }
.msg code, .msg pre { background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-size: 12px; }
.msg pre { padding: 10px; margin: 8px 0; overflow-x: auto; display: block; }

/* Streaming indicator */
.streaming::after { content: '|'; animation: blink 0.8s infinite; color: var(--accent); }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

/* Input */
.input-area { background: var(--surface); border-top: 1px solid var(--border); padding: 16px 20px; }
.input-row { display: flex; gap: 10px; }
textarea { flex: 1; background: var(--surface2); border: 1px solid var(--border); color: var(--text);
           border-radius: 8px; padding: 10px 14px; font-family: inherit; font-size: 13px; resize: none;
           outline: none; min-height: 44px; max-height: 200px; }
textarea:focus { border-color: var(--accent); }
button { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 10px 20px;
         font-family: inherit; font-size: 13px; cursor: pointer; font-weight: 600; }
button:hover { opacity: 0.9; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
.clear-btn { background: var(--surface2); color: var(--dim); border: 1px solid var(--border); }

/* Login */
.login { display: flex; justify-content: center; align-items: center; height: 100vh;
         background: var(--bg); }
.login-box { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
             padding: 32px; width: 360px; }
.login-box h2 { color: var(--accent); margin-bottom: 20px; font-size: 16px; }
.login-box input { width: 100%; background: var(--surface2); border: 1px solid var(--border);
                   color: var(--text); border-radius: 8px; padding: 10px 14px; font-family: inherit;
                   font-size: 13px; outline: none; margin-bottom: 16px; }
.login-box input:focus { border-color: var(--accent); }
.login-box button { width: 100%; }
.login-error { color: var(--red); font-size: 12px; margin-bottom: 10px; }
</style>
</head>
<body>
<div id="app"></div>
<script>
const API = '';
let messages = [];
let streaming = false;
let metrics = { prompt_tps: '-', gen_tps: '-', reasoning_tok: '-', completion_tok: '-',
                total_tok: '-', ttft: '-', total_time: '-', cached: '-' };

function render() {
  const app = document.getElementById('app');
  app.innerHTML = `
    <div class="header">
      <h1>Gemma 4 26B-A4B</h1>
      <div class="model-info">MoE 128 experts | TurboQuant | Vision</div>
    </div>
    <div class="metrics">
      <div class="metric"><span class="metric-label">Prompt Eval</span><span class="metric-value">${metrics.prompt_tps} t/s</span></div>
      <div class="metric"><span class="metric-label">Generation</span><span class="metric-value">${metrics.gen_tps} t/s</span></div>
      <div class="metric"><span class="metric-label">TTFT</span><span class="metric-value">${metrics.ttft}</span></div>
      <div class="metric"><span class="metric-label">Reasoning</span><span class="metric-value yellow">${metrics.reasoning_tok} tok</span></div>
      <div class="metric"><span class="metric-label">Completion</span><span class="metric-value">${metrics.completion_tok} tok</span></div>
      <div class="metric"><span class="metric-label">Total</span><span class="metric-value">${metrics.total_tok} tok</span></div>
      <div class="metric"><span class="metric-label">Total Time</span><span class="metric-value">${metrics.total_time}</span></div>
      <div class="metric"><span class="metric-label">Cache</span><span class="metric-value">${metrics.cached} tok</span></div>
    </div>
    <div class="chat" id="chat">
      ${messages.length === 0 ? '<div class="msg system">Send a message to start chatting. Supports text and image input.</div>' : ''}
      ${messages.map((m,i) => {
        if (m.role === 'user') return `<div class="msg user">${esc(m.content)}</div>`;
        let html = `<div class="msg assistant${streaming && i===messages.length-1 ? ' streaming' : ''}">`;
        if (m.reasoning) html += `<div class="thinking">${esc(m.reasoning)}</div>`;
        html += formatContent(m.content || '') + '</div>';
        return html;
      }).join('')}
    </div>
    <div class="input-area">
      <div class="input-row">
        <textarea id="input" placeholder="Type a message... (Shift+Enter for newline)" rows="1"
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"
          ${streaming ? 'disabled' : ''}></textarea>
        <button onclick="send()" ${streaming ? 'disabled' : ''}>Send</button>
        <button class="clear-btn" onclick="clearChat()">Clear</button>
      </div>
    </div>`;
  const chat = document.getElementById('chat');
  chat.scrollTop = chat.scrollHeight;
  if (!streaming) document.getElementById('input')?.focus();
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

function formatContent(s) {
  // Basic markdown: code blocks, inline code, bold
  return esc(s)
    .replace(/```(\\w*)\\n([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\\*\\*([^*]+)\\*\\*/g, '<b>$1</b>');
}

async function send() {
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text || streaming) return;

  messages.push({ role: 'user', content: text });
  messages.push({ role: 'assistant', content: '', reasoning: '' });
  streaming = true;
  metrics = { prompt_tps: '...', gen_tps: '...', reasoning_tok: '0', completion_tok: '0',
              total_tok: '0', ttft: '...', total_time: '...', cached: '...' };
  render();

  const t0 = performance.now();
  let ttft = null;

  try {
    const apiMessages = messages.filter(m => m.role === 'user').length > 0 ?
      messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })) : [];

    const resp = await fetch(API + '/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma',
        messages: apiMessages,
        max_tokens: 4096,
        temperature: 0.7,
        stream: true
      })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    const last = messages[messages.length - 1];
    let inReasoning = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;

        try {
          const chunk = JSON.parse(data);
          const delta = chunk.choices?.[0]?.delta;
          if (!delta) continue;

          if (ttft === null && (delta.content || delta.reasoning_content)) {
            ttft = performance.now() - t0;
            metrics.ttft = (ttft / 1000).toFixed(2) + 's';
          }

          if (delta.reasoning_content) {
            last.reasoning = (last.reasoning || '') + delta.reasoning_content;
            metrics.reasoning_tok = (last.reasoning.split(/\\s+/).length || 0).toString();
          }
          if (delta.content) {
            last.content = (last.content || '') + delta.content;
            metrics.completion_tok = (last.content.split(/\\s+/).length || 0).toString();
          }

          const usage = chunk.usage;
          if (usage) {
            metrics.total_tok = (usage.completion_tokens || 0).toString();
            metrics.cached = (usage.prompt_tokens_details?.cached_tokens || 0).toString();
            const promptTok = usage.prompt_tokens || 0;
            const completionTok = usage.completion_tokens || 0;
            const elapsed = (performance.now() - t0) / 1000;
            if (ttft && completionTok > 0) {
              const genTime = elapsed - (ttft / 1000);
              metrics.gen_tps = genTime > 0 ? (completionTok / genTime).toFixed(1) : '-';
            }
            if (ttft && promptTok > 0) {
              metrics.prompt_tps = (promptTok / (ttft / 1000)).toFixed(1);
            }
          }

          metrics.total_time = ((performance.now() - t0) / 1000).toFixed(1) + 's';
          render();
        } catch (e) {}
      }
    }

    // Final usage from non-streaming fallback
    const elapsed = (performance.now() - t0) / 1000;
    metrics.total_time = elapsed.toFixed(1) + 's';

  } catch (e) {
    const last = messages[messages.length - 1];
    last.content = 'Error: ' + e.message;
  }

  streaming = false;
  render();
}

function clearChat() { messages = []; metrics = { prompt_tps: '-', gen_tps: '-', reasoning_tok: '-',
  completion_tok: '-', total_tok: '-', ttft: '-', total_time: '-', cached: '-' }; render(); }

render();
</script>
</body>
</html>"""

LOGIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login — Gemma 4 Chat</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root { --bg: #0a0a0f; --surface: #13131a; --surface2: #1a1a24; --border: #2a2a3a;
        --text: #e0e0e8; --accent: #6c8aff; --red: #ff6b6b; }
body { font-family: 'SF Mono', monospace; background: var(--bg); color: var(--text);
       display: flex; justify-content: center; align-items: center; height: 100vh; }
.login-box { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
             padding: 32px; width: 360px; }
h2 { color: var(--accent); margin-bottom: 20px; font-size: 16px; }
input { width: 100%; background: var(--surface2); border: 1px solid var(--border); color: var(--text);
        border-radius: 8px; padding: 10px 14px; font-family: inherit; font-size: 13px;
        outline: none; margin-bottom: 16px; }
input:focus { border-color: var(--accent); }
button { width: 100%; background: var(--accent); color: #fff; border: none; border-radius: 8px;
         padding: 10px; font-family: inherit; font-size: 13px; cursor: pointer; font-weight: 600; }
.error { color: var(--red); font-size: 12px; margin-bottom: 10px; display: ERROR_DISPLAY; }
</style></head>
<body><div class="login-box"><h2>Gemma 4 A4B Chat</h2>
<div class="error">ERROR_MSG</div>
<form method="POST" action="/login">
<input type="password" name="token" placeholder="Enter access token" autofocus>
<button type="submit">Login</button>
</form></div></body></html>"""


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Suppress default logging

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/login":
            self._send_login()
            return

        if not check_auth(self.headers.get("Cookie", "")):
            self.send_response(302)
            self.send_header("Location", "/login")
            self.end_headers()
            return

        if path == "/" or path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/login":
            params = parse_qs(body.decode())
            token = params.get("token", [""])[0]
            if secrets.compare_digest(token, AUTH_TOKEN):
                sid = create_session()
                self.send_response(302)
                self.send_header("Set-Cookie", f"session={sid}; HttpOnly; Path=/; SameSite=Strict; Max-Age=86400")
                self.send_header("Location", "/")
                self.end_headers()
            else:
                self._send_login(error="Invalid token")
            return

        if not check_auth(self.headers.get("Cookie", "")):
            self.send_response(401)
            self.end_headers()
            return

        # Proxy API requests to llama-server
        if path.startswith("/v1/"):
            self._proxy_api(path, body)
            return

        self.send_response(404)
        self.end_headers()

    def _send_login(self, error=None):
        page = LOGIN_PAGE.replace("ERROR_MSG", error or "").replace(
            "ERROR_DISPLAY", "block" if error else "none")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def _proxy_api(self, path, body):
        """Proxy request to llama-server, streaming the response back."""
        url = f"{LLAMA_API}{path}"
        try:
            req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            resp = urlopen(req, timeout=300)

            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() in ("content-type", "transfer-encoding"):
                    self.send_header(key, val)
            self.end_headers()

            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Backend unavailable: {e}"}).encode())


def main():
    server = HTTPServer(("0.0.0.0", PORT), ChatHandler)
    print(f"Chat server running on http://0.0.0.0:{PORT}")
    print(f"Backend: {LLAMA_API}")
    print(f"Auth token: {AUTH_TOKEN[:8]}...{AUTH_TOKEN[-4:]}")
    print(f"\nOpen in browser and enter the auth token to start chatting.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
