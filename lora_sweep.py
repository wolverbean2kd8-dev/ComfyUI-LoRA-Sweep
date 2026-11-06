#!/usr/bin/env python3
"""
LoRA Sweep: run one prompt through every LoRA epoch in ComfyUI and compare the videos side by side.

Usage:
    python lora_sweep.py                      # ComfyUI at http://127.0.0.1:8189, UI at http://127.0.0.1:8899
    python lora_sweep.py --comfy http://127.0.0.1:8188 --port 8899

Needs only Python 3.8+ (standard library). Start ComfyUI first.
"""
import argparse, copy, json, mimetypes, os, re, sys, threading, time, uuid
import urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY = "http://127.0.0.1:8189"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
WF_PATH = os.path.join(ROOT, "workflow.json")
CLIENT_ID = uuid.uuid4().hex
LOCK = threading.Lock()
RUNS = {}
WF = {"data": None}
TEXT_KEYS = ("text", "prompt", "positive", "positive_prompt", "text_positive", "caption")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp", ".avi")


# ---------------------------------------------------------------- ComfyUI client
def comfy_get(path):
    with urllib.request.urlopen(COMFY + path, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def comfy_post(path, obj):
    req = urllib.request.Request(COMFY + path, data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")
        return json.loads(body) if body else {}


def explain_http_error(e):
    try:
        body = json.loads(e.read().decode("utf-8"))
    except Exception:
        return "ComfyUI returned HTTP %s" % e.code
    err = body.get("error", {})
    msg = err.get("message") or "ComfyUI rejected the workflow"
    det = err.get("details")
    nodes = body.get("node_errors") or {}
    extra = []
    for nid, ne in nodes.items():
        for item in ne.get("errors", []):
            extra.append("node %s: %s" % (nid, item.get("message", "")))
    return " | ".join([msg] + ([det] if det else []) + extra[:3])


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def combo_options(spec):
    if not isinstance(spec, (list, tuple)) or not spec:
        return []
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return []


def list_loras():
    classes = ["LoraLoaderModelOnly", "LoraLoader"]
    wf = WF["data"] or {}
    for node in wf.values():
        if "lora_name" in node.get("inputs", {}) and node["class_type"] not in classes:
            classes.append(node["class_type"])
    names = set()
    for cls in classes:
        try:
            info = comfy_get("/object_info/" + urllib.parse.quote(cls))[cls]["input"]
            for group in ("required", "optional"):
                if "lora_name" in info.get(group, {}):
                    names.update(combo_options(info[group]["lora_name"]))
        except Exception:
            pass
    return sorted(names, key=natural)


# ---------------------------------------------------------------- workflow analysis
def validate_workflow(wf):
    if isinstance(wf, dict) and "nodes" in wf and "links" in wf:
        raise ValueError("That is the regular workflow file. In ComfyUI use Workflow > Export (API) "
                         "and load that file instead.")
    if not isinstance(wf, dict) or not wf or not all(isinstance(v, dict) and "class_type" in v for v in wf.values()):
        raise ValueError("This does not look like a ComfyUI API-format workflow.")


def title_of(node):
    return (node.get("_meta") or {}).get("title") or node["class_type"]


def analyze(wf):
    loras, texts, seeds, outputs = [], [], [], []
    for nid, node in wf.items():
        ins = node.get("inputs", {})
        t = title_of(node)
        if isinstance(ins.get("lora_name"), str):
            loras.append({"id": nid, "class": node["class_type"], "title": t, "current": ins["lora_name"]})
        for key, val in ins.items():
            if not isinstance(val, str) or key == "lora_name":
                continue
            low = key.lower()
            cls = node["class_type"].lower()
            strict = key in TEXT_KEYS or any(w in low for w in ("prompt", "text", "caption", "string")) \
                or any(w in cls for w in ("string", "text", "prompt"))
            skip = any(w in low for w in ("name", "prefix", "format", "codec", "sampler", "scheduler", "device",
                                          "dtype", "mode", "resolution", "image", "video", "path", "type", "model"))
            if not strict and (skip or len(val) < 25):
                continue
            neg = "neg" in t.lower()
            texts.append({"id": nid, "key": key, "class": node["class_type"], "title": t,
                          "preview": val[:90], "negative": neg, "strict": strict})
        for key in ("seed", "noise_seed"):
            if isinstance(ins.get(key), (int, float)):
                seeds.append({"id": nid, "key": key})
        if any(w in node["class_type"] for w in ("Save", "VideoCombine", "Preview")):
            outputs.append({"id": nid, "class": node["class_type"], "title": t})
    texts.sort(key=lambda x: (not x["strict"], x["negative"]))
    return {"loras": loras, "texts": texts, "seeds": seeds, "outputs": outputs}


def load_workflow_file():
    try:
        with open(WF_PATH, "r", encoding="utf-8") as f:
            wf = json.load(f)
        validate_workflow(wf)
        WF["data"] = wf
    except Exception:
        WF["data"] = None


# ---------------------------------------------------------------- runs
def run_dir(rid):
    return os.path.join(ROOT, rid)


def save_run(run):
    with LOCK:
        os.makedirs(run_dir(run["id"]), exist_ok=True)
        with open(os.path.join(run_dir(run["id"]), "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1)


def build_prompt(run, job):
    wf = copy.deepcopy(WF["data"])
    ln = wf[run["lora_node"]]["inputs"]
    ln["lora_name"] = job["lora"]
    strength = 0.0 if job["baseline"] else run.get("strength")
    if strength is not None:
        for k in ("strength_model", "strength_clip", "strength"):
            if isinstance(ln.get(k), (int, float)):
                ln[k] = strength
    pn = run["prompt_node"]
    wf[pn["id"]]["inputs"][pn["key"]] = run["prompt"]
    if run.get("seed") is not None:
        for s in analyze(wf)["seeds"]:
            wf[s["id"]]["inputs"][s["key"]] = run["seed"]
    return wf


def collect_files(outputs):
    found = []
    for out in outputs.values():
        if not isinstance(out, dict):
            continue
        for val in out.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "filename" in item:
                        found.append(item)
    real = [f for f in found if f.get("type", "output") == "output"] or found
    vids = [f for f in real if f["filename"].lower().endswith(VIDEO_EXT)]
    return vids or real


def download(item, dest_dir, prefix):
    q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    ext = os.path.splitext(item["filename"])[1] or ".mp4"
    name = prefix + ext
    with urllib.request.urlopen(COMFY + "/view?" + q, timeout=120) as r, \
            open(os.path.join(dest_dir, name), "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    return name


def error_text(entry):
    for kind, data in entry.get("status", {}).get("messages", []):
        if kind == "execution_error":
            return "%s: %s" % (data.get("node_type", "node"), data.get("exception_message", "failed")[:300])
    return "ComfyUI reported an error"


def run_worker(run):
    os.makedirs(run_dir(run["id"]), exist_ok=True)
    for i, job in enumerate(run["jobs"]):
        if run["cancel"]:
            break
        try:
            res = comfy_post("/prompt", {"prompt": build_prompt(run, job), "client_id": CLIENT_ID})
            job["pid"] = res["prompt_id"]
            job["status"] = "queued"
        except urllib.error.HTTPError as e:
            job["status"], job["error"] = "error", explain_http_error(e)
        except Exception as e:
            job["status"], job["error"] = "error", "Could not reach ComfyUI: %s" % e
        save_run(run)
    while True:
        active = [j for j in run["jobs"] if j["status"] in ("queued", "running")]
        if not active:
            break
        if run["cancel"]:
            for j in active:
                try:
                    comfy_post("/queue", {"delete": [j["pid"]]})
                except Exception:
                    pass
                j["status"] = "cancelled"
            try:
                comfy_post("/interrupt", {})
            except Exception:
                pass
            break
        try:
            running = {item[1] for item in comfy_get("/queue").get("queue_running", [])}
        except Exception:
            running = set()
        for j in active:
            try:
                hist = comfy_get("/history/" + j["pid"])
            except Exception:
                continue
            entry = hist.get(j["pid"])
            if not entry:
                j["status"] = "running" if j["pid"] in running else "queued"
                continue
            if entry.get("status", {}).get("status_str") == "error":
                j["status"], j["error"] = "error", error_text(entry)
                continue
            files = collect_files(entry.get("outputs", {}))
            if not files:
                if entry.get("status", {}).get("completed"):
                    j["status"], j["error"] = "error", "Finished but no video or image output was found. Check that your workflow has a save node."
                continue
            try:
                idx = run["jobs"].index(j)
                name = download(files[0], run_dir(run["id"]), "%02d_%s" % (idx, re.sub(r"[^\w.-]+", "_", j["label"])))
                j["file"] = "/files/%s/%s" % (run["id"], name)
                j["status"] = "done"
            except Exception as e:
                j["status"], j["error"] = "error", "Download failed: %s" % e
        save_run(run)
        time.sleep(2)
    run["finished"] = True
    save_run(run)


def start_run(body):
    if not WF["data"]:
        raise ValueError("Load a workflow first.")
    loras = body.get("loras") or []
    if not loras:
        raise ValueError("Select at least one LoRA file.")
    wf = WF["data"]
    ln = body.get("lora_node")
    pn = body.get("prompt_node") or {}
    if ln not in wf or pn.get("id") not in wf:
        raise ValueError("Pick the LoRA node and the prompt node.")
    rid = time.strftime("%Y%m%d-%H%M%S")
    jobs = []
    if body.get("baseline"):
        jobs.append({"label": "baseline (no LoRA)", "lora": loras[0], "baseline": True, "status": "pending"})
    for name in loras:
        jobs.append({"label": os.path.splitext(os.path.basename(name))[0], "lora": name, "baseline": False, "status": "pending"})
    seed = body.get("seed")
    strength = body.get("strength")
    run = {"id": rid, "created": time.time(), "prompt": body.get("prompt", ""), "lora_node": ln, "prompt_node": pn,
           "seed": int(seed) if seed not in (None, "") else None,
           "strength": float(strength) if strength not in (None, "") else None,
           "jobs": jobs, "cancel": False, "finished": False}
    RUNS[rid] = run
    save_run(run)
    threading.Thread(target=run_worker, args=(run,), daemon=True).start()
    return run


def load_old_runs():
    if not os.path.isdir(ROOT):
        return
    for d in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, d, "run.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    run = json.load(f)
                for j in run["jobs"]:
                    if j["status"] in ("queued", "running", "pending"):
                        j["status"], j["error"] = "error", "Interrupted when the sweep app was closed."
                run["finished"] = True
                RUNS[run["id"]] = run
            except Exception:
                pass


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def serve_file(self, path):
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a == "":
                start = max(0, size - int(b))
            else:
                start = int(a)
                end = int(b) if b else size - 1
            end = min(end, size - 1)
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/":
                data = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/status":
                try:
                    comfy_get("/system_stats")
                    ok = True
                except Exception:
                    ok = False
                self.send_json({"comfy": ok, "url": COMFY, "workflow": WF["data"] is not None})
            elif path == "/api/workflow":
                if WF["data"] is None:
                    self.send_json({"error": "none"}, 404)
                else:
                    self.send_json(analyze(WF["data"]))
            elif path == "/api/loras":
                self.send_json({"loras": list_loras()})
            elif path == "/api/runs":
                runs = sorted(RUNS.values(), key=lambda r: r["created"], reverse=True)
                self.send_json([{"id": r["id"], "prompt": r["prompt"][:80], "count": len(r["jobs"])} for r in runs])
            elif path.startswith("/api/run/"):
                run = RUNS.get(path.split("/")[3])
                self.send_json(run, 200) if run else self.send_json({"error": "no such run"}, 404)
            elif path.startswith("/files/"):
                full = os.path.normpath(os.path.join(ROOT, urllib.parse.unquote(path[len("/files/"):])))
                if full.startswith(ROOT + os.sep) and os.path.isfile(full):
                    self.serve_file(full)
                else:
                    self.send_json({"error": "not found"}, 404)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/workflow":
                wf = self.body()
                validate_workflow(wf)
                os.makedirs(ROOT, exist_ok=True)
                with open(WF_PATH, "w", encoding="utf-8") as f:
                    json.dump(wf, f)
                WF["data"] = wf
                self.send_json(analyze(wf))
            elif path == "/api/run":
                self.send_json(start_run(self.body()))
            elif path.startswith("/api/run/") and path.endswith("/cancel"):
                run = RUNS.get(path.split("/")[3])
                if run:
                    run["cancel"] = True
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)


# ---------------------------------------------------------------- UI
INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoRA Sweep</title>
<style>
:root{--bg:#eef0f3;--panel:#fff;--panel2:#e6e9ee;--line:#d3d8e0;--fg:#171a20;--muted:#5d6675;--accent:#4351d9;--accent-fg:#fff;--bad:#c0392b;--good:#1f8a4c;--warn:#b86e00;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--panel:#171a20;--panel2:#20242c;--line:#2c323d;--fg:#e7e9ee;--muted:#8c95a4;--accent:#8c97ff;--accent-fg:#10122a;--bad:#ff7a6b;--good:#59d48b;--warn:#f0b44c;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;padding:20px max(16px,env(safe-area-inset-left))}
.page{max-width:1500px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
h1{font-size:26px;margin:0}h2{font-size:15px;margin:0 0 10px}
.sub{color:var(--muted);margin:4px 0 0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end}
.field{display:flex;flex-direction:column;gap:4px;min-width:0}
.field label,.lbl{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
input,select,textarea,button{font:inherit;color:inherit}
input,select,textarea{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:7px 9px;min-width:0}
textarea{width:100%;min-height:84px;resize:vertical}
button,.btn{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:7px 12px;cursor:pointer;font-weight:500}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
button:disabled{opacity:.5;cursor:not-allowed}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;border:1px solid var(--line)}
.ok{color:var(--good)}.bad{color:var(--bad)}
.loras{max-height:220px;overflow:auto;border:1px solid var(--line);border-radius:6px;padding:6px;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:2px 12px}
.loras label{display:flex;gap:6px;align-items:center;font-size:13px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--cw,320px),1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:6px;min-width:0}
.card.best{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.card h3{margin:0;font-size:16px}.card small{color:var(--muted);overflow-wrap:anywhere}
.frame{aspect-ratio:var(--ar,16/9);background:#0b0c0f;border-radius:5px;display:flex;align-items:center;justify-content:center;color:#9aa3b2;font-size:13px;text-align:center;padding:8px;overflow:hidden}
.frame video,.frame img{width:100%;height:100%;object-fit:contain}
.rate{display:flex;gap:3px}.rate button{flex:1;padding:3px 0;font-size:12px}.rate button.on{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
.rank{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.chip{border:1px solid var(--line);border-radius:999px;padding:3px 12px;font-size:12px;background:var(--panel)}
.hint{color:var(--muted);font-size:13px}
.err{color:var(--bad);font-size:12px;overflow-wrap:anywhere}
</style></head><body><div class="page">
<div><h1>LoRA Sweep</h1><p class="sub">Run one prompt through every LoRA epoch in ComfyUI, then compare the videos side by side. <span id="conn" class="pill">checking ComfyUI</span></p></div>

<div class="panel" id="setup">
 <h2>1. Workflow</h2>
 <p class="hint">In ComfyUI open your working MiniMax H3 workflow and choose Workflow > Export (API). Load that file here once; the app remembers it.</p>
 <div class="row"><input type="file" id="wfFile" accept=".json"><span id="wfMsg" class="hint"></span></div>
 <div class="row" id="pickers" hidden style="margin-top:12px">
  <div class="field"><label for="loraNode">LoRA loader node</label><select id="loraNode"></select></div>
  <div class="field" style="flex:1 1 320px"><label for="textNode">Prompt field</label><select id="textNode"></select></div>
 </div>
</div>

<div class="panel" id="gen" hidden>
 <h2>2. Prompt</h2>
 <textarea id="prompt" placeholder="One prompt that gets used for every epoch"></textarea>
 <div class="row" style="margin-top:10px">
  <div class="field"><label for="seed">Seed (same for every epoch)</label><input id="seed" type="number" value="42" style="width:140px"></div>
  <div class="field"><label for="strength">LoRA strength</label><input id="strength" type="number" step="0.05" placeholder="keep workflow value" style="width:170px"></div>
  <label class="row" style="align-items:center;gap:6px"><input type="checkbox" id="baseline" checked> Add a no-LoRA baseline</label>
 </div>
 <h2 style="margin-top:16px">3. Epochs to test</h2>
 <div class="row" style="margin-bottom:8px">
  <div class="field"><label for="filter">Filter file names</label><input id="filter" placeholder="e.g. my_lora"></div>
  <button id="allBtn" type="button">Select all shown</button><button id="noneBtn" type="button">Clear</button>
  <span id="selCount" class="hint"></span>
 </div>
 <div class="loras" id="loras"></div>
 <div class="row" style="margin-top:12px"><button class="primary" id="goBtn" type="button">Generate all</button><button id="cancelBtn" type="button" hidden>Cancel</button><span id="goMsg" class="hint"></span></div>
</div>

<div class="panel" id="results" hidden>
 <div class="row" style="margin-bottom:10px">
  <div class="field"><label for="runSel">Run</label><select id="runSel"></select></div>
  <button id="playBtn" type="button">Play all</button><button id="restartBtn" type="button">Restart</button>
  <div class="field"><label for="speed">Speed</label><select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select></div>
  <div class="field"><label for="size">Size</label><input type="range" id="size" min="200" max="700" value="320"></div>
  <span id="runMsg" class="hint"></span>
 </div>
 <div class="rank" id="rank" style="margin-bottom:10px"></div>
 <div class="grid" id="cards"></div>
</div>
</div>
<script>
var $=function(i){return document.getElementById(i)};
var A=null, allLoras=[], chosen=new Set(), runId=null, timer=null, ratings={};
function api(p,o){return fetch(p,o).then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.error||('HTTP '+r.status));return j;});});}
function post(p,b){return api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});}
function label(n){var m=n.match(/(\d+)(?!.*\d)/);return m?('Epoch '+parseInt(m[1],10)):n;}
function nat(a,b){return a.localeCompare(b,undefined,{numeric:true});}

function status(){api('/api/status').then(function(s){var c=$('conn');c.textContent=s.comfy?('ComfyUI connected at '+s.url):('Cannot reach ComfyUI at '+s.url+'. Start it, then reload.');c.className='pill '+(s.comfy?'ok':'bad');});}
function showAnalysis(a){
 A=a; $('pickers').hidden=false; $('gen').hidden=false;
 var ln=$('loraNode'); ln.innerHTML='';
 a.loras.forEach(function(l){var o=document.createElement('option');o.value=l.id;o.textContent='#'+l.id+' '+l.title+' ('+l.class+') now: '+l.current;ln.appendChild(o);});
 var tn=$('textNode'); tn.innerHTML=''; var first=-1;
 a.texts.forEach(function(t,i){var o=document.createElement('option');o.value=i;o.textContent='#'+t.id+' '+t.title+' ['+t.key+']'+(t.negative?' (negative)':'')+': '+t.preview;tn.appendChild(o);if(first<0&&!t.negative)first=i;});
 if(first>=0)tn.value=first;
 if(!a.loras.length)$('wfMsg').textContent='No LoRA loader found in this workflow. It needs a node with a lora_name field.';
 else $('wfMsg').textContent='Workflow loaded.';
 if(a.texts.length&&first>=0&&!$('prompt').value)$('prompt').value=a.texts[first].preview.length<90?a.texts[first].preview:'';
 api('/api/loras').then(function(r){allLoras=r.loras;drawLoras();});
}
$('wfFile').onchange=function(e){var f=e.target.files[0];if(!f)return;var fr=new FileReader();
 fr.onload=function(){try{post('/api/workflow',JSON.parse(fr.result)).then(showAnalysis).catch(function(x){$('wfMsg').textContent=x.message;});}catch(x){$('wfMsg').textContent='That file is not valid JSON.';}};fr.readAsText(f);};
function shown(){var f=$('filter').value.toLowerCase();return allLoras.filter(function(n){return n.toLowerCase().indexOf(f)>=0;});}
function drawLoras(){var box=$('loras');box.innerHTML='';
 if(!allLoras.length){box.textContent='No LoRA files reported by ComfyUI. Put them in ComfyUI/models/loras and refresh ComfyUI.';}
 shown().forEach(function(n){var l=document.createElement('label');var c=document.createElement('input');c.type='checkbox';c.checked=chosen.has(n);
  c.onchange=function(){c.checked?chosen.add(n):chosen.delete(n);count();};l.appendChild(c);l.appendChild(document.createTextNode(n));box.appendChild(l);});count();}
function count(){$('selCount').textContent=chosen.size+' selected';}
$('filter').oninput=drawLoras;
$('allBtn').onclick=function(){shown().forEach(function(n){chosen.add(n);});drawLoras();};
$('noneBtn').onclick=function(){chosen.clear();drawLoras();};

$('goBtn').onclick=function(){
 var t=A.texts[+$('textNode').value];
 if(!t){$('goMsg').textContent='Pick a prompt field.';return;}
 if(!$('prompt').value.trim()){$('goMsg').textContent='Write a prompt first.';return;}
 var loras=Array.from(chosen).sort(nat);
 $('goMsg').textContent='Starting...';
 post('/api/run',{lora_node:$('loraNode').value,prompt_node:{id:t.id,key:t.key},prompt:$('prompt').value,seed:$('seed').value,strength:$('strength').value,loras:loras,baseline:$('baseline').checked})
  .then(function(run){$('goMsg').textContent='';openRun(run.id);}).catch(function(x){$('goMsg').textContent=x.message;});
};
$('cancelBtn').onclick=function(){if(runId)post('/api/run/'+runId+'/cancel');};

function refreshRuns(sel){api('/api/runs').then(function(rs){var s=$('runSel');s.innerHTML='';rs.forEach(function(r){var o=document.createElement('option');o.value=r.id;o.textContent=r.id+' · '+r.count+' videos · '+r.prompt;s.appendChild(o);});if(sel)s.value=sel;$('results').hidden=!rs.length;});}
$('runSel').onchange=function(){openRun(this.value);};
function openRun(id){runId=id;try{ratings=JSON.parse(localStorage.getItem('sweep:'+id)||'{}');}catch(e){ratings={};}refreshRuns(id);poll();}
function poll(){clearTimeout(timer);api('/api/run/'+runId).then(function(run){drawRun(run);if(!run.finished)timer=setTimeout(poll,2000);}).catch(function(x){$('runMsg').textContent=x.message;});}

var firstAR=false;
function drawRun(run){
 var done=run.jobs.filter(function(j){return j.status==='done';}).length;
 $('runMsg').textContent=done+' of '+run.jobs.length+' finished'+(run.finished?'':'. Generating...');
 $('cancelBtn').hidden=run.finished;
 var cards=$('cards'), have={};
 Array.prototype.forEach.call(cards.children,function(c){have[c.dataset.label]=c;});
 run.jobs.forEach(function(j){
  var c=have[j.label];
  if(!c){c=document.createElement('div');c.className='card';c.dataset.label=j.label;
   c.innerHTML='<h3></h3><small></small><div class="frame"></div><div class="rate"></div><div class="err"></div>';
   var rt=c.querySelector('.rate');for(var v=1;v<=5;v++){(function(v){var b=document.createElement('button');b.textContent=v;b.onclick=function(){if(ratings[j.label]===v)delete ratings[j.label];else ratings[j.label]=v;try{localStorage.setItem('sweep:'+runId,JSON.stringify(ratings));}catch(e){}paint();};rt.appendChild(b);})(v);}
   cards.appendChild(c);}
  c.querySelector('h3').textContent=j.baseline?'Baseline':label(j.label);
  c.querySelector('small').textContent=j.baseline?'No LoRA effect (strength 0)':j.lora;
  c.querySelector('.err').textContent=j.error||'';
  var fr=c.querySelector('.frame');
  if(j.status==='done'&&j.file){
   if(fr.dataset.src!==j.file){fr.dataset.src=j.file;fr.innerHTML='';var isImg=/\.(gif|webp|png|jpe?g)$/i.test(j.file);var m=document.createElement(isImg?'img':'video');m.src=j.file;
    if(!isImg){m.muted=true;m.loop=true;m.playsInline=true;m.controls=true;m.preload='auto';m.addEventListener('loadedmetadata',function(){m.playbackRate=+$('speed').value;if(!firstAR&&m.videoWidth){firstAR=true;document.documentElement.style.setProperty('--ar',m.videoWidth+'/'+m.videoHeight);}});}
    fr.appendChild(m);}
  }else{fr.dataset.src='';fr.textContent=j.status==='error'?'Failed':(j.status==='running'?'Generating...':(j.status==='cancelled'?'Cancelled':'Waiting in queue'));}
 });
 paint();
}
function paint(){
 var cards=$('cards').children;
 Array.prototype.forEach.call(cards,function(c){var v=ratings[c.dataset.label]||0;Array.prototype.forEach.call(c.querySelectorAll('.rate button'),function(b,i){b.classList.toggle('on',i<v);});});
 var rows=Object.keys(ratings).map(function(k){return {k:k,v:ratings[k]};}).sort(function(a,b){return b.v-a.v;});
 var rk=$('rank');rk.innerHTML='';
 if(rows.length){var l=document.createElement('span');l.className='lbl';l.textContent='Your ratings';rk.appendChild(l);}
 rows.slice(0,8).forEach(function(r){var s=document.createElement('span');s.className='chip';s.textContent=(/baseline/.test(r.k)?'Baseline':label(r.k))+'  '+r.v+'/5';rk.appendChild(s);});
 var top=rows.length?rows[0].v:0;
 Array.prototype.forEach.call(cards,function(c){c.classList.toggle('best',top>0&&ratings[c.dataset.label]===top);});
}
function vids(){return Array.prototype.slice.call(document.querySelectorAll('#cards video'));}
$('playBtn').onclick=function(){var vs=vids();var playing=vs.some(function(v){return !v.paused;});
 if(playing){vs.forEach(function(v){v.pause();});this.textContent='Play all';}
 else{vs.forEach(function(v){v.currentTime=0;v.playbackRate=+$('speed').value;var p=v.play();if(p&&p.catch)p.catch(function(){});});this.textContent='Pause all';}};
$('restartBtn').onclick=function(){vids().forEach(function(v){v.currentTime=0;var p=v.play();if(p&&p.catch)p.catch(function(){});});$('playBtn').textContent='Pause all';};
$('speed').onchange=function(){vids().forEach(function(v){v.playbackRate=+$('speed').value;});};
$('size').oninput=function(){document.documentElement.style.setProperty('--cw',this.value+'px');};

status();
api('/api/workflow').then(showAnalysis).catch(function(){});
refreshRuns();
api('/api/runs').then(function(rs){if(rs.length)openRun(rs[0].id);});
</script></body></html>
"""


def main():
    global COMFY, ROOT, WF_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=COMFY, help="ComfyUI address")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--out", default=ROOT, help="folder for finished videos")
    args = ap.parse_args()
    COMFY = args.comfy.rstrip("/")
    ROOT = os.path.abspath(args.out)
    WF_PATH = os.path.join(ROOT, "workflow.json")
    os.makedirs(ROOT, exist_ok=True)
    load_workflow_file()
    load_old_runs()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    print("LoRA Sweep running. Open http://127.0.0.1:%d" % args.port)
    print("ComfyUI: %s   Output: %s" % (COMFY, ROOT))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
