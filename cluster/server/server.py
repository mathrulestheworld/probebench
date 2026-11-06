#!/usr/bin/env python3
"""Probe Bench server.

Runs inside an HTCondor GPU job on the ISyE cluster (started by run_session.sh)
and serves the web interface in index.html. Every request except /health needs
the session token that labctl.sh wrote into the session folder, so other people
on the cluster can't use your session.

One GPU operation runs at a time. Long operations (downloading, loading,
generating, analyzing) run as background tasks that the page polls.
The session ends itself after --idle-minutes without activity, to free the GPU.
"""
from __future__ import annotations

import argparse
import gc
import hmac
import json
import math
import os
import re
import secrets
import signal
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import download as dl  # noqa: E402

GiB = 2 ** 30
MAX_ANALYZE_TOKENS = 256
LENS_TOP_K = 5

SUGGESTED = [
    {"id": "google/gemma-4-12B-it", "size_gb": 22.3, "note": "Gemma 4 12B, instruction-tuned"},
    {"id": "google/gemma-4-12B", "size_gb": 22.3, "note": "Gemma 4 12B, base model"},
    {"id": "Qwen/Qwen3.5-9B", "size_gb": 18.0, "note": "Qwen 3.5 9B"},
    {"id": "allenai/Olmo-3-7B-Instruct", "size_gb": 13.6, "note": "Olmo 3 7B, fully open training"},
    {"id": "Qwen/Qwen3.8-27B", "size_gb": 51.7, "note": "Qwen 3.8 27B, needs an A100"},
]


class UserError(Exception):
    """An error with a message written for the person using the page."""


class Cancelled(Exception):
    pass


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def clean(x):
    """Make floats JSON-safe (no NaN/inf)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- model helpers

def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def find_structure(model):
    """Locate the decoder layers, the final norm and the output head.

    Works for the usual Hugging Face layouts: model.layers, model.language_model.layers,
    transformer.h, ... Vision or audio towers are skipped.
    """
    cfg = text_config(model)
    n_cfg = getattr(cfg, "num_hidden_layers", None)
    skip = ("vision", "visual", "audio", "image", "mtp", "draft")
    best = None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) == 0:
            continue
        if any(s in name.lower() for s in skip):
            continue
        score = 0
        if n_cfg is not None and len(mod) == n_cfg:
            score += 10
        if name.endswith(("layers", ".h", "blocks")):
            score += 5
        if "language_model" in name or "text" in name:
            score += 1
        cand = (score, len(mod), name, mod)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        raise UserError("Couldn't find this model's layers, so activations aren't available for it.")
    _, _, layers_path, layers = best
    parent = model.get_submodule(layers_path.rsplit(".", 1)[0]) if "." in layers_path else model
    norm, norm_path = None, None
    for attr in ("norm", "final_layernorm", "ln_f", "final_norm", "norm_f"):
        m = getattr(parent, attr, None)
        if isinstance(m, torch.nn.Module):
            norm = m
            norm_path = (layers_path.rsplit(".", 1)[0] + "." if "." in layers_path else "") + attr
            break
    head = model.get_output_embeddings()
    softcap = getattr(cfg, "final_logit_softcapping", None) or getattr(model.config, "final_logit_softcapping", None)
    return {"layers": layers, "layers_path": layers_path, "n_layers": len(layers),
            "norm": norm, "norm_path": norm_path, "head": head, "softcap": softcap}


def cget(cfg, key, default=None):
    """getattr that never raises: some configs (Gemma 4) refuse per-layer attributes at the top level."""
    try:
        return getattr(cfg, key)
    except Exception:  # noqa: BLE001
        return default


def ranges(nums):
    """[0,1,2,5,6,9] -> '0–2, 5–6, 9'"""
    out, start, prev = [], None, None
    for n in list(nums) + [None]:
        if start is None:
            start = prev = n
        elif n is not None and n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = n
    return ", ".join(out)


def tensor_bytes(t):
    return t.numel() * t.element_size() if torch.is_tensor(t) else 0


def nested_bytes(x):
    if torch.is_tensor(x):
        return tensor_bytes(x)
    if isinstance(x, dict):
        return sum(nested_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(nested_bytes(v) for v in x)
    return 0


def bf16_bytes(folder):
    """Bytes the model will take on the GPU in bf16, read from the safetensors headers (no weights loaded)."""
    total = 0
    for f in Path(folder).glob("*.safetensors"):
        try:
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n))
        except (OSError, ValueError):
            continue
        for name, meta in header.items():
            if name != "__metadata__":
                total += 2 * math.prod(meta.get("shape") or [1])
    return total


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80] or "model"


# --------------------------------------------------------------------------- the lab

class Lab:
    def __init__(self, args, token):
        self.args = args
        self.token = token
        self.session_dir = Path(args.session_dir)
        self.session_id = self.session_dir.name
        self.lab_dir = Path(args.lab_dir)
        self.exports_dir = self.lab_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.device = args.device
        self.started = time.time()
        self.last_activity = time.time()
        self.idle_seconds = max(1.0, args.idle_minutes * 60.0)
        self.gpu_lock = threading.Lock()
        self.busy = None
        self.tasks = {}
        self.tasks_lock = threading.Lock()
        self.model = None
        self.tok = None
        self.model_id = None
        self.structure = None
        self.info = None
        self.weights_index = None
        self.httpd = None
        self.ended = None
        self.tokstr_cache = {}
        self.special_ids = set()
        self.folder = None
        self.weights_bytes = 0
        self.last = None          # the last analysis: residual stream kept on the CPU for channel views
        self.last_gen = None      # memory facts from the last generation
        self.readme = None

    # ---- activity and shutdown
    def touch(self):
        self.last_activity = time.time()

    def running_tasks(self):
        with self.tasks_lock:
            return [t for t in self.tasks.values() if t["status"] == "running"]

    def idle_left(self):
        if self.running_tasks():
            return self.idle_seconds
        return max(0.0, self.idle_seconds - (time.time() - self.last_activity))

    def watchdog(self):
        while self.ended is None:
            time.sleep(5)
            if self.idle_left() <= 0:
                log("No activity; ending the session to free the GPU.")
                self.shutdown("idle")

    def shutdown(self, reason):
        if self.ended:
            return
        self.ended = reason
        try:
            write_atomic(self.session_dir / "ended.json", json.dumps({"reason": reason, "at": time.time()}))
        except OSError:
            pass
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()

    # ---- tasks
    def start_task(self, kind, fn, *args, uses_gpu=True):
        tid = secrets.token_hex(6)
        task = {"id": tid, "kind": kind, "status": "running", "progress": "Starting", "partial": "",
                "result": None, "error": None, "started": time.time(), "finished": None,
                "cancel": False, "fraction": None, "code": None}
        with self.tasks_lock:
            self.tasks[tid] = task
            if len(self.tasks) > 60:  # keep the newest
                for old in sorted(self.tasks.values(), key=lambda t: t["started"])[:-60]:
                    if old["status"] != "running":
                        self.tasks.pop(old["id"], None)

        def run():
            try:
                if uses_gpu:
                    with self.gpu_turn(task):
                        task["result"] = fn(task, *args)
                else:
                    task["result"] = fn(task, *args)
                task["status"] = "done"
            except Cancelled:
                task["status"] = "cancelled"
            except UserError as e:
                task["status"], task["error"] = "error", str(e)
            except torch.cuda.OutOfMemoryError:
                task["status"] = "error"
                task["error"] = ("The GPU ran out of memory. Use a shorter prompt, unload the model, "
                                 "or start a session with the Large or A100 size.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except dl.DownloadError as e:
                task["status"], task["error"] = "error", str(e)
            except Exception as e:  # noqa: BLE001
                log(traceback.format_exc())
                task["status"] = "error"
                task["error"] = f"{type(e).__name__}: {e}"
            finally:
                task["finished"] = time.time()
                task["progress"] = ""
                self.touch()
                took = task["finished"] - task["started"]
                log(f"{kind}: {task['status']} in {took:.1f} s" + (f" ({task['error']})" if task["error"] else ""))

        threading.Thread(target=run, daemon=True).start()
        return tid

    @contextmanager
    def gpu_turn(self, task):
        if not self.gpu_lock.acquire(blocking=False):
            task["progress"] = "Waiting for the current GPU task to finish"
            self.gpu_lock.acquire()
        self.busy = task["kind"]
        try:
            yield
        finally:
            self.busy = None
            self.gpu_lock.release()

    def task_view(self, tid):
        t = self.tasks.get(tid)
        if t is None:
            raise UserError("That task is no longer available.")
        return {k: t[k] for k in ("id", "kind", "status", "progress", "partial", "result", "error", "fraction", "code")}

    # ---- status
    def gpu_info(self):
        if self.device == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"name": torch.cuda.get_device_name(0), "total_gb": total / GiB,
                    "used_gb": (total - free) / GiB, "ours_gb": torch.cuda.memory_allocated() / GiB}
        return {"name": "CPU (test mode)", "total_gb": None, "used_gb": None, "ours_gb": None}

    def status(self):
        import transformers
        hf_home = dl.hf_home()
        return {
            "session": {"id": self.session_id, "node": socket.getfqdn(), "started": self.started,
                        "idle_minutes": self.args.idle_minutes, "idle_left_s": self.idle_left(),
                        "gpu_request_mb": self.args.gpu_request_mb, "mem_request_gb": self.args.mem_request_gb},
            "gpu": self.gpu_info(),
            "model": self.info,
            "busy": self.busy,
            "hf_home": str(hf_home),
            "hf_token_saved": (hf_home / "token").exists() or bool(os.environ.get("HF_TOKEN")),
            "exports_dir": str(self.exports_dir),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "python": sys.version.split()[0]},
        }

    # ---- models on disk
    def list_models(self):
        hub = dl.hf_home() / "hub"
        gpu = self.gpu_info()
        out = []
        for d in sorted(hub.glob("models--*")) if hub.is_dir() else []:
            repo = d.name[len("models--"):].replace("--", "/", 1)
            snap = None
            ref = d / "refs" / "main"
            if ref.is_file():
                cand = d / "snapshots" / ref.read_text().strip()
                if cand.is_dir():
                    snap = cand
            if snap is None:
                snaps = [p for p in (d / "snapshots").glob("*") if p.is_dir()] if (d / "snapshots").is_dir() else []
                if not snaps:
                    continue
                snap = max(snaps, key=lambda p: p.stat().st_mtime)
            weights = [f for f in snap.iterdir() if f.name.endswith((".safetensors", ".bin"))]
            try:
                size = sum(f.stat().st_size for f in weights)
            except OSError:
                size = 0
            incomplete = any((d / "blobs").glob("*.incomplete")) if (d / "blobs").is_dir() else False
            complete = bool(weights) and (snap / "config.json").exists() and not incomplete
            size_gb = size / GiB
            fits = None
            if gpu["total_gb"]:
                fits = size_gb + 1.0 < gpu["total_gb"]
            ram_ok = size_gb < 0.9 * self.args.mem_request_gb
            out.append({"id": repo, "size_gb": size_gb, "complete": complete, "fits_gpu": fits,
                        "fits_ram": ram_ok, "loaded": repo == self.model_id})
        return {"models": out, "suggested": SUGGESTED, "hf_home": str(dl.hf_home())}

    # ---- download (no GPU needed)
    def do_download(self, task, repo_id, token):
        task["progress"] = "Looking up the model on Hugging Face"
        self.show_code(task, "download", "\n".join([
            "from huggingface_hub import snapshot_download",
            f"snapshot_download({repo_id!r}, allow_patterns={dl.PATTERNS!r})   # into $HF_HOME = {dl.hf_home()}"]))

        def progress(done_b, total_b):
            task["fraction"] = done_b / total_b if total_b else None
            task["progress"] = f"Downloaded {done_b / GiB:.1f} of {total_b / GiB:.1f} GB"
            self.touch()

        dl.download(repo_id, token=token or None, on_progress=progress)
        return {"model": repo_id}

    # ---- load / unload
    def unload_locked(self):
        self.model = self.tok = self.structure = self.info = self.weights_index = None
        self.model_id = None
        self.tokstr_cache = {}
        self.special_ids = set()
        self.folder = None
        self.weights_bytes = 0
        self.last = self.last_gen = self.readme = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def do_load(self, task, ref):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        is_path = ref.startswith("/")
        if is_path and not Path(ref).is_dir():
            raise UserError(f"There's no folder at {ref} on the cluster.")
        if not is_path and not re.match(dl.REPO_RE, ref):
            raise UserError("Pick a downloaded model, or give the full path of a model folder on the cluster.")
        task["progress"] = "Unloading the previous model" if self.model is not None else "Reading the tokenizer"
        self.show_code(task, "load", "\n".join([
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            f"tok = AutoTokenizer.from_pretrained({ref!r})",
            f"model = AutoModelForCausalLM.from_pretrained({ref!r}, dtype=torch.bfloat16, device_map={self.device!r})",
            "model.eval()"]))
        self.unload_locked()
        t0 = time.time()
        folder = ref
        if not is_path:
            try:
                from huggingface_hub import snapshot_download
                folder = snapshot_download(ref, local_files_only=True)
            except Exception:  # noqa: BLE001
                folder = None
        expected = bf16_bytes(folder) if folder else 0
        done = threading.Event()

        def watch():
            base = torch.cuda.memory_allocated() if self.device == "cuda" else 0
            while not done.wait(0.5):
                if self.device == "cuda" and expected:
                    got = max(0, torch.cuda.memory_allocated() - base)
                    task["fraction"] = min(0.99, got / expected)
                    task["progress"] = (f"Loading weights onto the GPU: {got / GiB:.1f} of {expected / GiB:.1f} GB. "
                                        "The first load on a machine reads from the file server, so it's slower.")
        try:
            tok = AutoTokenizer.from_pretrained(ref, local_files_only=True)
            task["progress"] = "Loading weights onto the GPU" if self.device == "cuda" else "Loading weights"
            task["fraction"] = 0.0 if expected and self.device == "cuda" else None
            threading.Thread(target=watch, daemon=True).start()
            model = AutoModelForCausalLM.from_pretrained(ref, dtype=torch.bfloat16, device_map=self.device,
                                                         local_files_only=True)
            done.set()
        except OSError as e:
            where = "That folder" if is_path else f"{ref}"
            raise UserError(f"{where} couldn't be loaded. If it's a Hugging Face model, download it first. "
                            f"Details: {e}") from e
        except ValueError as e:
            raise UserError(f"Transformers can't load this as a text model: {e}") from e
        finally:
            done.set()
        task["fraction"] = 1.0
        model.eval()
        self.model, self.tok, self.model_id = model, tok, ref
        self.folder = folder if folder else (ref if is_path else None)
        self.structure = find_structure(model)
        self.special_ids = set(tok.all_special_ids) | {
            i for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items() if getattr(t, "special", False)}
        seen = set()
        for t in list(model.parameters()) + list(model.buffers()):
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                self.weights_bytes += tensor_bytes(t)
        cfg = text_config(model)
        n_params = sum(p.numel() for p in model.parameters())
        gpu = self.gpu_info()
        self.info = {
            "id": ref, "params_b": n_params / 1e9, "layers": self.structure["n_layers"],
            "width": getattr(cfg, "hidden_size", None), "vocab": getattr(cfg, "vocab_size", None),
            "model_type": getattr(model.config, "model_type", None), "architecture": type(model).__name__,
            "dtype": str(model.dtype).replace("torch.", ""),
            "chat_template": bool(getattr(tok, "chat_template", None)),
            "thinking_switch": "enable_thinking" in str(getattr(tok, "chat_template", "") or ""),
            "weights_gb": self.weights_bytes / GiB,
            "instruct_guess": bool(re.search(r"(-it\b|instruct|chat)", ref, re.I)),
            "layers_path": self.structure["layers_path"], "final_norm": self.structure["norm_path"],
            "softcap": self.structure["softcap"], "load_seconds": time.time() - t0,
            "gpu_gb": gpu["ours_gb"],
        }
        self.weights_index = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", ""),
                               "numel": p.numel()} for n, p in model.named_parameters()]
        log(f"Loaded {ref} in {self.info['load_seconds']:.0f}s")
        return self.info

    def require_model(self):
        if self.model is None:
            raise UserError("Load a model first.")

    # ---- tokenization
    def encode(self, prompt, chat=False, system="", thinking=False, to_device=True):
        tok = self.tok
        if chat:
            if not getattr(tok, "chat_template", None):
                raise UserError("This model has no chat template. Turn off Chat format.")
            msgs = [{"role": "user", "content": prompt}]
            if system.strip():
                msgs.insert(0, {"role": "system", "content": system})
            try:
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                               enable_thinking=thinking)
            except Exception:  # noqa: BLE001  (template without a system role)
                msgs = [{"role": "user", "content": (system + "\n\n" + prompt) if system.strip() else prompt}]
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                               enable_thinking=thinking)
            enc = tok(text, add_special_tokens=False, return_tensors="pt")
        else:
            enc = tok(prompt, return_tensors="pt")
        # Some tokenizers (Gemma 4) don't add <bos> to plain text, and those models break without it.
        bos = tok.bos_token_id
        if bos is not None and (enc["input_ids"].shape[1] == 0 or enc["input_ids"][0, 0].item() != bos):
            enc["input_ids"] = torch.cat([torch.tensor([[bos]]), enc["input_ids"]], dim=1)
            enc["attention_mask"] = torch.cat(
                [torch.ones(1, 1, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], dim=1)
        if enc["input_ids"].shape[1] == 0:
            raise UserError("The prompt is empty.")
        return {k: (v.to(self.model.device) if to_device else v) for k, v in enc.items()
                if k in ("input_ids", "attention_mask")}

    def render_input(self, p):
        """The exact tokens a prompt turns into, split into special tokens and ordinary text."""
        if self.model is None or self.tok is None:
            return {"segments": [], "count": 0, "special": 0, "cut": False,
                    "unavailable": "Load a model to see exactly what it receives."}
        enc = self.encode(str(p.get("prompt", "")), bool(p.get("chat")), str(p.get("system", "")),
                          bool(p.get("thinking")), to_device=False)
        ids = enc["input_ids"][0].tolist()
        shown, segs, buf = ids[:8192], [], []

        def flush():
            if buf:
                segs.append({"t": self.tok.decode(buf, skip_special_tokens=False), "s": False})
                buf.clear()
        for i in shown:
            if i in self.special_ids:
                flush()
                segs.append({"t": str(self.tok.convert_ids_to_tokens(i)), "s": True})
            else:
                buf.append(i)
        flush()
        return {"segments": segs, "count": len(ids), "cut": len(ids) > len(shown),
                "special": sum(1 for i in ids if i in self.special_ids)}

    def tokstr(self, i):
        s = self.tokstr_cache.get(i)
        if s is None:
            s = self.tok.decode([i], skip_special_tokens=False)
            if not s or "�" in s:
                t = self.tok.convert_ids_to_tokens(i)
                s = t if t else f"<id:{i}>"       # padded vocab slots have no token
            self.tokstr_cache[i] = s
        return s

    # ---- the code behind each action, shown in the page's console and in the session log
    def show_code(self, task, title, code):
        task["code"] = code
        log(f"{title}:\n" + "\n".join("    " + line for line in code.splitlines()))

    def code_encode(self, p):
        prompt, chat = str(p.get("prompt", "")), bool(p.get("chat"))
        q = lambda x: json.dumps(x, ensure_ascii=False)  # noqa: E731  (a Python string literal)
        lines = [f"prompt = {q(prompt[:4000])}" + ("  # cut to 4,000 characters here" if len(prompt) > 4000 else "")]
        if chat:
            msgs = ([f'{{"role": "system", "content": {q(p.get("system", ""))}}}'] if str(p.get("system", "")).strip() else [])
            msgs.append('{"role": "user", "content": prompt}')
            lines.append("messages = [" + ", ".join(msgs) + "]")
            lines.append("text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, "
                         f"enable_thinking={bool(p.get('thinking'))})")
            lines.append('enc = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)')
        else:
            lines.append('enc = tok(prompt, return_tensors="pt").to(model.device)')
        if self.tok is not None and self.tok.bos_token_id is not None:
            lines.append(f"# Probe Bench also puts {self.tok.bos_token} first if the tokenizer didn't (Gemma needs it)")
        return lines

    def code_capture(self):
        st = self.structure
        lp = st["layers_path"]
        return [
            f"layers = model.get_submodule({lp!r})",
            "resid = [None] * (len(layers) + 1)                # row 0: the input to layer 0 (the embeddings)",
            "def grab(i):",
            "    def hook(mod, args, out): resid[i + 1] = (out[0] if isinstance(out, tuple) else out).detach()",
            "    return hook",
            "hooks = [layers[0].register_forward_pre_hook(",
            "    lambda mod, args, kw: resid.__setitem__(0, (args[0] if args else kw['hidden_states']).detach()), with_kwargs=True)]",
            "hooks += [layer.register_forward_hook(grab(i)) for i, layer in enumerate(layers)]",
            "with torch.no_grad():",
            "    out = model(**enc, use_cache=False)",
            "for h in hooks: h.remove()",
            "resid = torch.stack([r[0] for r in resid])        # [layers + 1, tokens, width]",
        ]

    # ---- generation
    def do_generate(self, task, p):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        self.require_model()
        enc = self.encode(p.get("prompt", ""), bool(p.get("chat")), p.get("system", ""), bool(p.get("thinking")))
        max_new = int(min(max(int(p.get("max_new_tokens", 200)), 1), 4096))
        temperature = float(p.get("temperature", 0) or 0)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        task["steps"] = 0

        class StopFlag(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                task["steps"] += 1
                task["fraction"] = min(1.0, task["steps"] / max_new)
                task["progress"] = f"Generating: {task['steps']} of up to {max_new} tokens"
                return torch.full((input_ids.shape[0],), bool(task["cancel"]), dtype=torch.bool,
                                  device=input_ids.device)

        sampling = (f"do_sample=True, temperature={temperature:g}, top_p={float(p.get('top_p', 0.95) or 0.95):g}"
                    if temperature > 0 else "do_sample=False")
        self.show_code(task, "generate", "\n".join(self.code_encode(p) + [
            f"out = model.generate(**enc, max_new_tokens={max_new}, {sampling})",
            'reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)',
            f"# {int(enc['input_ids'].shape[1])} prompt tokens on {self.device}"]))
        kwargs = dict(**enc, max_new_tokens=max_new, streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([StopFlag()]),
                      pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=float(p.get("top_p", 0.95) or 0.95))
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        box = {}

        cuda = self.device == "cuda" and torch.cuda.is_available()

        def run():
            try:
                with torch.no_grad():
                    out = self.model.generate(**kwargs, return_dict_in_generate=True)
                box["seq"] = out.sequences
                box["kv"] = self.kv_report(getattr(out, "past_key_values", None))
                del out
            except Exception as e:  # noqa: BLE001
                box["err"] = e
                streamer.end()

        task["progress"] = "Generating"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() if cuda else 0
        t0 = time.time()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        for piece in streamer:
            task["partial"] += piece
        th.join()
        if "err" in box:
            raise box["err"]
        seconds = time.time() - t0
        n_prompt = int(enc["input_ids"].shape[1])
        n_new = int(box["seq"].shape[1] - n_prompt) if "seq" in box else 0
        self.last_gen = {"prompt_tokens": n_prompt, "new_tokens": n_new, "at": time.time(),
                         "peak_extra_gb": (torch.cuda.max_memory_allocated() - base) / GiB if cuda else None,
                         "kv": box.get("kv")}
        box.clear()
        gc.collect()
        return {"text": task["partial"], "new_tokens": n_new, "prompt_tokens": n_prompt,
                "seconds": seconds, "tokens_per_s": n_new / seconds if seconds > 0 else None,
                "stopped": bool(task["cancel"]), "model": self.model_id}

    # ---- memory
    def layer_types(self):
        if self.model is None:
            return None
        lt = cget(text_config(self.model), "layer_types")
        return list(lt) if lt else None

    def kv_report(self, cache):
        """Size of the KV cache that generate() built, layer by layer."""
        if cache is None:
            return None
        layers = getattr(cache, "layers", None)
        rows = []
        if layers is not None:
            for i, L in enumerate(layers):
                k, v = getattr(L, "keys", None), getattr(L, "values", None)
                extra = sum(nested_bytes(getattr(L, a, None)) for a in ("conv_states", "recurrent_states", "ssm_states"))
                n = int(k.shape[-2]) if torch.is_tensor(k) and k.ndim >= 2 else 0
                b = tensor_bytes(k) + (tensor_bytes(v) if v is not k else 0) + extra
                rows.append({"layer": i, "kind": type(L).__name__, "tokens": n, "bytes": b})
        elif hasattr(cache, "key_cache"):
            for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
                rows.append({"layer": i, "kind": "DynamicLayer", "tokens": int(k.shape[-2]),
                             "bytes": tensor_bytes(k) + tensor_bytes(v)})
        if not rows:
            return None
        lt = self.layer_types() or []
        for r in rows:
            r["type"] = lt[r["layer"]] if r["layer"] < len(lt) else None
        longest = max(r["tokens"] for r in rows)
        growing = [r for r in rows if r["tokens"] and r["tokens"] >= longest]
        per_token = sum(r["bytes"] / r["tokens"] for r in growing)
        return {"layers": rows, "total_bytes": sum(r["bytes"] for r in rows), "tokens": longest,
                "per_token_bytes": per_token, "growing_layers": len(growing)}

    def kv_estimate(self):
        """Bytes of KV cache per extra token, from the config, for when nothing has been generated yet."""
        d = self.text_config_dict()
        n = d.get("num_hidden_layers") or 0
        heads = d.get("num_attention_heads") or 1
        kv = d.get("num_key_value_heads") or heads
        hd = d.get("head_dim") or ((d.get("hidden_size") or 0) // heads)
        per_layer = d.get("per_layer_config") or {}
        lt = d.get("layer_types") or []
        el = 2
        total = 0
        for i in range(n):
            t = lt[i] if i < len(lt) else "full_attention"
            if t in ("sliding_attention", "linear_attention", "chunked_attention"):
                continue
            o = per_layer.get(str(i), {}) if isinstance(per_layer, dict) else {}
            k_, h_ = o.get("num_key_value_heads", kv), o.get("head_dim", hd)
            total += 2 * k_ * h_ * el
        return total or None

    def memory(self):
        out = {"cuda": False, "weights_gb": self.weights_bytes / GiB if self.model is not None else None,
               "last_gen": self.last_gen, "model": self.model_id,
               "kv_estimate_bytes": self.kv_estimate() if self.model is not None else None,
               "analysis_cpu_gb": tensor_bytes(self.last["resid"]) / GiB if self.last else None}
        if self.device == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            alloc, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            w = self.weights_bytes if self.model is not None else 0
            out.update({
                "cuda": True, "name": torch.cuda.get_device_name(0), "total_gb": total / GiB, "free_gb": free / GiB,
                "parts": [
                    {"key": "weights", "gb": w / GiB},
                    {"key": "working", "gb": max(0, alloc - w) / GiB},
                    {"key": "cache", "gb": max(0, reserved - alloc) / GiB},
                    {"key": "outside", "gb": max(0, (total - free) - reserved) / GiB},
                ],
                "busy": self.busy})
        return out

    # ---- model card
    def text_config_dict(self):
        if self.model is None:
            return {}
        d = self.model.config.to_dict()
        return d.get("text_config") or d

    def card(self):
        self.require_model()
        full = self.model.config.to_dict()
        d = self.text_config_dict()
        st = self.structure
        facts = []

        def add(label, value, note=None):
            if value not in (None, "", [], {}):
                facts.append({"k": label, "v": value, "note": note})
        archs = full.get("architectures") or [type(self.model).__name__]
        add("Architecture", archs[0], f"model_type {full.get('model_type')}")
        add("Layers", st["n_layers"])
        add("Width", d.get("hidden_size"), "Size of the residual stream, the vector each token carries between layers")
        if d.get("num_experts"):
            add("Experts", f"{d['num_experts']} experts, {d.get('num_experts_per_tok') or d.get('top_k_experts')} used per token",
                f"Each expert's MLP width is {d.get('moe_intermediate_size')}" if d.get("moe_intermediate_size") else None)
        add("MLP width", d.get("intermediate_size"), "Hidden size inside each feed-forward block")
        heads, kv, hd = d.get("num_attention_heads"), d.get("num_key_value_heads"), d.get("head_dim")
        if heads:
            add("Attention heads", f"{heads} query, {kv or heads} key/value",
                "Fewer key/value heads than query heads (grouped-query attention) makes the KV cache smaller"
                if kv and kv < heads else None)
        add("Head size", hd)
        per = d.get("per_layer_config") or {}
        lt = list(d.get("layer_types") or [])
        if isinstance(per, dict) and per:
            groups = {}
            for k, o in per.items():
                groups.setdefault(json.dumps(o, sort_keys=True), []).append(int(k))
            words = {"head_dim": "head size {}", "num_key_value_heads": "{} key/value heads", "num_attention_heads": "{} query heads",
                     "intermediate_size": "MLP width {}", "sliding_window": "window {}"}
            for key, ls in groups.items():
                o = json.loads(key)
                kind = lt[ls[0]].replace("_", "-", 1).replace("_", " ") if ls[0] < len(lt) else None
                bits = [words.get(k, k.replace("_", " ") + " {}").format(v) for k, v in o.items()]
                many = len(ls) > 1
                add(f"Different in layer{'s' if many else ''} {ranges(sorted(ls))}", ", ".join(bits).replace("1 key/value heads", "1 key/value head"),
                    (f"These are the {kind} layers" if many else f"This is the {kind} layer") if kind else None)
        if d.get("attention_k_eq_v"):
            add("Keys and values", "Shared in full-attention layers", "attention_k_eq_v: the same projection serves as key and value")
        if lt:
            names = {"sliding_attention": f"sliding-window ({d.get('sliding_window'):,} tokens)" if d.get("sliding_window") else "sliding-window",
                     "full_attention": "full attention", "linear_attention": "linear attention",
                     "chunked_attention": "chunked attention"}
            period = next((q for q in range(1, len(lt) + 1) if all(lt[i] == lt[i % q] for i in range(len(lt)))), len(lt))
            if period < len(lt):
                parts, prev, cnt = [], None, 0
                for t in lt[:period] + [None]:
                    if t == prev:
                        cnt += 1
                    else:
                        if prev is not None:
                            parts.append(f"{cnt} {names.get(prev, prev.replace('_', ' '))}")
                        prev, cnt = t, 1
                add("Layer pattern", ", then ".join(parts) + f", repeated {len(lt) // period} times",
                    "Sliding-window layers only attend to recent tokens, so their KV cache stops growing")
            else:
                from collections import Counter
                add("Layer types", ", ".join(f"{n} {names.get(t, t)}" for t, n in Counter(lt).items()))
        add("Context length", f"{d['max_position_embeddings']:,} tokens" if d.get("max_position_embeddings") else None,
            "The longest input the model was configured for")
        add("Vocabulary", f"{d['vocab_size']:,} tokens" if d.get("vocab_size") else None)
        rp = d.get("rope_parameters") or {}
        if d.get("rope_theta"):
            add("Rotary base", d["rope_theta"])
        elif isinstance(rp, dict) and rp:
            if "rope_theta" in rp:
                add("Rotary base", rp["rope_theta"], rp.get("rope_type"))
            else:
                add("Rotary base", ", ".join(f"{k.replace('_attention', '')}: {v.get('rope_theta')}" for k, v in rp.items()
                                             if isinstance(v, dict)))
        add("Activation", d.get("hidden_activation") or d.get("hidden_act") or d.get("activation_function"))
        tied = d.get("tie_word_embeddings", full.get("tie_word_embeddings"))
        add("Tied embeddings", None if tied is None else ("Yes" if tied else "No"),
            "Yes means the output head reuses the input embedding matrix")
        add("Logit softcap", d.get("final_logit_softcapping"), "Final logits are squashed with tanh to stay within this value")
        add("Norm epsilon", d.get("rms_norm_eps") or d.get("layer_norm_eps") or d.get("layer_norm_epsilon"))
        add("Stored as", d.get("dtype") or full.get("dtype") or full.get("torch_dtype"))
        extra = [k.replace("_config", "") for k in full if k.endswith("_config") and k != "text_config" and full[k]]
        if extra:
            add("Other parts", ", ".join(extra) + " encoder" + ("s" if len(extra) > 1 else ""),
                "Loaded but not used by Probe Bench, which works with text")

        # where the parameters are
        lp = st["layers_path"] + "."
        parts = {}
        layer_sig = {}
        for n, prm in self.model.named_parameters():
            low = n.lower()
            if any(x in low for x in ("vision", "visual", "audio", "image", "multi_modal", "mm_")):
                part = "Vision and audio encoders"
            elif n.startswith(lp):
                idx, _, rel = n[len(lp):].partition(".")
                layer_sig.setdefault(int(idx) if idx.isdigit() else idx, []).append((rel, tuple(prm.shape)))
                r = rel.lower()
                mod = r.rsplit(".", 2)[-2] if r.count(".") >= 1 else r
                if "norm" in mod or mod.startswith("ln"):
                    part = "Norms"
                elif any(x in r for x in ("expert", "moe", "router", "gate.weight")) and "gate_proj" not in r:
                    part = "Experts and router"
                elif any(x in r for x in ("attn", "attention", "mixer")):
                    part = "Attention"
                elif any(x in r for x in ("mlp", "feed_forward", "ffn")):
                    part = "MLP"
                else:
                    part = "Other layer parts"
            elif any(x in low for x in ("embed", "wte", "wpe")):
                part = "Embeddings"
            elif "lm_head" in low or "output" in low.split(".")[0]:
                part = "Output head"
            elif "norm" in low:
                part = "Norms"
            else:
                part = "Other"
            e = parts.setdefault(part, {"part": part, "params": 0, "bytes": 0})
            e["params"] += prm.numel()
            e["bytes"] += tensor_bytes(prm)
        total = sum(e["params"] for e in parts.values()) or 1
        for e in parts.values():
            e["share"] = e["params"] / total
        groups = {}
        for i, sig in layer_sig.items():
            groups.setdefault(tuple(sig), []).append(i)
        layer_groups = [{"layers": sorted(ls, key=lambda x: (isinstance(x, str), x)),
                         "range": ranges(sorted(x for x in ls if isinstance(x, int))),
                         "type": lt[ls[0]] if isinstance(ls[0], int) and ls[0] < len(lt) else None,
                         "params": [{"rel": r, "shape": list(sh)} for r, sh in sig]}
                        for sig, ls in groups.items()]
        layer_groups.sort(key=lambda g: g["layers"][0] if isinstance(g["layers"][0], int) else 1e9)
        url = None if (self.model_id or "").startswith("/") else f"https://huggingface.co/{self.model_id}"
        return {"model": self.model_id, "url": url, "facts": facts,
                "parts": sorted(parts.values(), key=lambda e: -e["params"]), "total_params": total,
                "layers_path": st["layers_path"], "layer_types": lt or None, "layer_groups": layer_groups,
                "config": full}

    def card_readme(self):
        """The model's README (its Hugging Face model card): from disk, or fetched once if online."""
        self.require_model()
        if self.readme is not None:
            return self.readme
        folder = Path(self.folder) if self.folder else None
        text, source = None, None
        if folder and (folder / "README.md").is_file():
            text, source = (folder / "README.md").read_text(errors="replace"), "downloaded"
        elif not (self.model_id or "").startswith("/"):
            box = {}

            def fetch():
                try:
                    from huggingface_hub import hf_hub_download
                    box["path"] = hf_hub_download(self.model_id, "README.md")
                except Exception as e:  # noqa: BLE001
                    box["err"] = str(e)
            th = threading.Thread(target=fetch, daemon=True)
            th.start()
            th.join(10)
            if box.get("path"):
                text, source = Path(box["path"]).read_text(errors="replace"), "fetched"
        self.readme = {"text": text[:400_000] if text else None, "source": source}
        return self.readme

    # ---- activations
    def capture(self, enc):
        st = self.structure
        layers = st["layers"]
        resid = [None] * (len(layers) + 1)

        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            resid[0] = h.detach()

        def make(i):
            def post(mod, args, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                resid[i + 1] = h.detach()
            return post

        hooks = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
        hooks += [layer.register_forward_hook(make(i)) for i, layer in enumerate(layers)]
        try:
            with torch.no_grad():
                out = self.model(**enc, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        if any(r is None for r in resid):
            raise UserError("Couldn't capture every layer's activations for this architecture.")
        return resid, out.logits

    def lens_logits(self, h):
        st = self.structure
        x = st["norm"](h) if st["norm"] is not None else h
        logits = st["head"](x).float()
        if st["softcap"]:
            logits = torch.tanh(logits / st["softcap"]) * st["softcap"]
        return logits

    def analyze_inputs(self, p):
        self.require_model()
        enc = self.encode(p.get("prompt", ""), bool(p.get("chat")), p.get("system", ""), bool(p.get("thinking")))
        n = enc["input_ids"].shape[1]
        truncated = n > MAX_ANALYZE_TOKENS
        if truncated:
            enc = {k: v[:, :MAX_ANALYZE_TOKENS] for k, v in enc.items()}
        return enc, truncated, n

    def do_analyze(self, task, p):
        enc, truncated, n_total = self.analyze_inputs(p)
        st = self.structure
        lens = [f"norm, head = model.get_submodule({st['norm_path']!r}), model.get_output_embeddings()" if st["norm_path"]
                else "norm, head = (lambda x: x), model.get_output_embeddings()",
                "logits = head(norm(resid)).float()                # the logit lens: every layer read as if it were the last"]
        if st["softcap"]:
            lens.append(f"logits = torch.tanh(logits / {st['softcap']:g}) * {st['softcap']:g}   # this model's final logit softcap")
        self.show_code(task, "analyze", "\n".join(self.code_encode(p) + (
            [f'enc = {{k: v[:, :{MAX_ANALYZE_TOKENS}] for k, v in enc.items()}}   # first {MAX_ANALYZE_TOKENS} tokens'] if truncated else [])
            + self.code_capture() + [
            "state_size = resid.float().norm(dim=-1)            # the State size map: [layers + 1, tokens]",
            "# resid[:, :, c] is the One channel map; resid[:, t] is one token at every layer; resid[l] is one layer for every token"] + lens + [
            "top5 = logits.softmax(-1).topk(5)                  # the prediction table and the Top prediction map"]))
        task["progress"] = "Running the prompt through the model"
        resid, logits = self.capture(enc)
        ids = enc["input_ids"][0].tolist()
        task["progress"] = "Computing the layer-by-layer predictions"
        norms, lens_ids, lens_p = [], [], []
        used = set(ids)
        with torch.no_grad():
            for li, h in enumerate(resid):
                task["fraction"] = li / len(resid)
                task["progress"] = f"Computing the layer-by-layer predictions: layer {li} of {len(resid) - 1}"
                x = h[0]
                norms.append([round(v, 3) for v in x.float().norm(dim=-1).tolist()])
                probs = torch.softmax(self.lens_logits(x), dim=-1)
                pv, pi = probs.topk(LENS_TOP_K, dim=-1)
                lens_ids.append(pi.tolist())
                lens_p.append([[round(v, 4) for v in row] for row in pv.tolist()])
                for row in lens_ids[-1]:
                    used.update(row)
            final_top = torch.softmax(logits[0].float(), -1).argmax(-1).tolist()
        matches = lens_ids[-1] and [row[0] for row in lens_ids[-1]] == final_top
        task["progress"] = "Finding the largest channels"
        stack = torch.stack([h[0] for h in resid]).cpu().float()      # [layers + 1, tokens, width]
        del resid
        aid = secrets.token_hex(4)
        self.last = {"aid": aid, "resid": stack}
        width = stack.shape[-1]
        T = stack.shape[1]

        def top(view, offset):
            a = view.abs().reshape(-1, width)
            mx, idx = a.max(0)
            k = min(16, width)
            vals, chans = mx.topk(k)
            out = []
            for v, c in zip(vals.tolist(), chans.tolist()):
                flat = idx[c].item()
                li, pos = divmod(flat, view.shape[1])
                out.append({"c": c, "max_abs": round(v, 4), "layer": li, "pos": pos + offset,
                            "value": round(view[li, pos, c].item(), 4)})
            return out
        top_any = top(stack, 0)
        top_skip = top(stack[:, 1:, :], 1) if T >= 3 else top_any
        return {"model": self.model_id, "aid": aid, "width": width, "top_channels": top_any,
                "top_channels_skip": top_skip, "ids": ids, "tokens": [self.tokstr(i) for i in ids],
                "vocab": {str(i): self.tokstr(i) for i in used}, "layers": stack.shape[0] - 1,
                "norms": norms, "lens_ids": lens_ids, "lens_p": lens_p, "truncated": truncated,
                "total_tokens": n_total, "max_tokens": MAX_ANALYZE_TOKENS, "lens_matches_output": bool(matches),
                "chat": bool(p.get("chat"))}

    def last_analysis(self, aid):
        if not self.last or self.last["aid"] != aid:
            raise UserError("These results are from an earlier analysis. Run Analyze again.")
        return self.last["resid"]

    def channel(self, aid, c):
        r = self.last_analysis(aid)
        if not 0 <= c < r.shape[-1]:
            raise UserError(f"Pick a channel from 0 to {r.shape[-1] - 1}.")
        vals = r[:, :, c].tolist()
        return {"channel": c, "values": [[float(f"{v:.5g}") for v in row] for row in vals]}

    def vector(self, aid, pos):
        r = self.last_analysis(aid)
        if not 0 <= pos < r.shape[1]:
            raise UserError("That token isn't in the analysis.")
        return r[:, pos, :].contiguous().numpy().astype("<f4").tobytes(), (r.shape[0], r.shape[-1])

    def layer_slice(self, aid, layer):
        """Every channel of every token at one layer: [tokens, width], float32."""
        r = self.last_analysis(aid)
        if not 0 <= layer < r.shape[0]:
            raise UserError(f"Pick a layer from 0 to {r.shape[0] - 1}.")
        return r[layer].contiguous().numpy().astype("<f4").tobytes(), (r.shape[1], r.shape[-1])

    def do_export(self, task, p):
        enc, truncated, _ = self.analyze_inputs(p)
        self.show_code(task, "save activations", "\n".join(self.code_encode(p) + self.code_capture() + [
            "torch.save({'residual_stream': resid.to(torch.bfloat16).cpu(), 'tokens': ...}, path)   # in ~/probebench/exports"]))
        task["progress"] = "Running the prompt through the model"
        resid, _ = self.capture(enc)
        stack = torch.stack([h[0] for h in resid]).to(torch.bfloat16).cpu()
        ids = enc["input_ids"][0].cpu()
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{slug(self.model_id.split('/')[-1])}.pt"
        path = self.exports_dir / name
        task["progress"] = "Saving"
        torch.save({
            "model": self.model_id, "prompt": p.get("prompt", ""), "chat": bool(p.get("chat")),
            "system": p.get("system", ""), "input_ids": ids, "tokens": [self.tokstr(i) for i in ids.tolist()],
            "residual_stream": stack, "truncated": truncated,
            "description": ("residual_stream has shape [layers + 1, tokens, width]. Row 0 is the input to the "
                            "first layer (the embeddings); row i is the output of layer i, before the final norm."),
        }, path)
        return {"path": str(path), "name": name, "size_mb": path.stat().st_size / 2 ** 20,
                "shape": list(stack.shape)}

    # ---- weights
    def do_weight_stats(self, task, name):
        self.require_model()
        self.show_code(task, "weight statistics", "\n".join([
            f"w = model.get_parameter({name!r}).float()",
            "w.mean(), w.std(), w.min(), w.max(), w.norm()      # the numbers at the top",
            "torch.histc(w, bins=60)                            # the histogram (from a 4M-value sample for big matrices)",
            "s = torch.linalg.svdvals(w)                        # singular values (Probe Bench uses the Gram matrix, same result)",
            "stable_rank = (s ** 2).sum() / s[0] ** 2"]))
        params = dict(self.model.named_parameters())
        if name not in params:
            raise UserError(f"There's no parameter called {name}.")
        w = params[name].detach()
        flat = w.reshape(-1)
        n = flat.numel()
        task["progress"] = "Computing statistics"
        task["fraction"] = 0.1
        total = torch.zeros((), dtype=torch.float64, device=w.device)
        total_sq = torch.zeros((), dtype=torch.float64, device=w.device)
        mn, mx = float("inf"), float("-inf")
        for chunk in flat.split(32 * 2 ** 20):
            c = chunk.float()
            total += c.sum(dtype=torch.float64)
            total_sq += (c * c).sum(dtype=torch.float64)
            mn = min(mn, c.min().item())
            mx = max(mx, c.max().item())
        mean = (total / n).item()
        var = max((total_sq / n).item() - mean * mean, 0.0)
        frob = math.sqrt(total_sq.item())
        sample = flat if n <= 4_000_000 else flat[torch.randint(n, (4_000_000,), device=w.device)]
        sample = sample.float()
        lo, hi = sample.quantile(torch.tensor([0.001, 0.999], device=sample.device)).tolist() \
            if sample.numel() > 1 else (mn, mx)
        if hi <= lo:
            lo, hi = mn, mx if mx > mn else mn + 1e-6
        bins = 60
        counts = torch.histc(sample.clamp(lo, hi), bins=bins, min=lo, max=hi)
        scale = n / sample.numel()
        hist = {"lo": lo, "hi": hi, "counts": [round(c * scale) for c in counts.tolist()],
                "below": round((sample < lo).sum().item() * scale), "above": round((sample > hi).sum().item() * scale)}
        result = {"name": name, "shape": list(w.shape), "dtype": str(w.dtype).replace("torch.", ""), "numel": n,
                  "mean": mean, "std": math.sqrt(var), "min": mn, "max": mx, "frobenius": frob, "hist": hist,
                  "sampled": n > 4_000_000, "singular_values": None, "sv_note": None}
        if w.ndim == 2:
            r, c = w.shape
            k = min(r, c)
            if k > 8192:
                result["sv_note"] = "Too large for an exact spectrum here."
            else:
                task["progress"] = "Computing singular values"
                task["fraction"] = 0.5
                # Singular values from the Gram matrix of the smaller side, built in chunks.
                m = w if c <= r else w.t()
                gram = torch.zeros((k, k), dtype=torch.float32, device=w.device)
                for chunk in m.split(8192, dim=0):
                    cf = chunk.float()
                    gram += cf.t() @ cf
                ev = torch.linalg.eigvalsh(gram.double() if self.device == "cpu" else gram).clamp_min(0)
                s = ev.flip(0).sqrt().float()
                s_list = s.tolist()
                total_s = s.sum().item()
                pnorm = s / total_s if total_s > 0 else s
                pnorm = pnorm[pnorm > 0]
                eff_rank = math.exp(-(pnorm * pnorm.log()).sum().item()) if pnorm.numel() else 0.0
                stable_rank = (s.pow(2).sum() / s[0].pow(2)).item() if s_list and s_list[0] > 0 else 0.0
                result.update({"singular_values": [round(v, 5) for v in s_list], "effective_rank": eff_rank,
                               "stable_rank": stable_rank, "spectral_norm": s_list[0] if s_list else 0.0})
        return result


    def do_weight_across(self, task, rel):
        """Size and spectrum summaries of the same matrix in every layer."""
        self.require_model()
        self.show_code(task, "across layers", "\n".join([
            f"for i in range({self.structure['n_layers']}):",
            f"    w = model.get_parameter(f'{self.structure['layers_path']}.{{i}}.{rel}').float()",
            "    fro = w.norm()",
            "    sigma = torch.linalg.matrix_norm(w, ord=2)    # Probe Bench estimates this with 40 steps of power iteration",
            "    stable_rank = fro ** 2 / sigma ** 2"]))
        params = dict(self.model.named_parameters())
        lp = self.structure["layers_path"]
        n = self.structure["n_layers"]
        lt = self.layer_types() or []
        rows = []
        for i in range(n):
            task["fraction"] = i / n
            task["progress"] = f"Layer {i} of {n - 1}"
            if task["cancel"]:
                raise Cancelled()
            w = params.get(f"{lp}.{i}.{rel}")
            row = {"layer": i, "type": lt[i] if i < len(lt) else None}
            if w is None:
                rows.append(row)
                continue
            wf = w.detach().float()
            fro = wf.norm().item()
            row.update({"frobenius": fro, "rms": fro / math.sqrt(wf.numel()), "max_abs": wf.abs().max().item(),
                        "shape": list(w.shape)})
            if wf.ndim == 2 and min(wf.shape) > 1:
                g = torch.Generator(device=wf.device).manual_seed(0)
                v = torch.randn(wf.shape[1], device=wf.device, generator=g)
                v /= v.norm()
                sigma = 0.0
                for _ in range(40):
                    u = wf @ v
                    sigma = u.norm().item()
                    if sigma == 0:
                        break
                    v = wf.t() @ (u / sigma)
                    v /= v.norm()
                row["spectral"] = sigma
                row["stable_rank"] = (fro * fro) / (sigma * sigma) if sigma > 0 else None
            del wf
            rows.append(row)
        return {"rel": rel, "rows": rows}


# --------------------------------------------------------------------------- HTTP

UNAUTHORIZED_PAGE = """<!doctype html><meta charset="utf-8"><title>Probe Bench</title>
<body style="font-family: 'Avenir Next', 'Segoe UI', system-ui, sans-serif; max-width: 34rem;
margin: 4rem auto; padding: 0 1rem; color: #1f2a36; line-height: 1.5">
<h1 style="color:#003057;font-size:1.4rem">This session needs its access link</h1>
<p>Open Probe Bench with the link that <code>probebench</code> printed in Terminal.
It includes a one-time key, so only you can use this GPU session.</p></body>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ProbeBench/1"
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # quiet access log
        pass

    @property
    def lab(self) -> Lab:
        return self.server.lab

    # ---- helpers
    def send_json(self, obj, code=200):
        body = json.dumps(clean(obj), separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text, code=200, ctype="text/html; charset=utf-8", extra=None):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def cookie_name(self):
        return f"pb_{self.lab.session_id}".replace("-", "_")

    def authorized(self):
        supplied = self.headers.get("X-Probe-Token", "")
        if not supplied:
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == self.cookie_name():
                    supplied = v
        return bool(supplied) and hmac.compare_digest(supplied, self.lab.token)

    def body_json(self):
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise UserError("Requests must be JSON.")
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise UserError("That request is too large.")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise UserError("The request wasn't valid JSON.") from e

    # ---- routing
    def do_GET(self):
        self.handle_safely("GET")

    def do_POST(self):
        self.handle_safely("POST")

    def handle_safely(self, method):
        try:
            self.route(method)
        except UserError as e:
            self.send_json({"error": str(e)}, 400)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc())
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def route(self, method):
        url = urlparse(self.path)
        path = url.path
        lab = self.lab

        if path == "/health":
            return self.send_json({"ok": True, "session": lab.session_id})

        if path == "/" and method == "GET":
            token = (parse_qs(url.query).get("token") or [""])[0]
            if token and hmac.compare_digest(token, lab.token):
                lab.touch()
                cookie = f"{self.cookie_name()}={lab.token}; Path=/; HttpOnly; SameSite=Strict"
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", cookie)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self.authorized():
                return self.send_text(UNAUTHORIZED_PAGE, 401)
            return self.send_text((HERE / "index.html").read_text())

        if not path.startswith("/api/"):
            return self.send_json({"error": "Not found"}, 404)
        if not self.authorized():
            return self.send_json({"error": "This session needs its access link."}, 401)

        if method == "GET":
            if path == "/api/status":
                return self.send_json(lab.status())
            if path.startswith("/api/task/"):
                return self.send_json(lab.task_view(path.rsplit("/", 1)[1]))
            lab.touch()
            if path == "/api/models":
                return self.send_json(lab.list_models())
            if path == "/api/weights":
                lab.require_model()
                return self.send_json({"model": lab.model_id, "params": lab.weights_index})
            if path == "/api/memory":
                return self.send_json(lab.memory())
            if path == "/api/card":
                return self.send_json(lab.card())
            if path == "/api/card/readme":
                return self.send_json(lab.card_readme())
            q = parse_qs(url.query)
            if path == "/api/act/channel":
                return self.send_json(lab.channel((q.get("aid") or [""])[0], int((q.get("c") or ["-1"])[0])))
            if path == "/api/act/vector":
                data, shape = lab.vector((q.get("aid") or [""])[0], int((q.get("pos") or ["-1"])[0]))
                return self.send_bytes(data, {"X-Shape": f"{shape[0]},{shape[1]}"})
            if path == "/api/act/layer":
                data, shape = lab.layer_slice((q.get("aid") or [""])[0], int((q.get("layer") or ["-1"])[0]))
                return self.send_bytes(data, {"X-Shape": f"{shape[0]},{shape[1]}"})
            if path.startswith("/api/exports/"):
                return self.send_export(path.rsplit("/", 1)[1])
            return self.send_json({"error": "Not found"}, 404)

        body = self.body_json()
        lab.touch()
        if path == "/api/keepalive":
            return self.send_json({"ok": True, "idle_left_s": lab.idle_left()})
        if path == "/api/shutdown":
            self.send_json({"ok": True})
            lab.shutdown("ended from the page")
            return
        if path == "/api/download":
            model = str(body.get("model", "")).strip()
            tid = lab.start_task("download", lab.do_download, model, str(body.get("token", "")).strip(),
                                 uses_gpu=False)
            return self.send_json({"task": tid})
        if path == "/api/load":
            return self.send_json({"task": lab.start_task("load", lab.do_load, str(body.get("model", "")).strip())})
        if path == "/api/unload":
            def unload(task):
                lab.unload_locked()
                return {"ok": True}
            return self.send_json({"task": lab.start_task("unload", unload)})
        if path == "/api/generate":
            return self.send_json({"task": lab.start_task("generate", lab.do_generate, body)})
        if path == "/api/analyze":
            return self.send_json({"task": lab.start_task("analyze", lab.do_analyze, body)})
        if path == "/api/export":
            return self.send_json({"task": lab.start_task("export", lab.do_export, body)})
        if path == "/api/render":
            return self.send_json(lab.render_input(body))
        if path == "/api/memory/release":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.send_json(lab.memory())
        if path == "/api/weights/across":
            return self.send_json({"task": lab.start_task("weights", lab.do_weight_across, str(body.get("rel", "")))})
        if path == "/api/weights/stats":
            return self.send_json({"task": lab.start_task("weights", lab.do_weight_stats, str(body.get("name", "")))})
        if path == "/api/cancel":
            t = lab.tasks.get(str(body.get("task", "")))
            if t:
                t["cancel"] = True
            return self.send_json({"ok": bool(t)})
        return self.send_json({"error": "Not found"}, 404)

    def send_export(self, name):
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.pt", name):
            raise UserError("Not a valid export name.")
        f = self.lab.exports_dir / name
        if not f.is_file():
            return self.send_json({"error": "That file no longer exists."}, 404)
        size = f.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with f.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--lab-dir", required=True)
    ap.add_argument("--idle-minutes", type=float, default=30)
    ap.add_argument("--gpu-request-mb", type=int, default=30000)
    ap.add_argument("--mem-request-gb", type=float, default=32)
    ap.add_argument("--device", default=os.environ.get("PROBEBENCH_DEVICE", "cuda"))
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    sdir = Path(args.session_dir)
    token = (sdir / "token").read_text().strip()
    if len(token) < 16:
        sys.exit("session token missing or too short")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("No GPU is visible. Was the job started with request_gpus and apptainer --nv?")

    lab = Lab(args, token)
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    httpd.daemon_threads = True
    httpd.lab = lab
    lab.httpd = httpd
    host, port = socket.getfqdn(), httpd.server_address[1]
    write_atomic(sdir / "server.json", json.dumps({"host": host, "port": port, "pid": os.getpid(),
                                                    "started": lab.started}))
    write_atomic(sdir / "server.txt", f"{host} {port}\n")
    signal.signal(signal.SIGTERM, lambda *_: lab.shutdown("stopped"))
    log(f"Probe Bench listening on {host}:{port} (session {lab.session_id}, "
        f"ends after {args.idle_minutes:g} idle minutes)")
    threading.Thread(target=lab.watchdog, daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        lab.ended = lab.ended or "stopped"
        log(f"Session ended ({lab.ended}).")
        try:
            write_atomic(sdir / "ended.json", json.dumps({"reason": lab.ended, "at": time.time()}))
        except OSError:
            pass


if __name__ == "__main__":
    main()
