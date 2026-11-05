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
                "cancel": False, "fraction": None}
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
        return {k: t[k] for k in ("id", "kind", "status", "progress", "partial", "result", "error", "fraction")}

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
        self.structure = find_structure(model)
        cfg = text_config(model)
        n_params = sum(p.numel() for p in model.parameters())
        gpu = self.gpu_info()
        self.info = {
            "id": ref, "params_b": n_params / 1e9, "layers": self.structure["n_layers"],
            "width": getattr(cfg, "hidden_size", None), "vocab": getattr(cfg, "vocab_size", None),
            "model_type": getattr(model.config, "model_type", None), "architecture": type(model).__name__,
            "dtype": str(model.dtype).replace("torch.", ""),
            "chat_template": bool(getattr(tok, "chat_template", None)),
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
    def encode(self, prompt, chat=False, system="", thinking=False):
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
        return {k: v.to(self.model.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def tokstr(self, i):
        s = self.tokstr_cache.get(i)
        if s is None:
            s = self.tok.decode([i], skip_special_tokens=False)
            if not s or "�" in s:
                s = str(self.tok.convert_ids_to_tokens(i))
            self.tokstr_cache[i] = s
        return s

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

        kwargs = dict(**enc, max_new_tokens=max_new, streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([StopFlag()]),
                      pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=float(p.get("top_p", 0.95) or 0.95))
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        box = {}

        def run():
            try:
                with torch.no_grad():
                    box["out"] = self.model.generate(**kwargs)
            except Exception as e:  # noqa: BLE001
                box["err"] = e
                streamer.end()

        task["progress"] = "Generating"
        t0 = time.time()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        for piece in streamer:
            task["partial"] += piece
        th.join()
        if "err" in box:
            raise box["err"]
        seconds = time.time() - t0
        n_new = int(box["out"].shape[1] - enc["input_ids"].shape[1]) if "out" in box else 0
        return {"text": task["partial"], "new_tokens": n_new, "prompt_tokens": int(enc["input_ids"].shape[1]),
                "seconds": seconds, "tokens_per_s": n_new / seconds if seconds > 0 else None,
                "stopped": bool(task["cancel"]), "model": self.model_id}

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
        return {"model": self.model_id, "ids": ids, "tokens": [self.tokstr(i) for i in ids],
                "vocab": {str(i): self.tokstr(i) for i in used}, "layers": len(resid) - 1,
                "norms": norms, "lens_ids": lens_ids, "lens_p": lens_p, "truncated": truncated,
                "total_tokens": n_total, "max_tokens": MAX_ANALYZE_TOKENS, "lens_matches_output": bool(matches),
                "chat": bool(p.get("chat"))}

    def do_export(self, task, p):
        enc, truncated, _ = self.analyze_inputs(p)
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
