#!/usr/bin/env python3
"""Download a model's weights and tokenizer from Hugging Face into HF_HOME.

Used by the web interface (inside a GPU session) and by `probebench download`
(as a CPU-only job). Only top-level weights, configs and tokenizer files are
fetched; extra folders such as `original/` or ONNX/GGUF exports are skipped.

Command line:  python download.py <org/model>
"""
import fnmatch
import os
import sys
import threading
import time
from pathlib import Path

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken", "README.md"]
IGNORE = ["*/*"]  # skip subfolders
REPO_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"


class DownloadError(Exception):
    """A download problem with a message meant for the person using the tool."""


def hf_home():
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def repo_dir(repo_id):
    return hf_home() / "hub" / ("models--" + repo_id.replace("/", "--"))


def bytes_present(repo_id):
    blobs = repo_dir(repo_id) / "blobs"
    if not blobs.is_dir():
        return 0
    total = 0
    for f in blobs.iterdir():
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def plan(repo_id, token=None):
    """Return (allow_patterns, total_bytes) for the files we will fetch."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except GatedRepoError:
        raise DownloadError(
            f"{repo_id} is gated. Accept its license on huggingface.co, then save a "
            "Hugging Face token (run `probebench token` on your Mac, or paste one here).")
    except RepositoryNotFoundError:
        raise DownloadError(f"There is no model called {repo_id} on Hugging Face, "
                            "or it is private. Check the spelling, including capital letters.")
    siblings = info.siblings or []
    names = [s.rfilename for s in siblings]
    patterns = list(PATTERNS)
    if not any(n.endswith(".safetensors") and "/" not in n for n in names):
        patterns.append("*.bin")
    chosen = [s for s in siblings
              if "/" not in s.rfilename and any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    if not any(s.rfilename.endswith((".safetensors", ".bin")) for s in chosen):
        raise DownloadError(f"{repo_id} has no PyTorch weights at the top level of its repository.")
    return patterns, sum((s.size or 0) for s in chosen)


def download(repo_id, token=None, on_progress=None):
    """Download repo_id into HF_HOME. Calls on_progress(done_bytes, total_bytes) about once a second."""
    import re
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    if not re.match(REPO_RE, repo_id or ""):
        raise DownloadError("Use the Hugging Face name of the model, like google/gemma-4-12B-it.")
    patterns, total = plan(repo_id, token)
    done = threading.Event()

    def watch():
        while not done.wait(1.0):
            if on_progress:
                on_progress(min(bytes_present(repo_id), total), total)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        path = snapshot_download(repo_id, allow_patterns=patterns, ignore_patterns=IGNORE, token=token)
    except GatedRepoError:
        raise DownloadError(f"{repo_id} is gated. Accept its license on huggingface.co and save a token.")
    finally:
        done.set()
    if on_progress:
        on_progress(total, total)
    return path


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: download.py <org/model>")
    repo_id = sys.argv[1]
    last = [0.0]

    def report(done_b, total_b):
        now = time.time()
        if now - last[0] >= 10 or done_b >= total_b:
            last[0] = now
            pct = 100.0 * done_b / total_b if total_b else 100.0
            print(f"PROGRESS {done_b} {total_b} {pct:.0f}%", flush=True)

    try:
        path = download(repo_id, on_progress=report)
    except DownloadError as e:
        print(f"ERROR {e}", flush=True)
        sys.exit(2)
    print(f"DONE {path}", flush=True)


if __name__ == "__main__":
    main()
