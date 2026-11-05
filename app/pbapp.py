#!/usr/bin/env python3
"""Probe Bench for Mac: the local control server behind the Probe Bench app.

It runs on your Mac (started by Probe Bench.app), serves the control page at
http://127.0.0.1:7850, and does everything the page asks for over SSH:
signing in through the castle gateway (password and Duo prompts are shown on the
page), setting up your SSH key, preparing the cluster, starting HTCondor GPU
sessions, tunnelling your browser to them and ending them.

Standard library only, compatible with the Python 3.9 that ships with Apple's
command-line tools. The cluster side is the same as the `probebench` command:
cluster/labctl.sh, run on compute01.
"""
from __future__ import annotations

import hmac
import json
import os
import random
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import io
import threading
import time
import traceback
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VERSION = "2.0"
HERE = Path(__file__).resolve().parent            # .../ProbeBench/app
ROOT = HERE.parent                                # .../ProbeBench (has cluster/)
STATE_DIR = Path(os.environ.get("PROBEBENCH_STATE", str(ROOT))).expanduser()
HOME = Path.home()
CONF = HOME / ".probebench.conf"
SSH_DIR = HOME / ".ssh"
SSH_CONFIG = SSH_DIR / "config"
GW, SUBMIT = "probebench-gw", "probebench-submit"
RLAB = "~/probebench"
LABCTL = f"bash {RLAB}/app/labctl.sh"
GATEWAYS = {"castle": "castle.isye.gatech.edu", "keep": "keep.isye.gatech.edu"}
ACCOUNT_FORM = "https://webforms.isye.gatech.edu/other-forms/isye-unix-account-request"

SIZES = {
    "standard": {"gpu_mb": 30000, "mem_gb": 32, "title": "Standard",
                 "blurb": "A GPU with 30 GB or more. Fits models up to about 13B parameters, such as Gemma 4 12B."},
    "large": {"gpu_mb": 40000, "mem_gb": 48, "title": "Large",
              "blurb": "40 GB or more (A6000 or A100). Fits up to about 20B parameters, or long prompts and big batches."},
    "a100": {"gpu_mb": 70000, "mem_gb": 64, "title": "A100",
             "blurb": "An 80 GB A100. Fits up to about 35B parameters, such as Qwen 3.8 27B. There are only three, so expect to wait."},
}
SUGGESTED = [
    {"id": "google/gemma-4-12B-it", "size": "22 GB", "note": "Gemma 4 12B, instruction-tuned"},
    {"id": "google/gemma-4-12B", "size": "22 GB", "note": "Gemma 4 12B, base model"},
    {"id": "Qwen/Qwen3.5-9B", "size": "18 GB", "note": "Qwen 3.5 9B"},
    {"id": "allenai/Olmo-3-7B-Instruct", "size": "14 GB", "note": "Olmo 3 7B, fully open training data"},
    {"id": "Qwen/Qwen3.8-27B", "size": "52 GB", "note": "Qwen 3.8 27B, needs the A100 size"},
]
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

SETUP_STEPS = [
    ("signin", "Sign in to the cluster"),
    ("key", "Set up your SSH key"),
    ("copy", "Copy Probe Bench to the cluster"),
    ("container", "Prepare the software"),
]


def now():
    return time.time()


def read_conf():
    conf = {"ISYE_USER": "", "GATEWAY": GATEWAYS["castle"]}
    if CONF.exists():
        for line in CONF.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in conf:
                conf[k.strip()] = v.strip().strip("'\"")
    return conf


def write_conf(conf):
    CONF.write_text(f"ISYE_USER={conf['ISYE_USER']}\nGATEWAY={conf['GATEWAY']}\n")


def ssh_block(user, gateway):
    common = ("    ControlMaster auto\n    ControlPath ~/.ssh/cm-%r@%h:%p\n    ControlPersist 8h\n"
              "    ServerAliveInterval 60\n    ServerAliveCountMax 3\n    StrictHostKeyChecking accept-new\n")
    return (f"# >>> probebench >>>  (written by probebench setup; safe to delete)\n"
            f"Host {GW}\n    HostName {gateway}\n    User {user}\n{common}"
            f"Host {SUBMIT}\n    HostName compute01\n    User {user}\n    ProxyJump {GW}\n{common}"
            f"# <<< probebench <<<\n")


def ensure_ssh_config(user, gateway):
    SSH_DIR.mkdir(mode=0o700, exist_ok=True)
    old = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""
    kept, skip = [], False
    for line in old.splitlines(keepends=True):
        if line.startswith("# >>> probebench >>>"):
            skip = True
        if not skip:
            kept.append(line)
        if line.startswith("# <<< probebench <<<"):
            skip = False
    text = "".join(kept)
    if text and not text.endswith("\n"):
        text += "\n"
    SSH_CONFIG.write_text(text + ssh_block(user, gateway))
    os.chmod(SSH_CONFIG, 0o600)


def find_key():
    for k in ("id_ed25519", "id_ecdsa", "id_rsa"):
        if (SSH_DIR / k).exists() and (SSH_DIR / f"{k}.pub").exists():
            return SSH_DIR / k
    return None


def free_port(start):
    for p in range(start, start + 40):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free local port")


def http_get(url, headers=None, timeout=4):
    req = urllib.request.Request(url, headers=headers or {})
    with NO_PROXY.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Problem(Exception):
    """An error with a message written for the person using the app."""


# --------------------------------------------------------------------------- sessions

class Session:
    def __init__(self, sid, jid, size=None, idle=None):
        self.sid, self.jid, self.size, self.idle = sid, jid, size, idle
        self.state = "waiting"      # waiting starting connecting ready ending ended error
        self.message = "Waiting for a GPU"
        self.detail = ""
        self.host = self.port = None
        self.lport = None
        self.token = None
        self.url = None
        self.mode = None
        self.tunnel = None
        self.created = now()
        self.ready_at = None
        self.gpu = None
        self.model = None
        self.stop = threading.Event()
        self.opened = False
        self.phase = "submit"
        self.phase_started = {"submit": now()}
        self.queue = None          # {"waiting": n, "running": n} from HTCondor
        self.fit = None            # {"free": n, "total": n} GPUs that fit this size

    def set_phase(self, phase):
        if self.phase != phase:
            self.phase = phase
            self.phase_started[phase] = now()

    def view(self):
        return {"sid": self.sid, "jid": self.jid, "size": self.size, "state": self.state,
                "message": self.message, "detail": self.detail, "host": (self.host or "").split(".")[0] or None,
                "gpu": self.gpu, "model": self.model, "url": self.url if self.state == "ready" else None,
                "created": self.created, "ready_at": self.ready_at, "idle": self.idle,
                "phase": self.phase, "phase_started": self.phase_started, "queue": self.queue, "fit": self.fit,
                "now": now()}


# --------------------------------------------------------------------------- the app

class App:
    def __init__(self, port, key):
        self.port, self.key = port, key
        self.askpass_key = secrets.token_hex(16)
        self.lock = threading.RLock()
        self.log = deque(maxlen=400)
        self.prompt = None
        self.ops = {}
        self.sessions = {}
        self.setup = {sid: {"status": "pending", "detail": "", "fraction": None, "started": None, "typical": None}
                      for sid, _ in SETUP_STEPS}
        self.cluster = None          # summary from labctl
        self.gpus = None
        self.downloads = {}
        self.signed_in = False
        self.signin_stderr = deque(maxlen=60)
        self.httpd = None
        self.quitting = False

    # ---- logging and state
    def note(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        with self.lock:
            self.log.append(line)
        print(line, file=sys.stderr, flush=True)

    def op_running(self, name):
        o = self.ops.get(name)
        return bool(o and o["status"] == "running")

    def run_op(self, name, fn, *args):
        with self.lock:
            if self.op_running(name):
                raise Problem("That's already running.")
            self.ops[name] = {"status": "running", "message": "", "error": None, "started": now()}
        op = self.ops[name]

        def go():
            try:
                fn(op, *args)
                op["status"] = "done"
            except Problem as e:
                op["status"], op["error"] = "error", str(e)
                self.note(f"{name}: {e}")
            except Exception as e:  # noqa: BLE001
                op["status"], op["error"] = "error", f"Something went wrong: {e}"
                self.note(f"{name} failed:\n{traceback.format_exc()}")
            finally:
                op["finished"] = now()
        threading.Thread(target=go, daemon=True).start()

    def state(self):
        conf = read_conf()
        with self.lock:
            p = self.prompt
            return {
                "version": VERSION,
                "user": conf["ISYE_USER"], "gateway": conf["GATEWAY"],
                "configured": bool(conf["ISYE_USER"]) and SSH_CONFIG.exists() and f"Host {GW}" in SSH_CONFIG.read_text(),
                "setup_done": self.setup_complete(),
                "signed_in": self.signed_in,
                "has_key": find_key() is not None,
                "prompt": None if p is None else {k: p[k] for k in ("id", "kind", "title", "text", "options")},
                "ops": {k: {kk: v.get(kk) for kk in ("status", "message", "error", "started")} for k, v in self.ops.items()},
                "setup": self.setup,
                "cluster": self.cluster,
                "gpus": self.gpus,
                "sizes": [{"id": k, **{kk: v[kk] for kk in ("title", "blurb", "gpu_mb", "mem_gb")}} for k, v in SIZES.items()],
                "suggested": SUGGESTED,
                "sessions": [s.view() for s in sorted(self.sessions.values(), key=lambda s: s.created, reverse=True)
                             if s.state != "ended" or now() - (s.ready_at or s.created) < 3600],
                "downloads": self.downloads,
                "log": list(self.log)[-80:],
                "account_form": ACCOUNT_FORM,
            }

    def setup_complete(self):
        marker = STATE_DIR / "setup-done"
        return marker.exists()

    # ---- ssh
    def ssh(self, host, cmd, stdin=None, timeout=90, check=True):
        args = ["ssh"]
        if stdin is None:
            args.append("-n")
        args += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=25", host, cmd]
        try:
            p = subprocess.run(args, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Problem("The cluster didn't answer in time. Check your internet connection and try again.")
        out, err = p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
        if check and p.returncode != 0:
            if p.returncode == 255:
                self.refresh_signin()
                if not self.signed_in:
                    raise Problem("You're signed out of the cluster. Sign in again to continue.")
                raise Problem("compute01 didn't accept the connection. " + self.ssh_error_hint(err))
            raise Problem((out + err).strip().splitlines()[-1] if (out + err).strip() else "A cluster command failed.")
        return out

    def ssh_error_hint(self, err):
        if "Permission denied" in err:
            return ("Your ISyE account may not have cluster access yet. Email Kelley Ross at "
                    "kelley.ross@isye.gatech.edu with your username to ask for it.")
        if "Could not resolve" in err or "timed out" in err.lower():
            return "Check your internet connection. If castle is down, switch the gateway to keep in Settings."
        return err.strip().splitlines()[-1] if err.strip() else ""

    def labctl(self, *args, stdin=None, timeout=90):
        out = self.ssh(SUBMIT, LABCTL + " " + " ".join(shlex.quote(str(a)) for a in args), stdin=stdin, timeout=timeout)
        first = out.strip().splitlines()[0] if out.strip() else ""
        if first.startswith("ERROR "):
            raise Problem(first[6:])
        return out

    def refresh_signin(self):
        try:
            rc = subprocess.run(["ssh", "-O", "check", GW], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10).returncode
        except subprocess.TimeoutExpired:
            rc = 1
        self.signed_in = rc == 0
        return self.signed_in

    # ---- sign-in with password + Duo through the page
    def do_signin(self, op):
        conf = read_conf()
        if not conf["ISYE_USER"]:
            raise Problem("Enter your Georgia Tech username first.")
        ensure_ssh_config(conf["ISYE_USER"], conf["GATEWAY"])
        if self.refresh_signin():
            op["message"] = "Already signed in."
            return
        op["message"] = f"Connecting to {conf['GATEWAY']}"
        self.note(f"Signing in to {conf['GATEWAY']} as {conf['ISYE_USER']}")
        env = dict(os.environ, SSH_ASKPASS=str(HERE / "askpass"), SSH_ASKPASS_REQUIRE="force",
                   DISPLAY=os.environ.get("DISPLAY", ":0"), PB_PORT=str(self.port), PB_ASKPASS_KEY=self.askpass_key)
        self.signin_stderr.clear()
        p = subprocess.Popen(["ssh", "-o", "ConnectTimeout=25", "-o", "NumberOfPasswordPrompts=3", GW, "true"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, start_new_session=True)

        def read_err():
            for raw in iter(p.stderr.readline, b""):
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self.signin_stderr.append(line)
        reader = threading.Thread(target=read_err, daemon=True)
        reader.start()
        try:
            rc = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            raise Problem("Signing in took too long. Try again.")
        finally:
            self.cancel_prompt()
        reader.join(timeout=2)
        err = "\n".join(self.signin_stderr)
        if rc != 0 or not self.refresh_signin():
            if "Permission denied" in err:
                raise Problem("The password or the Duo approval didn't go through. Check your Georgia Tech "
                              "password and try again.")
            if "Could not resolve" in err or "timed out" in err.lower() or "Connection refused" in err:
                raise Problem(f"Couldn't reach {conf['GATEWAY']}. Check your internet connection. "
                              "If it's down, switch to the keep gateway in Settings.")
            raise Problem("Sign-in didn't finish. " + (err.splitlines()[-1] if err else "Try again."))
        op["message"] = "Signed in."
        self.note("Signed in.")
        if self.setup_complete():
            self.after_signin()

    def after_signin(self):
        for fn in (self.refresh_cluster, self.refresh_gpus):
            try:
                fn()
            except Problem as e:
                self.note(f"Couldn't refresh: {e}")

    def askpass(self, prompt_text):
        """Called (via the askpass helper) when ssh needs a password, Duo choice or passphrase."""
        time.sleep(0.2)  # let ssh's explanatory stderr lines arrive first
        text = prompt_text.strip()
        context = "\n".join(self.signin_stderr)
        shown = re.sub(r"^\([^)]*@[^)]*\)\s*", "", text, flags=re.M).strip()
        options = [{"value": n, "label": lab.strip()} for n, lab in
                   re.findall(r"^\s*(\d+)\.\s+(.+?)\s*$", text + "\n" + context, re.M)]
        conf = read_conf()
        if re.search(r"passcode or option|duo", text + context, re.I) and not re.search(r"password:\s*$", text, re.I):
            kind, title = "duo", "Approve with Duo"
            seen, uniq = set(), []
            for o in options:
                if o["value"] not in seen:
                    seen.add(o["value"])
                    uniq.append(o)
            options = uniq
            shown = (f"Approve the sign-in for {conf['ISYE_USER']} with one of the Duo devices on that account, "
                     "or type a passcode from the Duo app.")
        elif re.search(r"passphrase", text, re.I):
            kind, title, options = "secret", "SSH key passphrase", []
            shown = "Your SSH key has a passphrase. Type it to unlock the key."
        elif re.search(r"password", text, re.I):
            kind, title, options = "secret", "Georgia Tech password", []
            shown = (f"For {conf['ISYE_USER']} on {conf['GATEWAY']}, ISyE's gateway. "
                     "It's the same password you use for other Georgia Tech logins.")
        elif re.search(r"yes/no", text, re.I):
            kind, title = "choice", "Confirm"
            options = [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]
        else:
            kind, title, options = "text", "The cluster is asking", []
        ev = threading.Event()
        p = {"id": secrets.token_hex(6), "kind": kind, "title": title, "text": shown, "options": options,
             "event": ev, "answer": None}
        with self.lock:
            self.prompt = p
        got = ev.wait(600)
        with self.lock:
            if self.prompt is p:
                self.prompt = None
        return p["answer"] if got else None

    def answer(self, pid, value, cancel=False):
        with self.lock:
            p = self.prompt
            if p is None or p["id"] != pid:
                raise Problem("That question has expired.")
            p["answer"] = None if cancel else str(value)
            self.prompt = None
            p["event"].set()

    def cancel_prompt(self):
        with self.lock:
            p, self.prompt = self.prompt, None
        if p:
            p["event"].set()

    # ---- setup
    def set_step(self, sid, status, detail="", fraction=None, typical=None):
        with self.lock:
            old = self.setup.get(sid) or {}
            started = old.get("started") if old.get("status") == "running" and status == "running" else None
            if status == "running" and started is None:
                started = now()
            self.setup[sid] = {"status": status, "detail": detail, "fraction": fraction, "started": started,
                               "typical": typical if typical is not None else old.get("typical")}

    def do_setup(self, op):
        for sid, _ in SETUP_STEPS:
            self.set_step(sid, "pending")
        try:
            (STATE_DIR / "setup-done").unlink()
        except FileNotFoundError:
            pass
        conf = read_conf()
        # 1. sign in
        self.set_step("signin", "running", f"Connecting to {conf['GATEWAY']}")
        try:
            self.do_signin(op)
        except Problem as e:
            self.set_step("signin", "error", str(e))
            raise
        self.set_step("signin", "done", f"Signed in as {conf['ISYE_USER']}.")
        # 2. ssh key
        self.set_step("key", "running", "Checking for a key on this Mac")
        key = find_key()
        if key is None:
            self.set_step("key", "running", "Creating a key in ~/.ssh")
            r = subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(SSH_DIR / "id_ed25519"),
                                "-C", f"{conf['ISYE_USER']} probebench"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0:
                self.set_step("key", "error", "Couldn't create an SSH key: " + r.stderr.decode().strip())
                raise Problem("Couldn't create an SSH key.")
            key = SSH_DIR / "id_ed25519"
        rc = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", SUBMIT, "true"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90)
        if rc.returncode != 0:
            self.set_step("key", "running", "Adding the key to your ISyE account")
            pub = (Path(str(key) + ".pub")).read_text().strip()
            cmd = ("umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
                   f"grep -qxF {shlex.quote(pub)} ~/.ssh/authorized_keys || echo {shlex.quote(pub)} >> ~/.ssh/authorized_keys")
            self.ssh(GW, cmd)
            rc = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", SUBMIT, "true"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90)
            if rc.returncode != 0:
                msg = ("compute01, the cluster's submit machine, didn't accept your login. Your ISyE account may "
                       "not have cluster access yet. Email Kelley Ross (kelley.ross@isye.gatech.edu) your username.")
                self.set_step("key", "error", msg)
                raise Problem(msg)
        self.set_step("key", "done", f"Using {key.name}. You can reach compute01 without typing your password.")
        # 3. copy
        self.set_step("copy", "running", "Copying to ~/probebench on the cluster")
        self.sync_cluster()
        self.set_step("copy", "done", "Copied to ~/probebench on the cluster.")
        # 4. container
        self.set_step("container", "running", "Checking the container")
        st = self.labctl("container-status").strip()
        if not st.startswith("sandbox"):
            reused = False
            if st == "missing" and self.labctl("link-existing").strip() == "linked":
                reused, st = True, "sif"
            jid = self.labctl("build").strip()
            building = not st.startswith("sif")
            typical = "15 to 30 minutes" if building else "5 to 15 minutes"
            lead = ("Reusing the container you built in ~/probe. " if reused else "")

            def report(msg):
                if isinstance(msg, dict):
                    self.set_step("container", "running", lead + msg["text"], msg.get("fraction"), typical)
                else:
                    self.set_step("container", "running", lead + msg[0].upper() + msg[1:], None, typical)
            self.set_step("container", "running", lead + "Submitting a CPU-only job", 0.0, typical)
            self.wait_job(jid, "build", report, building=building)
            if self.labctl("container-status").strip() == "missing":
                self.set_step("container", "error", "The build job finished without a container. Check the log.")
                raise Problem("The container build didn't produce a container.")
        self.set_step("container", "done", "PyTorch and Transformers are ready on the cluster.")
        (STATE_DIR / "setup-done").write_text(time.ctime())
        self.note("Setup finished.")
        self.after_signin()

    def sync_cluster(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            src = ROOT / "cluster"
            for path in sorted(src.rglob("*")):
                if "__pycache__" in path.parts:
                    continue
                tar.add(str(path), arcname=str(path.relative_to(src)), recursive=False)
        self.ssh(SUBMIT, f"mkdir -p {RLAB}/app && tar -C {RLAB}/app -xf - && chmod +x {RLAB}/app/*.sh",
                 stdin=buf.getvalue(), timeout=120)

    BUILD_STAGES = [(0.10, "Downloading the base image"), (0.55, "Installing PyTorch with CUDA support"),
                    (0.75, "Installing Hugging Face Transformers and other tools"), (0.85, "Finishing the install"),
                    (1.01, "Packing the container into one file")]

    def build_report(self, jid, building):
        stage, _, frac = self.labctl("build-progress", jid).strip().partition(" ")
        f = float(frac or 0)
        if stage == "build":
            label = next(lab for lim, lab in self.BUILD_STAGES if f < lim)
            return {"fraction": 0.8 * min(1.0, f / 0.85), "text": f"Building the container: {label.lower()}"}
        if stage == "unpack":
            base = 0.8 if building else 0.0
            span = 0.2 if building else 1.0
            return {"fraction": base + span * f * 0.97,
                    "text": "Unpacking the container into a folder so sessions start faster"}
        if stage == "check":
            return {"fraction": 0.98, "text": "Checking that PyTorch loads in the container"}
        return {"fraction": 0.0, "text": "The job has started"}

    def wait_job(self, jid, kind, report, building=False):
        if not jid.isdigit():
            raise Problem(f"Couldn't submit the {kind} job.")
        start = now()
        while not self.quitting:
            st = self.labctl("job-state", jid).strip()
            mins = int((now() - start) // 60)
            el = f"{mins} min" if mins else "under a minute"
            if st == "IDLE":
                report({"fraction": 0.0, "text": f"Waiting for the cluster to start job {jid} on a free CPU "
                        f"(waiting {el}). Jobs that don't need a GPU usually start within a minute or two."})
            elif st == "RUNNING":
                msg = f"running (job {jid}, {el} so far)"
                if kind == "build":
                    report(self.build_report(jid, building))
                    time.sleep(8)
                    continue
                if kind == "download":
                    prog = self.labctl("download-progress", jid).strip()
                    m = re.match(r"PROGRESS (\d+) (\d+)", prog)
                    if m and int(m.group(2)):
                        done_b, total_b = int(m.group(1)), int(m.group(2))
                        report({"fraction": done_b / total_b,
                                "text": f"{done_b / 2**30:.1f} of {total_b / 2**30:.1f} GB downloaded"})
                        time.sleep(8)
                        continue
                report(msg)
            elif st.startswith("HELD"):
                self.labctl("stop-job", jid)
                raise Problem(f"HTCondor held the {kind} job: {st[5:]}")
            elif st.startswith("DONE"):
                if kind == "download":
                    prog = self.labctl("download-progress", jid).strip()
                    if prog.startswith("ERROR"):
                        raise Problem(prog[6:])
                if st != "DONE 0":
                    log = self.labctl("job-log", jid, kind).strip().splitlines()[-6:]
                    raise Problem(f"The {kind} job failed. " + " ".join(log))
                return
            elif st == "REMOVED":
                raise Problem(f"The {kind} job was removed.")
            time.sleep(8)

    # ---- cluster info
    def refresh_cluster(self):
        out = self.labctl("summary")
        info = {"container": "missing", "token": False, "hf_home": "", "models": [], "sessions": []}
        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "container":
                info["container"] = parts[1] if len(parts) > 1 else "missing"
            elif parts[0] == "token":
                info["token"] = parts[1:2] == ["yes"]
            elif parts[0] == "hf_home":
                info["hf_home"] = " ".join(parts[1:])
            elif parts[0] == "model" and len(parts) >= 3:
                info["models"].append({"id": parts[1], "size": parts[2]})
            elif parts[0] == "session" and len(parts) >= 4:
                info["sessions"].append({"sid": parts[1], "jid": parts[2], "state": parts[3]})
        info["updated"] = now()
        with self.lock:
            self.cluster = info
            for s in info["sessions"]:  # sessions started elsewhere (or before the app restarted)
                if s["sid"] not in self.sessions:
                    sess = Session(s["sid"], s["jid"])
                    sess.state, sess.message = "found", "Running. Connect to use it."
                    sess.set_phase("found")
                    self.sessions[s["sid"]] = sess
        return info

    def refresh_gpus(self):
        out = self.ssh(SUBMIT, "condor_status -gpus -compact", timeout=60, check=False)
        kinds = []
        # Summary rows end in nine counts: Total Owner Unclaimed Claimed Preempting Matched Drain Backfill BkIdle.
        # Anchor on those, because GPU names contain numbers too (GeForce RTX 5090).
        for m in re.finditer(r"^\s*Capability ([\d.]+), ([\d.]+) GB, (.+?)((?:\s+\d+){9})\s*$", out, re.M):
            mem_gb, name = float(m.group(2)), m.group(3).strip()
            n = [int(x) for x in m.group(4).split()]
            free = n[2]
            kinds.append({"name": name, "mem_gb": mem_gb, "free": free, "in_use": n[1] + sum(n[3:])})
        merged = {}
        for k in kinds:
            key = (k["name"], round(k["mem_gb"]))
            if key in merged:
                merged[key]["free"] += k["free"]
                merged[key]["in_use"] += k["in_use"]
            else:
                merged[key] = dict(k)
        kinds = sorted(merged.values(), key=lambda k: -k["mem_gb"])
        per_size = {}
        for sid, s in SIZES.items():
            match = [k for k in kinds if k["mem_gb"] * 1024 >= s["gpu_mb"]]
            per_size[sid] = {"free": sum(k["free"] for k in match), "total": sum(k["free"] + k["in_use"] for k in match)}
        with self.lock:
            self.gpus = {"kinds": kinds, "per_size": per_size, "updated": now()} if kinds else None

    def do_refresh(self, op):
        if not self.refresh_signin():
            op["message"] = "Signed out."
            return
        self.refresh_cluster()
        try:
            self.refresh_gpus()
        except Problem:
            pass

    # ---- sessions
    def do_start(self, op, size, idle):
        s = SIZES.get(size)
        if s is None:
            raise Problem("Pick a GPU size.")
        if not self.refresh_signin():
            raise Problem("Sign in first.")
        op["message"] = "Copying the latest Probe Bench to the cluster"
        self.sync_cluster()
        op["message"] = "Asking HTCondor for a GPU"
        out = self.labctl("new-session", s["gpu_mb"], s["mem_gb"], idle).strip().split()
        if len(out) != 2:
            raise Problem("Couldn't submit the session job.")
        sess = Session(out[0], out[1], size, idle)
        with self.lock:
            self.sessions[sess.sid] = sess
        sess.set_phase("queue")
        if self.gpus and size in self.gpus["per_size"]:
            sess.fit = self.gpus["per_size"][size]
        self.note(f"Session {sess.sid} submitted as job {sess.jid} ({s['title']}).")
        threading.Thread(target=self.run_session, args=(sess, True), daemon=True).start()

    def connect_existing(self, sid):
        with self.lock:
            sess = self.sessions.get(sid)
        if sess is None:
            raise Problem("That session is gone.")
        if sess.state not in ("found", "error", "disconnected"):
            raise Problem("That session is already connected.")
        sess.state, sess.message = "starting", "Connecting"
        threading.Thread(target=self.run_session, args=(sess, True), daemon=True).start()

    def session_state(self, sess):
        return self.labctl("session-state", sess.sid).strip()

    def run_session(self, sess, open_browser):
        try:
            self.wait_ready(sess)
            if sess.stop.is_set():
                return
            sess.set_phase("connect")
            sess.state, sess.message = "connecting", f"Connecting your browser to {sess.host.split('.')[0]}"
            sess.token = self.labctl("token", sess.sid).strip()
            self.open_tunnel(sess)
            sess.state, sess.message, sess.ready_at = "ready", "Running", now()
            sess.set_phase("ready")
            self.note(f"Session {sess.sid} ready on {sess.host}.")
            self.fetch_session_info(sess)
            if open_browser and not sess.opened:
                sess.opened = True
                self.open_url(sess.url)
            self.monitor(sess)
        except Problem as e:
            if sess.state not in ("ended", "ending"):
                sess.state, sess.message = "error", str(e)
            self.note(f"Session {sess.sid}: {e}")
        except Exception as e:  # noqa: BLE001
            sess.state, sess.message = "error", f"Something went wrong: {e}"
            self.note(traceback.format_exc())
        finally:
            self.close_tunnel(sess)

    def wait_ready(self, sess):
        hinted = False
        while not sess.stop.is_set():
            st = self.session_state(sess)
            if st == "WAITING":
                waited = now() - sess.phase_started.get("queue", now())
                sess.state = "waiting"
                sess.set_phase("queue")
                sess.message = "Waiting for a free GPU"
                if now() - getattr(sess, "_queue_checked", 0) > 30:
                    sess._queue_checked = now()
                    try:
                        q = self.labctl("queue-info", sess.sid).split()
                        sess.queue = {"waiting": int(q[1]), "running": int(q[3])}
                        self.refresh_gpus()
                        if self.gpus and sess.size in self.gpus["per_size"]:
                            sess.fit = self.gpus["per_size"][sess.size]
                    except (Problem, ValueError, IndexError):
                        pass
                if waited > 120 and not hinted:
                    hinted = True
                    sess.detail = self.labctl("why", sess.sid).strip()
            elif st.startswith("STARTING"):
                sess.state = "starting"
                sess.set_phase("start")
                host = st[9:].strip()
                sess.message = f"Got a GPU{(' on ' + host.split('.')[0]) if host else ''}. Starting the server"
                sess.detail = ""
            elif st.startswith("READY"):
                _, sess.host, sess.port, sess.jid = st.split()[:4]
                return
            elif st.startswith("HELD"):
                self.labctl("stop", sess.sid)
                raise Problem(f"HTCondor held the job: {st[5:]}")
            elif st.startswith("ENDED"):
                sess.state = "ended"
                log = self.labctl("log", sess.sid).strip().splitlines()[-4:]
                sess.message = "The session stopped before it was ready."
                sess.detail = "\n".join(log)
                raise Problem(sess.message)
            sess.stop.wait(4)

    def start_tunnel(self, sess):
        log = open(STATE_DIR / "logs" / f"tunnel-{sess.sid}.log", "w")
        if sess.mode == "direct":
            args = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                    "-L", f"127.0.0.1:{sess.lport}:{sess.host}:{sess.port}", SUBMIT]
        else:
            rport = random.randint(20000, 50000)
            args = ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                    "-L", f"127.0.0.1:{sess.lport}:127.0.0.1:{rport}", SUBMIT,
                    f"condor_ssh_to_job -ssh 'ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:{rport}:127.0.0.1:{sess.port}' {sess.jid}"]
        sess.tunnel = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)

    def tunnel_ok(self, sess, tries=25):
        for _ in range(tries):
            try:
                j = http_get(f"http://127.0.0.1:{sess.lport}/health", timeout=3)
                if j.get("session") == sess.sid:
                    return True
            except Exception:  # noqa: BLE001
                pass
            if sess.tunnel.poll() is not None:
                return False
            time.sleep(1)
        return False

    def open_tunnel(self, sess):
        sess.lport = free_port(7860)
        ok = self.labctl("check-port", sess.host, sess.port).strip() == "OK"
        for mode in (["direct", "condor"] if ok else ["condor"]):
            sess.mode = mode
            self.start_tunnel(sess)
            if self.tunnel_ok(sess):
                sess.url = f"http://127.0.0.1:{sess.lport}/?token={sess.token}"
                self.note(f"Tunnel for {sess.sid} is up ({mode}).")
                return
            self.close_tunnel(sess)
        raise Problem("Couldn't connect your browser to the session. See the activity log for details.")

    def close_tunnel(self, sess):
        t, sess.tunnel = sess.tunnel, None
        if t and t.poll() is None:
            t.terminate()
            try:
                t.wait(5)
            except subprocess.TimeoutExpired:
                t.kill()

    def fetch_session_info(self, sess):
        try:
            st = http_get(f"http://127.0.0.1:{sess.lport}/api/status", {"X-Probe-Token": sess.token})
            sess.gpu = st.get("gpu", {}).get("name")
            m = st.get("model")
            sess.model = m.get("id") if m else None
        except Exception:  # noqa: BLE001
            pass

    def monitor(self, sess):
        n = 0
        while not sess.stop.wait(5):
            n += 1
            if sess.tunnel is None or sess.tunnel.poll() is not None:
                if not self.refresh_signin():
                    sess.state, sess.message = "disconnected", "Signed out of the cluster. Sign in to reconnect."
                    while not sess.stop.wait(5) and not self.signed_in:
                        pass
                    if sess.stop.is_set():
                        return
                try:
                    st = self.session_state(sess)
                except Problem:
                    continue
                if st.startswith("READY"):
                    self.note(f"Reconnecting {sess.sid}.")
                    self.start_tunnel(sess)
                    if self.tunnel_ok(sess):
                        sess.state, sess.message = "ready", "Running"
                    continue
                self.mark_ended(sess, st)
                return
            if n % 2 == 0:
                self.fetch_session_info(sess)      # GPU and loaded model, through the tunnel
            if n % 6 == 0:
                try:
                    st = self.session_state(sess)
                except Problem:
                    continue
                if not st.startswith("READY"):
                    self.mark_ended(sess, st)
                    return

    def mark_ended(self, sess, st):
        reason = "it ended"
        m = re.search(r'"reason":\s*"([^"]+)"', st)
        if m:
            reason = {"idle": "it was idle", "ended from the page": "you ended it from the page",
                      "stopped": "it was stopped"}.get(m.group(1), m.group(1))
        sess.state, sess.message = "ended", f"Ended because {reason}. The GPU is free again."
        self.note(f"Session {sess.sid} ended ({reason}).")

    def do_end(self, op, sid):
        with self.lock:
            sess = self.sessions.get(sid)
        if sess is None:
            raise Problem("That session is gone.")
        sess.state, sess.message = "ending", "Ending the session"
        sess.stop.set()
        self.close_tunnel(sess)
        self.labctl("stop", sid)
        sess.state, sess.message = "ended", "Ended. The GPU is free again."
        self.note(f"Session {sid} ended.")

    def open_url(self, url):
        opener = "open" if sys.platform == "darwin" or os.environ.get("PROBEBENCH_OPEN") else "xdg-open"
        try:
            subprocess.Popen([os.environ.get("PROBEBENCH_OPEN", opener), url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass

    # ---- downloads and token
    def do_download(self, op, model):
        if not REPO_RE.match(model):
            raise Problem("Use the model's Hugging Face name, like google/gemma-4-12B.")
        with self.lock:
            self.downloads[model] = {"status": "running", "text": "Starting a CPU-only download job", "fraction": None}
        d = self.downloads[model]
        try:
            jid = self.labctl("download", model).strip()

            def report(msg):
                if isinstance(msg, dict):
                    d.update(msg)
                else:
                    d["text"] = msg[0].upper() + msg[1:]
            self.wait_job(jid, "download", report)
            d.update(status="done", text="Downloaded.", fraction=1.0)
            self.note(f"Downloaded {model}.")
        except Problem as e:
            d.update(status="error", text=str(e))
            raise
        finally:
            try:
                self.refresh_cluster()
            except Problem:
                pass

    def save_token(self, token):
        token = token.strip()
        if not re.match(r"^hf_[A-Za-z0-9]+$", token):
            raise Problem("Hugging Face tokens start with hf_. Copy it again from huggingface.co/settings/tokens.")
        out = self.labctl("save-token", stdin=(token + "\n").encode()).strip()
        if out != "saved":
            raise Problem(out or "Couldn't save the token.")
        self.refresh_cluster()

    # ---- quitting
    def live_sessions(self):
        with self.lock:
            return [s for s in self.sessions.values() if s.state not in ("ended", "error")]

    def quit_app(self, end_sessions=True):
        self.quitting = True
        if end_sessions:
            for s in self.live_sessions():
                try:
                    s.stop.set()
                    self.close_tunnel(s)
                    self.labctl("stop", s.sid, timeout=30)
                except Problem:
                    pass
        self.cancel_prompt()
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()


# --------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "ProbeBenchMac/2"

    def log_message(self, fmt, *args):
        pass

    @property
    def app(self) -> App:
        return self.server.app

    def send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def host_ok(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.app.port}", f"localhost:{self.app.port}")

    def authed(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "pbmac" and hmac.compare_digest(v, self.app.key):
                return True
        return hmac.compare_digest(self.headers.get("X-PB-Key", ""), self.app.key)

    def body(self):
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise Problem("Requests must be JSON.")
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        self.safe("GET")

    def do_POST(self):
        self.safe("POST")

    def safe(self, method):
        try:
            if not self.host_ok():
                return self.send(403, {"error": "Wrong host."})
            self.route(method)
        except Problem as e:
            self.send(400, {"error": str(e)})
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            self.app.note(traceback.format_exc())
            self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def route(self, method):
        url = urlparse(self.path)
        path, app = url.path, self.app
        if path == "/health":
            return self.send(200, {"ok": True, "app": "probebench", "version": VERSION})
        if path == "/askpass" and method == "POST":
            if not hmac.compare_digest(self.headers.get("X-PB-Askpass", ""), app.askpass_key):
                return self.send(403, {"error": "no"})
            ans = app.askpass(self.body().get("prompt", ""))
            if ans is None:
                return self.send(410, b"", "text/plain")
            return self.send(200, ans.encode(), "text/plain")
        if path == "/" and method == "GET":
            key = (parse_qs(url.query).get("key") or [""])[0]
            if key and hmac.compare_digest(key, app.key):
                return self.send(303, b"", "text/plain", {"Location": "/",
                                 "Set-Cookie": f"pbmac={app.key}; Path=/; HttpOnly; SameSite=Strict"})
            if not self.authed():
                return self.send(401, "<!doctype html><meta charset=utf-8><title>Probe Bench</title>"
                                 "<p style='font-family:system-ui;margin:3rem'>Open Probe Bench from your "
                                 "Applications folder.</p>", "text/html")
            return self.send(200, (HERE / "control.html").read_text(), "text/html",
                             {"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                              "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"})
        if not self.authed():
            return self.send(401, {"error": "Open Probe Bench from your Applications folder."})
        if method == "GET":
            if path == "/api/state":
                return self.send(200, app.state())
            if path == "/api/sessions-live":
                return self.send(200, {"count": len(app.live_sessions())})
            return self.send(404, {"error": "Not found"})
        b = self.body()
        if path == "/api/account":
            user = re.sub(r"\s+", "", str(b.get("user", ""))).lower()
            if not re.match(r"^[a-z][a-z0-9]*$", user):
                raise Problem("That doesn't look like a Georgia Tech username. It's the part of your GT email "
                              "before @gatech.edu, like gburdell3.")
            gw = GATEWAYS.get(b.get("gateway", "castle"), GATEWAYS["castle"])
            old = read_conf()
            if old["ISYE_USER"] and old["ISYE_USER"] != user:
                subprocess.run(["ssh", "-O", "exit", GW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            write_conf({"ISYE_USER": user, "GATEWAY": gw})
            ensure_ssh_config(user, gw)
            app.note(f"Account set to {user} via {gw}.")
            return self.send(200, {"ok": True})
        if path == "/api/account/reset":   # use a different Georgia Tech account on this Mac
            if app.live_sessions():
                raise Problem("End your running sessions first. They belong to the account that's signed in now.")
            subprocess.run(["ssh", "-O", "exit", GW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            old = read_conf()
            write_conf({"ISYE_USER": "", "GATEWAY": old["GATEWAY"]})
            try:
                (STATE_DIR / "setup-done").unlink()
            except FileNotFoundError:
                pass
            with app.lock:
                app.cluster = app.gpus = None
                app.sessions = {k: v for k, v in app.sessions.items() if v.state not in ("ended", "error", "found")}
                app.downloads = {}
                app.ops = {k: v for k, v in app.ops.items() if v["status"] == "running"}
                for sid, _ in SETUP_STEPS:
                    app.setup[sid] = {"status": "pending", "detail": "", "fraction": None, "started": None, "typical": None}
            app.refresh_signin()
            app.note(f"Signed out {old['ISYE_USER'] or 'the previous account'}; ready for a different account.")
            return self.send(200, {"ok": True})
        if path == "/api/setup":
            app.run_op("setup", app.do_setup)
            return self.send(200, {"ok": True})
        if path == "/api/signin":
            app.run_op("signin", app.do_signin)
            return self.send(200, {"ok": True})
        if path == "/api/signout":
            subprocess.run(["ssh", "-O", "exit", GW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            app.refresh_signin()
            app.note("Signed out.")
            return self.send(200, {"ok": True})
        if path == "/api/answer":
            app.answer(str(b.get("id", "")), b.get("answer", ""), bool(b.get("cancel")))
            return self.send(200, {"ok": True})
        if path == "/api/refresh":
            if not app.op_running("refresh"):
                app.run_op("refresh", app.do_refresh)
            return self.send(200, {"ok": True})
        if path == "/api/start":
            idle = int(b.get("idle", 30))
            if idle not in (15, 30, 60, 120, 240):
                idle = 30
            app.run_op("start", app.do_start, str(b.get("size", "standard")), idle)
            return self.send(200, {"ok": True})
        if path == "/api/connect":
            app.connect_existing(str(b.get("sid", "")))
            return self.send(200, {"ok": True})
        if path == "/api/end":
            app.run_op("end-" + str(b.get("sid", "")), app.do_end, str(b.get("sid", "")))
            return self.send(200, {"ok": True})
        if path == "/api/open":
            sess = app.sessions.get(str(b.get("sid", "")))
            if not sess or not sess.url:
                raise Problem("That session isn't ready.")
            app.open_url(sess.url)
            return self.send(200, {"ok": True})
        if path == "/api/download":
            model = str(b.get("model", "")).strip()
            app.run_op("download-" + model, app.do_download, model)
            return self.send(200, {"ok": True})
        if path == "/api/token":
            app.save_token(str(b.get("token", "")))
            return self.send(200, {"ok": True})
        if path == "/api/quit":
            self.send(200, {"ok": True})
            app.quit_app(end_sessions=bool(b.get("end_sessions", True)))
            return
        return self.send(404, {"error": "Not found"})


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "logs").mkdir(exist_ok=True)
    key_file = STATE_DIR / "app.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(24))
        os.chmod(key_file, 0o600)
    key = key_file.read_text().strip()
    port = free_port(int(os.environ.get("PROBEBENCH_PORT", "7850")))
    app = App(port, key)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    httpd.app = app
    app.httpd = httpd
    (STATE_DIR / "app.port").write_text(str(port))
    signal.signal(signal.SIGTERM, lambda *_: app.quit_app(end_sessions=True))
    app.note(f"Probe Bench {VERSION} control page on http://127.0.0.1:{port}")
    app.refresh_signin()
    if read_conf()["ISYE_USER"] and app.signed_in:
        app.run_op("refresh", app.do_refresh)
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        for s in list(app.sessions.values()):
            app.close_tunnel(s)
        try:
            (STATE_DIR / "app.port").unlink()
        except FileNotFoundError:
            pass
        app.note("Probe Bench control page stopped.")


if __name__ == "__main__":
    main()
