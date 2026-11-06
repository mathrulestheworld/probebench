#!/bin/bash
# Probe Bench cluster helper. Runs on compute01; the probebench launcher on your
# Mac calls it over ssh. Everything lives in ~/probebench on the cluster:
#   app/        these scripts and the web server (copied from your Mac on every start)
#   lab.sif     the Apptainer container (lab-sandbox/ is an unpacked copy for faster starts)
#   sessions/   one folder per GPU session: access token, logs, server address
#   jobs/       logs of container builds and model downloads
#   exports/    activations saved from the web page
#   avoid       machines a job got stuck on (each job is kept off them for 7 days)
# Optional overrides go in ~/probebench/config, e.g.  HF_HOME_DIR=/some/other/hf_cache
# or, to keep every job off some machines for good:  AVOID_MACHINES="isye-yancey isye-hpc1231-4"
set -uo pipefail

LAB="${PROBEBENCH_LAB:-$HOME/probebench}"
APP="$LAB/app"
HF_HOME_DIR="$HOME/hf_cache"
[ -f "$LAB/config" ] && . "$LAB/config"
IMG="$LAB/lab.sif"
SANDBOX="$LAB/lab-sandbox"
SESS="$LAB/sessions"
JOBS="$LAB/jobs"
mkdir -p "$SESS" "$JOBS" "$LAB/exports" "$HF_HOME_DIR"

err() { echo "ERROR $*"; exit 1; }
AVOID_FILE="$LAB/avoid"
AVOID_DAYS=7

avoid_list() {  # full names of machines to keep jobs off: AVOID_MACHINES from the config, plus recent entries in $AVOID_FILE
  local cutoff h t
  cutoff=$(( $(date +%s) - AVOID_DAYS * 86400 ))
  {
    for h in ${AVOID_MACHINES:-}; do echo "$h"; done
    if [ -f "$AVOID_FILE" ]; then
      while read -r h t; do [ -n "$h" ] && [ "${t:-0}" -ge "$cutoff" ] 2>/dev/null && echo "$h"; done < "$AVOID_FILE"
    fi
  } | while read -r h; do
    [[ "$h" =~ ^[A-Za-z0-9.-]+$ ]] || continue
    [[ "$h" == *.* ]] || h="$h.isye.gatech.edu"
    echo "$h"
  done | sort -u
}

requirements_line() {  # an HTCondor requirements line keeping the job off those machines (nothing if there are none)
  local r="" h
  while read -r h; do [ -n "$h" ] && r="${r:+$r && }(Machine =!= \"$h\")"; done < <(avoid_list)
  if [ -n "$r" ]; then echo "requirements = $r"; fi
}

job_log() {  # the HTCondor event log of a job Probe Bench submitted
  local f d
  for f in "$JOBS/build.$1.log" "$JOBS/download.$1.log"; do [ -f "$f" ] && { echo "$f"; return; }; done
  for d in "$SESS"/*/; do [ "$(cat "$d/jobid" 2>/dev/null)" = "$1" ] && { echo "${d}job.log"; return; }; done
}
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

submit() {  # submit a file, print the cluster id or an ERROR line
  local out
  if ! out=$(condor_submit -terse "$1" 2>&1); then
    err "condor_submit failed: $(echo "$out" | tr '\n' ' ')"
  fi
  echo "$out" | grep -Eo '^[0-9]+' | head -1
}

job_state() {  # IDLE | RUNNING | HELD <reason> | DONE <exit code> | REMOVED | GONE
  local jid=$1 line code
  line=$(condor_q "$jid" -af JobStatus HoldReason 2>/dev/null | head -1)
  if [ -n "$line" ]; then
    code=${line%% *}
    case "$code" in
      1) echo IDLE ;;
      2|6) echo RUNNING ;;
      5) echo "HELD ${line#* }" ;;
      3) echo REMOVED ;;
      4) echo "DONE 0" ;;
      7) echo IDLE ;;
      *) echo "OTHER $line" ;;
    esac
    return
  fi
  line=$(condor_history "$jid" -limit 1 -af JobStatus ExitCode 2>/dev/null | head -1)
  if [ -z "$line" ]; then echo GONE; return; fi
  set -- $line
  if [ "$1" = 3 ]; then echo REMOVED; else echo "DONE ${2:-?}"; fi
}

session_jid() { cat "$SESS/$1/jobid" 2>/dev/null || err "No session called $1."; }

cmd=${1:-help}
shift || true
case "$cmd" in

  container-status)
    if [ -f "$SANDBOX/.probebench-ok" ]; then echo "sandbox $SANDBOX"
    elif [ -e "$IMG" ]; then echo "sif $(readlink -f "$IMG")"
    else echo missing; fi ;;

  link-existing)  # reuse a container you already built by hand in ~/probe
    if [ ! -e "$IMG" ] && [ -f "$HOME/probe/probe.sif" ]; then ln -s "$HOME/probe/probe.sif" "$IMG" && echo linked
    else echo none; fi ;;

  build)  # build the container if needed, then unpack it (CPU-only job)
    # If a build is already queued or running (say, from an interrupted setup), reuse it.
    running=$(condor_q -constraint 'ProbeBench =?= "build" && JobStatus < 3' -af ClusterId 2>/dev/null | head -1)
    if [ -n "$running" ]; then echo "$running"; exit 0; fi
    cat > "$JOBS/build.sub" <<EOF
universe = vanilla
executable = $APP/run_build.sh
arguments = $LAB
transfer_executable = false
initialdir = $JOBS
output = build.\$(Cluster).out
error = build.\$(Cluster).err
log = build.\$(Cluster).log
request_cpus = 4
request_memory = 16GB
request_disk = 30GB
batch_name = probebench-build
+ProbeBench = "build"
$(requirements_line)
queue
EOF
    submit "$JOBS/build.sub" ;;

  download)  # download a model in a CPU-only job
    model=${1:-}
    [[ "$model" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || err "Use the Hugging Face name of the model, like google/gemma-4-12B-it."
    cat > "$JOBS/download.sub" <<EOF
universe = vanilla
executable = $APP/run_download.sh
arguments = $LAB $model $HF_HOME_DIR
transfer_executable = false
initialdir = $JOBS
output = download.\$(Cluster).out
error = download.\$(Cluster).err
log = download.\$(Cluster).log
request_cpus = 2
request_memory = 8GB
request_disk = 12GB
batch_name = probebench-download
+ProbeBench = "download"
$(requirements_line)
queue
EOF
    submit "$JOBS/download.sub" ;;

  download-progress)  # last progress line of a download job
    grep -E '^(PROGRESS|DONE|ERROR)' "$JOBS/download.$1.out" 2>/dev/null | tail -1 ;;

  job-state) job_state "$1" ;;

  evictions)  # evictions <cluster id>: "<times evicted> <last machine>", from the job's own event log
    is_int "${1:-}" || err "evictions needs a job number."
    logf=$(job_log "$1")
    [ -n "$logf" ] && [ -f "$logf" ] || { echo "0 -"; exit 0; }
    awk -v j="$1" '
      /^[0-9][0-9][0-9] \(/ { split($2, a, "."); cur = substr(a[1], 2) }
      cur == j && /Job was evicted/ { n++ }
      cur == j && match($0, /slot[0-9_]+@[A-Za-z0-9.-]+/) { h = substr($0, RSTART, RLENGTH); sub(/.*@/, "", h) }
      cur == j && h == "" && match($0, /alias=[A-Za-z0-9.-]+/) { h = substr($0, RSTART + 6, RLENGTH - 6) }
      END { print n + 0, (h == "" ? "-" : h) }' "$logf" ;;

  resubmit)  # resubmit <cluster id> <machine>: remove a job that keeps getting evicted there, and submit it again elsewhere
    jid=${1:-}; host=${2:-}
    is_int "$jid" || err "resubmit needs a job number."
    [[ "$host" =~ ^[A-Za-z0-9.-]+$ ]] || err "resubmit needs a machine name."
    logf=$(job_log "$jid")
    case "$logf" in
      "$JOBS"/build.*) sub="$JOBS/build.sub" ;;
      "$JOBS"/download.*) sub="$JOBS/download.sub" ;;
      */job.log) sub="$(dirname "$logf")/session.sub" ;;
      *) err "Couldn't find how job $jid was submitted." ;;
    esac
    [ -f "$sub" ] || err "Couldn't find $sub."
    condor_rm "$jid" >/dev/null 2>&1
    echo "$host $(date +%s)" >> "$AVOID_FILE"
    req=$(requirements_line)
    awk -v r="$req" '/^requirements = /{ next } /^queue$/ && r != "" { print r } { print }' "$sub" > "$sub.tmp" && mv "$sub.tmp" "$sub"
    new=$(submit "$sub")
    case "$new" in ERROR*|"") echo "${new:-ERROR condor_submit printed nothing}"; exit 1 ;; esac
    case "$sub" in */session.sub) echo "$new" > "$(dirname "$sub")/jobid" ;; esac
    echo "$new" ;;

  avoid)  # machines jobs are kept off: "<machine> config" or "<machine> learned <unix time>"
    for h in ${AVOID_MACHINES:-}; do echo "$h config"; done
    if [ -f "$AVOID_FILE" ]; then
      cutoff=$(( $(date +%s) - AVOID_DAYS * 86400 ))
      sort -k2,2nr "$AVOID_FILE" | awk -v c="$cutoff" '$2 >= c && !seen[$1]++ { print $1, "learned", $2 }'
    fi ;;

  clear-avoid)  # forget the machines Probe Bench learned to avoid (not the ones in the config)
    rm -f "$AVOID_FILE"; echo cleared ;;

  build-progress)  # <stage> <fraction 0-1> : how far a build job has got
    out="$JOBS/build.$1.out"; errf="$JOBS/build.$1.err"
    phase=$(grep -E '^PHASE ' "$out" 2>/dev/null | tail -1)
    case "$phase" in
      "PHASE unpack"*)
        expected=${phase##* }
        size=$(du -sb "$LAB"/lab-sandbox-new.* 2>/dev/null | awk '{s+=$1} END{print s+0}')
        awk -v s="$size" -v e="$expected" 'BEGIN{f = (e > 0) ? s / e : 0; if (f > 0.97) f = 0.97; printf "unpack %.3f\n", f}' ;;
      "PHASE check"*) echo "check 0.98" ;;
      "PHASE build"*)
        log=$(cat "$out" "$errf" 2>/dev/null)
        f=0.02
        echo "$log" | grep -qE "Copying blob|Getting image source" && f=0.05
        echo "$log" | grep -qE "Running post scriptlet|^\+ pip" && f=0.10
        echo "$log" | grep -qE "Downloading (https://download.pytorch.org|torch-)" && f=0.15
        echo "$log" | grep -qE "Downloading nvidia" && f=0.30
        echo "$log" | grep -qE "Collecting transformers" && f=0.55
        echo "$log" | grep -qE "Installing collected packages" && f=0.65
        echo "$log" | grep -qE "Successfully installed" && f=0.75
        echo "$log" | grep -qE "Creating SIF file" && f=0.85
        echo "build $f" ;;
      *) echo "start 0" ;;
    esac ;;

  queue-info)  # why a session is waiting: <other GPU jobs waiting> <GPU jobs running>
    jid=$(session_jid "$1")
    q=$(condor_q "$jid" -af QDate 2>/dev/null | head -1); is_int "$q" || q=0
    waiting=$(condor_q -allusers -constraint "JobStatus == 1 && RequestGPUs > 0 && ClusterId != $jid" -af ClusterId 2>/dev/null | wc -l)
    running=$(condor_q -allusers -constraint "JobStatus == 2 && RequestGPUs > 0" -af ClusterId 2>/dev/null | wc -l)
    echo "waiting $waiting running $running" ;;

  job-log)  # job-log <cluster id> <build|download>
    for f in "$JOBS/$2.$1.out" "$JOBS/$2.$1.err"; do
      [ -s "$f" ] && { echo "--- $(basename "$f")"; tail -n 25 "$f"; }
    done ;;

  new-session)  # new-session <gpu MB> <RAM GB> <idle minutes>
    gpu_mb=${1:-30000}; mem_gb=${2:-32}; idle=${3:-30}
    is_int "$gpu_mb" && is_int "$mem_gb" && is_int "$idle" || err "new-session needs whole numbers."
    if [ -f "$SANDBOX/.probebench-ok" ]; then disk=4; else disk=12; fi
    sid="$(date +%Y%m%d-%H%M%S)-$RANDOM"
    d="$SESS/$sid"
    mkdir -p "$d" && chmod 700 "$d"
    (umask 077; head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$d/token")
    printf 'HF_HOME=%q\n' "$HF_HOME_DIR" > "$d/env"
    cat > "$d/session.sub" <<EOF
universe = vanilla
executable = $APP/run_session.sh
arguments = $LAB $d $idle $gpu_mb $mem_gb
transfer_executable = false
initialdir = $d
output = server.out
error = server.err
log = job.log
request_gpus = 1
require_gpus = GlobalMemoryMb >= $gpu_mb
request_cpus = 4
request_memory = ${mem_gb}GB
request_disk = ${disk}GB
batch_name = probebench
+ProbeBench = "session"
+ProbeBenchSession = "$sid"
$(requirements_line)
queue
EOF
    jid=$(submit "$d/session.sub") || { echo "$jid"; exit 1; }
    case "$jid" in ERROR*) echo "$jid"; exit 1 ;; esac
    echo "$jid" > "$d/jobid"
    echo "$sid $jid" ;;

  session-state)  # WAITING | STARTING <host> | READY <host> <port> <jid> | HELD <reason> | ENDED <why>
    sid=$1; d="$SESS/$sid"; jid=$(session_jid "$sid")
    st=$(job_state "$jid")
    host=$(condor_q "$jid" -af RemoteHost 2>/dev/null | head -1); host=${host#*@}
    case "$st" in
      RUNNING)
        if [ -f "$d/ended.json" ]; then echo "ENDED $(tr -d '\n' < "$d/ended.json")"
        elif [ -s "$d/server.txt" ]; then
          read -r shost port < "$d/server.txt"
          [ -n "$host" ] && [ "$host" != undefined ] || host=$shost
          echo "READY $host $port $jid"
        else echo "STARTING ${host:-}"; fi ;;
      IDLE) echo WAITING ;;
      HELD*) echo "$st" ;;
      *) if [ -f "$d/ended.json" ]; then echo "ENDED $(tr -d '\n' < "$d/ended.json")"; else echo "ENDED $st"; fi ;;
    esac ;;

  why)  # short summary of why a session job hasn't started
    jid=$(session_jid "$1")
    condor_q -better-analyze "$jid" 2>/dev/null | grep -E "machines|are able to run|rejected by|serving other users|reject your job" | sed 's/^ *//' ;;

  token) cat "$SESS/$1/token" ;;

  check-port)  # can compute01 reach the server directly?
    if timeout 3 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; then echo OK; else echo FAIL; fi ;;

  stop)
    jid=$(session_jid "$1"); condor_rm "$jid" >/dev/null 2>&1; echo stopped ;;

  stop-job)
    is_int "${1:-}" && condor_rm "$1" >/dev/null 2>&1; echo stopped ;;

  stop-all)
    condor_rm -constraint 'ProbeBench =?= "session"' 2>&1 | grep -v "^$" || true ;;

  list)  # running or waiting sessions: <sid> <jid> <state>
    condor_q -constraint 'ProbeBench =?= "session"' -af ProbeBenchSession ClusterId JobStatus 2>/dev/null |
      while read -r sid jid code; do
        case "$code" in 1) s=waiting ;; 2) s=running ;; 5) s=held ;; *) s="status-$code" ;; esac
        echo "$sid $jid $s"
      done ;;

  log)  # log [sid]: server log of a session (latest if not given)
    sid=${1:-$(ls -1t "$SESS" 2>/dev/null | head -1)}
    [ -n "$sid" ] || err "No sessions yet."
    for f in "$SESS/$sid/server.err" "$SESS/$sid/server.out"; do
      [ -s "$f" ] && { echo "--- $f"; tail -n 40 "$f"; }
    done ;;

  models)  # downloaded models: <name> <size>
    for d in "$HF_HOME_DIR"/hub/models--*; do
      [ -d "$d" ] || continue
      name=${d##*/models--}; name=${name/--//}
      echo "$name $(du -sh "$d" 2>/dev/null | cut -f1)"
    done ;;

  save-token)  # reads a Hugging Face token on stdin
    IFS= read -r tok || true
    [[ "$tok" =~ ^hf_[A-Za-z0-9]+$ ]] || err "That doesn't look like a Hugging Face token (they start with hf_)."
    (umask 077; printf '%s' "$tok" > "$HF_HOME_DIR/token")
    echo saved ;;

  summary)  # one-call machine-readable summary for the Mac app
    echo "container $(bash "$0" container-status | cut -d' ' -f1)"
    if [ -f "$HF_HOME_DIR/token" ]; then echo "token yes"; else echo "token no"; fi
    echo "hf_home $HF_HOME_DIR"
    bash "$0" models | sed 's/^/model /'
    bash "$0" list | sed 's/^/session /'
    bash "$0" avoid | sed 's/^/avoid /' ;;

  status)
    echo "Probe Bench folder:  $LAB"
    echo "Model cache:         $HF_HOME_DIR"
    case "$(bash "$0" container-status)" in
      sandbox*) echo "Container:           ready (unpacked)" ;;
      sif*) echo "Container:           ready (not unpacked; sessions start a little slower)" ;;
      *) echo "Container:           not built yet (run probebench setup)" ;;
    esac
    if [ -f "$HF_HOME_DIR/token" ]; then echo "Hugging Face token:  saved"; else echo "Hugging Face token:  not saved (only needed for gated models)"; fi
    echo
    echo "Downloaded models:"
    out=$(bash "$0" models); if [ -n "$out" ]; then echo "$out" | sed 's/^/  /'; else echo "  none"; fi
    echo
    echo "Sessions:"
    out=$(bash "$0" list); if [ -n "$out" ]; then echo "$out" | sed 's/^/  /'; else echo "  none running"; fi ;;

  *)
    echo "usage: labctl.sh <command> ...   (normally run by the probebench launcher)"; exit 2 ;;
esac
