#!/bin/bash
# Runs inside the HTCondor GPU job: starts the Probe Bench server in the container.
# Arguments: <lab folder> <session folder> <idle minutes> <GPU MB requested> <RAM GB requested>
LAB=$1; SDIR=$2; IDLE=$3; GPU_MB=$4; MEM_GB=$5

set -a; . "$SDIR/env"; set +a          # HF_HOME
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
# Where Apptainer unpacks lab.sif if there's no unpacked copy (inside the job's scratch space).
export APPTAINER_TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"

IMG="$LAB/lab.sif"
if [ -f "$LAB/lab-sandbox/.probebench-ok" ] && apptainer exec "$LAB/lab-sandbox" true 2>/dev/null; then
  IMG="$LAB/lab-sandbox"
fi
echo "$(date '+%F %T') starting on $(hostname) using $IMG" >&2
exec apptainer exec --nv "$IMG" python3 "$LAB/app/server/server.py" \
  --session-dir "$SDIR" --lab-dir "$LAB" --idle-minutes "$IDLE" \
  --gpu-request-mb "$GPU_MB" --mem-request-gb "$MEM_GB"
