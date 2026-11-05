#!/bin/bash
# CPU-only HTCondor job: downloads a model from Hugging Face into the shared model cache.
# Arguments: <lab folder> <org/model> <HF_HOME>
LAB=$1; MODEL=$2
export HF_HOME=$3 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export APPTAINER_TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
IMG="$LAB/lab.sif"
[ -f "$LAB/lab-sandbox/.probebench-ok" ] && IMG="$LAB/lab-sandbox"
exec apptainer exec "$IMG" python3 "$LAB/app/server/download.py" "$MODEL"
