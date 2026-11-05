#!/bin/bash
# CPU-only HTCondor job: builds the container (if missing), then unpacks it into a
# folder so GPU sessions don't have to unpack 4 GB every time they start.
set -uo pipefail
LAB=$1
SCR="${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}"
export APPTAINER_CACHEDIR="$SCR/cache" APPTAINER_TMPDIR="$SCR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

rm_tree() { if [ -e "$1" ]; then chmod -R u+w "$1" 2>/dev/null; rm -rf "$1"; fi; }

if [ ! -e "$LAB/lab.sif" ]; then
  echo "PHASE build"
  echo "Building the container. This takes 10 to 20 minutes."
  rm -f "$LAB/lab-new.sif"
  apptainer build --force "$LAB/lab-new.sif" "$LAB/app/lab.def" || { echo "BUILD_FAILED"; exit 1; }
  mv -f "$LAB/lab-new.sif" "$LAB/lab.sif"
fi

if [ -f "$LAB/lab-sandbox/.probebench-ok" ]; then
  echo "The container is already unpacked."
  echo "BUILD_OK"; exit 0
fi
NEW="$LAB/lab-sandbox-new.$$"          # unique per job, so two jobs can't clash
SIF_BYTES=$(stat -L -c %s "$LAB/lab.sif" 2>/dev/null || echo 4000000000)
echo "PHASE unpack $(( SIF_BYTES * 12 / 5 ))"   # unpacked size is roughly 2.4 times the image
echo "Unpacking the container so sessions start faster."
if apptainer build --fix-perms --sandbox "$NEW" "$LAB/lab.sif" &&
   echo "PHASE check" &&
   apptainer exec "$NEW" python3 -c "import torch, transformers" ; then
  touch "$NEW/.probebench-ok"
  if [ -f "$LAB/lab-sandbox/.probebench-ok" ]; then
    rm_tree "$NEW"                     # another job finished first
  else
    rm_tree "$LAB/lab-sandbox"
    mv "$NEW" "$LAB/lab-sandbox"
  fi
  echo "Unpacked to $LAB/lab-sandbox"
else
  rm_tree "$NEW"
  echo "Couldn't unpack the container. Sessions will use lab.sif directly, which starts a little slower."
fi
echo "BUILD_OK"
