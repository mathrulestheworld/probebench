#!/bin/bash
# Used by Probe Bench.app. Starts the control page server if it isn't running, then opens it.
#   start.sh            make sure it's running and open the page in your browser
#   start.sh --ensure   make sure it's running
#   start.sh --count    print how many GPU sessions are running
#   start.sh --quit     end sessions and stop the server
#   start.sh --restart  stop the server but leave sessions running (used when updating)
APPDIR="$HOME/Library/Application Support/ProbeBench"
PY="${PROBEBENCH_PYTHON:-/usr/bin/python3}"
port() { cat "$APPDIR/app.port" 2>/dev/null; }
key() { cat "$APPDIR/app.key" 2>/dev/null; }
healthy() {
  local p; p=$(port)
  [ -n "$p" ] && curl -fsS -m 2 --noproxy '*' "http://127.0.0.1:$p/health" 2>/dev/null | grep -q '"app": "probebench"'
}
post() { curl -fsS -m "$2" --noproxy '*' -H "X-PB-Key: $(key)" -H 'Content-Type: application/json' -d "$3" "http://127.0.0.1:$(port)$1"; }
ensure() {
  healthy && return 0
  mkdir -p "$APPDIR/logs"
  nohup "$PY" "$APPDIR/app/pbapp.py" >> "$APPDIR/logs/app.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do healthy && return 0; sleep 0.25; done
  echo "The control page didn't start. Details are in $APPDIR/logs/app.log" >&2
  return 1
}
case "${1:-}" in
  --ensure) ensure ;;
  --count)
    if healthy; then
      curl -fsS -m 5 --noproxy '*' -H "X-PB-Key: $(key)" "http://127.0.0.1:$(port)/api/sessions-live" | sed -E 's/.*"count": *([0-9]+).*/\1/'
    else echo 0; fi ;;
  --quit) healthy && { post /api/quit 90 '{"end_sessions": true}' >/dev/null; sleep 1; }; exit 0 ;;
  --restart) healthy && { post /api/quit 30 '{"end_sessions": false}' >/dev/null; sleep 1; }; exit 0 ;;
  *) ensure || exit 1; open "http://127.0.0.1:$(port)/?key=$(key)" ;;
esac
