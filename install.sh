#!/bin/bash
# Probe Bench installer for macOS. Run it by pasting this into Terminal:
#
#   curl -fsSL https://raw.githubusercontent.com/mathrulestheworld/probebench/main/install.sh | bash
#
# It installs Probe Bench into ~/Library/Application Support/ProbeBench, builds the
# Probe Bench app in your Applications folder (on your Mac, so macOS doesn't block it),
# adds a `probebench` Terminal command, and opens the app. Run it again to update.
set -euo pipefail

REPO="${PROBEBENCH_REPO:-mathrulestheworld/probebench}"
REF="${PROBEBENCH_REF:-main}"
TARBALL="${PROBEBENCH_TARBALL:-https://github.com/$REPO/archive/refs/heads/$REF.tar.gz}"
APPDIR="$HOME/Library/Application Support/ProbeBench"
PLISTBUDDY="${PLISTBUDDY:-/usr/libexec/PlistBuddy}"

if [ -t 1 ]; then B=$'\033[1m'; R=$'\033[0m'; G=$'\033[32m'; RED=$'\033[31m'; else B=""; R=""; G=""; RED=""; fi
step() { printf '%s==>%s %s%s%s\n' "$G" "$R" "$B" "$*" "$R"; }
say() { printf '    %s\n' "$*"; }
die() { printf '%sError:%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || die "This installer is for macOS. On Linux, use the probebench command from the repository."

step "Checking this Mac"
if ! xcode-select -p >/dev/null 2>&1; then
  say "Probe Bench needs Apple's Command Line Tools, a free download from Apple that includes Python."
  xcode-select --install >/dev/null 2>&1 || true
  die "A window should have opened asking to install them. Click Install, wait for it to finish (a few minutes), then paste the install command again."
fi
/usr/bin/python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null ||
  die "Python 3.9 or newer isn't working on this Mac. Run 'xcode-select --install', then try again."
say "Command Line Tools and Python are installed."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Run from an unzipped copy (bash path/to/install.sh) to install that copy instead of downloading.
LOCAL=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/app/pbapp.py" ]; then
  LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -n "$LOCAL" ] && [ -z "${PROBEBENCH_TARBALL:-}" ]; then
  step "Installing from $LOCAL"
  cp -R "$LOCAL/app" "$LOCAL/cluster" "$LOCAL/probebench" "$tmp/"
  [ -f "$LOCAL/README.md" ] && cp "$LOCAL/README.md" "$tmp/"
else
  step "Downloading Probe Bench"
  curl -fsSL "$TARBALL" | tar -xz -C "$tmp" --strip-components 1 ||
    die "Couldn't download Probe Bench from $TARBALL. Check your internet connection."
fi
[ -f "$tmp/app/pbapp.py" ] && [ -f "$tmp/cluster/labctl.sh" ] || die "The download is incomplete. Try again."

step "Installing"
if [ -x "$APPDIR/app/start.sh" ]; then
  "$APPDIR/app/start.sh" --restart || true    # stop the old control page; running sessions keep going
fi
mkdir -p "$APPDIR/logs"
rm -rf "$APPDIR/app.new" "$APPDIR/cluster.new"
cp -R "$tmp/app" "$APPDIR/app.new"
cp -R "$tmp/cluster" "$APPDIR/cluster.new"
rm -rf "$APPDIR/app" "$APPDIR/cluster"
mv "$APPDIR/app.new" "$APPDIR/app"
mv "$APPDIR/cluster.new" "$APPDIR/cluster"
cp "$tmp/probebench" "$APPDIR/probebench"
[ -f "$tmp/README.md" ] && cp "$tmp/README.md" "$APPDIR/README.md"
chmod +x "$APPDIR/probebench" "$APPDIR/app/start.sh" "$APPDIR/app/askpass" "$APPDIR/cluster/"*.sh
xattr -dr com.apple.quarantine "$APPDIR" 2>/dev/null || true
say "Files are in $APPDIR"

step "Building the Probe Bench app"
if [ -w /Applications ]; then DEST=/Applications; else DEST="$HOME/Applications"; mkdir -p "$DEST"; fi
APP="$DEST/Probe Bench.app"
if pgrep -f "$APP/Contents/MacOS" >/dev/null 2>&1; then
  osascript -e 'tell application id "local.probebench.app" to quit' >/dev/null 2>&1 || true
  sleep 1
fi
rm -rf "$APP"
osacompile -s -o "$APP" "$APPDIR/app/ProbeBench.applescript" || die "Couldn't build the app with osacompile."
cp "$APPDIR/app/icon.icns" "$APP/Contents/Resources/applet.icns"
PLIST="$APP/Contents/Info.plist"
"$PLISTBUDDY" -c "Set :CFBundleIdentifier local.probebench.app" "$PLIST" 2>/dev/null ||
  "$PLISTBUDDY" -c "Add :CFBundleIdentifier string local.probebench.app" "$PLIST"
"$PLISTBUDDY" -c "Set :CFBundleName Probe Bench" "$PLIST" 2>/dev/null ||
  "$PLISTBUDDY" -c "Add :CFBundleName string Probe Bench" "$PLIST"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
touch "$APP"
say "Installed $APP"

step "Adding the probebench command for Terminal"
rc="$HOME/.zshrc"; [ "${SHELL##*/}" = bash ] && rc="$HOME/.bash_profile"
touch "$rc"
grep -v "alias probebench=" "$rc" > "$tmp/rc" || true
printf "alias probebench='bash \"%s/probebench\"'\n" "$APPDIR" >> "$tmp/rc"
cat "$tmp/rc" > "$rc"
say "Added to $rc (new Terminal windows pick it up)."

step "Opening Probe Bench"
if [ -z "${PROBEBENCH_NO_OPEN:-}" ]; then open "$APP"; fi
printf '\n%sProbe Bench is installed.%s It opens in your browser. Next time, open it from %s.\n' "$B" "$R" "$DEST"
