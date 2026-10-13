#!/bin/bash
# Build the runnable tree: ANEMLL Forge at the pinned commit, this repo's patches applied in order, this repo's new
# files copied in, and the Python environment set up.
#   scripts/prepare.sh                 # into ./forge (default)
#   FORGE_DIR=~/src/forge scripts/prepare.sh
# Environment: FORGE_DIR (./forge), FORGE_REPO, FORGE_COMMIT (see scripts/config.sh), PYTHON (python3.11), NO_VENV=1
# skips the venv. Running it again on an existing FORGE_DIR is safe: it checks out the pinned commit, resets
# tracked files, and applies the patches again.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=config.sh
source "$HERE/scripts/config.sh"
FORGE_DIR=${FORGE_DIR:-$HERE/forge}
PYTHON=${PYTHON:-python3.11}

die() { echo "prepare: $*" >&2; exit 1; }
command -v git >/dev/null || die "git is not installed (xcode-select --install)"

if [ ! -d "$FORGE_DIR/.git" ]; then
  echo "prepare: cloning $FORGE_REPO into $FORGE_DIR"
  git clone --quiet "$FORGE_REPO" "$FORGE_DIR"
fi
cd "$FORGE_DIR"
git fetch --quiet origin "$FORGE_COMMIT" 2>/dev/null || git fetch --quiet origin
git cat-file -e "$FORGE_COMMIT^{commit}" 2>/dev/null || die "commit $FORGE_COMMIT not found in $FORGE_REPO"
# a clean tree at the pin: tracked files reset, untracked files from an earlier run removed (the venv and the
# built Swift bridge are kept)
git checkout --quiet --detach "$FORGE_COMMIT"
git reset --quiet --hard "$FORGE_COMMIT"
git clean -q -fd -e .venv -e .venv-convert -e coreai/swift_bridge/libcoreai_bridge.dylib

echo "prepare: ANEMLL Forge at $(git rev-parse --short HEAD)"
for p in "$HERE"/patches/*.patch; do
  git apply --whitespace=nowarn "$p" || die "patch $(basename "$p") does not apply to $FORGE_COMMIT"
  echo "prepare: applied $(basename "$p")"
done
# new files (converter, parity tool, M6 benchmark tools, tests)
cp -R "$HERE/overlay/." "$FORGE_DIR/"
echo "prepare: copied $(cd "$HERE/overlay" && find . -type f | wc -l | tr -d ' ') new files"

if [ "${NO_VENV:-0}" != 1 ]; then
  if [ ! -x .venv/bin/python ]; then
    if command -v uv >/dev/null; then uv venv -q -p 3.11 .venv
    else "$PYTHON" -m venv .venv; fi
  fi
  if command -v uv >/dev/null; then uv pip install -q -p .venv/bin/python -r requirements-inference.txt
  else .venv/bin/pip install -q -r requirements-inference.txt; fi
  echo "prepare: Python environment in $FORGE_DIR/.venv"
fi
[ -f coreai/swift_bridge/libcoreai_bridge.dylib ] || \
  echo "prepare: next, build the Swift bridge: (cd $FORGE_DIR && bash coreai/swift_bridge/build.sh)  # Xcode 27 SDK"
echo "prepare: done. Forge tree: $FORGE_DIR"
