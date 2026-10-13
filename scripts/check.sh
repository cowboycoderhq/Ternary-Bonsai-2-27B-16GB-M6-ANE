#!/bin/bash
# Prepare the Forge tree (scripts/prepare.sh), then run the full test suite inside it.
#   scripts/check.sh          # FORGE_DIR and the other prepare.sh variables apply
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
FORGE_DIR=${FORGE_DIR:-$HERE/forge}
FORGE_DIR=$FORGE_DIR "$HERE/scripts/prepare.sh"
cd "$FORGE_DIR"
.venv/bin/python -m unittest discover -s tests
