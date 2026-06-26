#!/usr/bin/env bash
# Build a standalone macOS bundle of the Synapse backend via PyInstaller.
# Output: dist/synapse-backend/  (onedir, ~80-150 MB depending on deps)
#
# The bundle is meant to be embedded inside the Compose Desktop .app and
# launched as a LaunchAgent on tester machines.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "[error] No .venv found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if ! command -v pyinstaller >/dev/null; then
    echo "[install] pyinstaller..."
    pip install --upgrade pip >/dev/null
    pip install pyinstaller
fi

echo "[clean] removing previous build/dist..."
rm -rf dist build synapse-backend.spec

echo "[build] PyInstaller onedir..."
pyinstaller \
    --noconfirm \
    --name synapse-backend \
    --onedir \
    --console \
    --collect-all fastembed \
    --collect-all apsw \
    --collect-all sqlite_vec \
    --collect-all huggingface_hub \
    --collect-all tokenizers \
    --collect-all onnxruntime \
    --collect-all zeroconf \
    --collect-all ifaddr \
    --collect-all networkx \
    --collect-all dateparser \
    --copy-metadata fastembed \
    --copy-metadata anthropic \
    --hidden-import sqlite_vec \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    backend_entry.py

# Stamp the bundle with a version marker. The desktop app compares this against the
# installed backend at launch and auto-reinstalls when they differ, so a new .dmg's
# backend reaches testers even if their old one is still running under KeepAlive
# (SYN-105). ~/.synapse data is untouched by reinstall. Timestamped so every rebuild
# (even same commit, e.g. a venv-sync fix) is treated as a new version.
VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)-$(date +%Y%m%d%H%M%S)"
echo "$VERSION" > dist/synapse-backend/BACKEND_VERSION
echo "[stamp] BACKEND_VERSION=$VERSION"

echo
echo "[done] Bundle at: $(pwd)/dist/synapse-backend/"
echo "       Test it with: ./dist/synapse-backend/synapse-backend"
