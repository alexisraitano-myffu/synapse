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

echo
echo "[done] Bundle at: $(pwd)/dist/synapse-backend/"
echo "       Test it with: ./dist/synapse-backend/synapse-backend"
