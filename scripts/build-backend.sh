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

# Post-migration coeur Rust (SYN-110/111/112) : stockage, cerveau et sync
# vivent dans le module compilé synapse_core (wheel maturin du repo
# synapse-core, pas sur PyPI). fastembed/apsw/sqlite_vec/dateparser ont quitté
# requirements.txt. Les fichiers modèle ne sont PAS bundlés : le backend les
# télécharge au premier besoin (core_store._download_model, ~130 Mo one-time).
if ! python -c "import synapse_core" 2>/dev/null; then
    echo "[error] synapse_core absent de la venv. Builder la wheel :" >&2
    echo "        cd ../synapse-core && maturin build --release -m crates/synapse-core-py/Cargo.toml" >&2
    echo "        pip install ../synapse-core/target/wheels/synapse_core-*.whl" >&2
    exit 1
fi

echo "[build] PyInstaller onedir..."
pyinstaller \
    --noconfirm \
    --name synapse-backend \
    --onedir \
    --console \
    --collect-all synapse_core \
    --collect-all zeroconf \
    --collect-all ifaddr \
    --collect-all networkx \
    --copy-metadata anthropic \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    backend_entry.py

# Ship the core's prompts as data next to the binary: backend_entry.py mirrors
# them into SYNAPSE_HOME/prompts at startup (nothing else deploys them on a
# tester machine, and the shipped brain requires the matching prompt set).
PROMPTS_SRC="../synapse-core/prompts"
if [ ! -f "$PROMPTS_SRC/manifest.json" ]; then
    echo "[error] $PROMPTS_SRC introuvable — le bundle a besoin des prompts du repo synapse-core." >&2
    exit 1
fi
mkdir -p dist/synapse-backend/prompts
cp "$PROMPTS_SRC"/*.md "$PROMPTS_SRC"/manifest.json dist/synapse-backend/prompts/
echo "[prompts] $(ls dist/synapse-backend/prompts | wc -l | tr -d ' ') fichiers embarqués"

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
