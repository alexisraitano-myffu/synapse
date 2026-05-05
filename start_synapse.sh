#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/.venv/bin/activate"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

PORT="${API_PORT:-8765}"
TOKEN="${API_TOKEN:-NOT SET — configure .env}"

echo "==============================="
echo "  Synapse — démarrage"
echo "==============================="
echo ""

echo "[1/3] Lancement du serveur API (port $PORT)..."
cd "$SCRIPT_DIR"

# Libérer le port si un ancien process l'occupe
lsof -ti tcp:"$PORT" | xargs kill -9 2>/dev/null || true

uvicorn api.server:app --host 0.0.0.0 --port "$PORT" \
    --log-level warning >> /tmp/synapse_api.log 2>&1 &
API_PID=$!

# Attendre la vraie réponse HTTP (max 15s)
echo "      Attente du serveur..."
SERVER_READY=0
for i in $(seq 1 15); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "      ✅ Serveur prêt (PID $API_PID)"
        SERVER_READY=1
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "ERREUR : le serveur a crashé. Log : /tmp/synapse_api.log"
        cat /tmp/synapse_api.log
        exit 1
    fi
    sleep 1
done
if [ "$SERVER_READY" -eq 0 ]; then
    echo "ERREUR : le serveur ne répond pas après 15s."
    cat /tmp/synapse_api.log
    kill "$API_PID" 2>/dev/null
    exit 1
fi

echo "[2/3] Lancement du Dream Cycle (toutes les heures)..."
(
    while true; do
        python "$SCRIPT_DIR/run_cycle.py" 2>&1 | tail -3
        sleep 3600
    done
) &
CYCLE_PID=$!
echo "      ✅ Dream Cycle actif (PID $CYCLE_PID)"

cleanup() {
    echo ""
    echo "Arrêt en cours..."
    kill "$API_PID" "$CYCLE_PID" 2>/dev/null || true
    wait "$API_PID" "$CYCLE_PID" 2>/dev/null || true
    echo "Terminé."
}
trap cleanup EXIT INT TERM

echo ""
echo "==============================="
echo "🔑 Token : $TOKEN"
echo "==============================="
echo ""
echo "[3/3] Lancement du tunnel Cloudflare..."
echo "      L'URL publique s'affiche ci-dessous."
echo ""

bash "$SCRIPT_DIR/setup_tunnel.sh"
