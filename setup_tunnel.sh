#!/usr/bin/env bash
set -e

if ! command -v cloudflared &>/dev/null; then
    echo "cloudflared not found — downloading..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz"
    else
        URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
    fi
    curl -L "$URL" -o /tmp/cloudflared.tgz
    tar -xzf /tmp/cloudflared.tgz -C /tmp
    sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
    echo "cloudflared installed."
fi

PORT="${API_PORT:-8765}"
echo "Launching Cloudflare Tunnel → http://localhost:$PORT"
echo "Your public URL will appear below (format: https://xxxxx.trycloudflare.com)"
echo "For a persistent URL: cloudflared login"
echo ""
cloudflared tunnel --url "http://localhost:$PORT"
