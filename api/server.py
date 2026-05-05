import json
import os
import sys
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from db import cursor_to_dicts, first_row, get_connection, init_db

init_db()

API_TOKEN = os.getenv("API_TOKEN", "")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8765"))
PWA_DIR = Path(__file__).parent.parent / "pwa"

app = FastAPI(title="Synapse Mobile API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN not configured in .env")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


class InboxBody(BaseModel):
    content: str
    type: str = "text"
    source: str = "pwa"


class ValidateBody(BaseModel):
    confirmed: bool
    correction: str | None = None


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    conn = get_connection()
    try:
        inbox_count = conn.execute(
            "SELECT COUNT(*) FROM inbox WHERE processed_at IS NULL"
        ).fetchone()[0]
        entities_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        return {"status": "ok", "inbox_count": inbox_count, "entities_count": entities_count}
    finally:
        conn.close()


@app.post("/inbox")
def add_text(body: InboxBody, _=Depends(verify_token)):
    content = body.content
    if body.type != "text":
        content = f"[{body.type}] {content}"

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO inbox (content, source) VALUES (?, ?)",
                (content, body.source),
            )
        return {"id": str(conn.last_insert_rowid()), "status": "queued"}
    finally:
        conn.close()


@app.post("/inbox/audio")
async def add_audio(file: UploadFile = File(...), _=Depends(verify_token)):
    audio_bytes = await file.read()
    content = f"[audio — {len(audio_bytes)} bytes, en attente de transcription]"

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO inbox (content, source, raw_file) VALUES (?, ?, ?)",
                (content, "mobile-audio", audio_bytes),
            )
        return {"id": str(conn.last_insert_rowid()), "status": "queued"}
    finally:
        conn.close()


@app.post("/inbox/image")
async def add_image(file: UploadFile = File(...), _=Depends(verify_token)):
    image_bytes = await file.read()
    content = f"[image — {len(image_bytes)} bytes, en attente d'analyse]"

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO inbox (content, source, raw_file) VALUES (?, ?, ?)",
                (content, "mobile-image", image_bytes),
            )
        return {"id": str(conn.last_insert_rowid()), "status": "queued"}
    finally:
        conn.close()


@app.get("/pending")
def list_pending(_=Depends(verify_token)):
    conn = get_connection()
    try:
        results = []
        for item in cursor_to_dicts(conn.execute(
            "SELECT id, fact_data, created_at FROM pending_facts ORDER BY created_at DESC"
        )):
            try:
                fact_data = json.loads(item["fact_data"])
            except (ValueError, TypeError):
                fact_data = item["fact_data"]

            summary = ""
            if isinstance(fact_data, dict):
                entity = fact_data.get("entity_canonical", "?")
                predicate = fact_data.get("predicate", "?")
                value = fact_data.get("value", "?")
                summary = f"{entity} — {predicate}: {value}"

            results.append({
                "id": item["id"],
                "summary": summary,
                "fact_data": fact_data,
                "created_at": item["created_at"],
            })
        return results
    finally:
        conn.close()


@app.post("/validate/{fact_id}")
def validate_fact(fact_id: str, body: ValidateBody, _=Depends(verify_token)):
    conn = get_connection()
    try:
        pending = first_row(conn.execute(
            "SELECT id, fact_data FROM pending_facts WHERE id=?", (fact_id,)
        ))
        if not pending:
            raise HTTPException(status_code=404, detail=f"fact_id '{fact_id}' not found")

        try:
            fact_data = json.loads(pending["fact_data"])
        except (ValueError, TypeError):
            raise HTTPException(status_code=500, detail="invalid fact_data in DB")

        with conn:
            if not body.confirmed:
                conn.execute("DELETE FROM pending_facts WHERE id=?", (fact_id,))
                return {"status": "rejected"}

            if body.correction:
                fact_data["value"] = body.correction

            entity_name = fact_data.get("entity_canonical", "unknown")
            row = conn.execute(
                "SELECT id FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (entity_name,)
            ).fetchone()

            if row:
                entity_id = row[0]
            else:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO entities (id, canonical_name) VALUES (?,?)",
                    (entity_id, entity_name),
                )

            conn.execute(
                "INSERT INTO facts "
                "(id, entity_id, predicate, value, confidence, source_inbox_id, persistence_value) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()), entity_id,
                    fact_data.get("predicate"), fact_data.get("value"),
                    0.95,
                    fact_data.get("source_inbox_id"),
                    fact_data.get("persistence_value", 3),
                ),
            )
            conn.execute("DELETE FROM pending_facts WHERE id=?", (fact_id,))

        return {"status": "validated"}
    finally:
        conn.close()


# ── PWA static files (must be mounted last) ───────────────────────────────────
app.mount("/", StaticFiles(directory=str(PWA_DIR), html=True), name="pwa")


if __name__ == "__main__":
    uvicorn.run("server:app", host=API_HOST, port=API_PORT, app_dir=str(Path(__file__).parent))
