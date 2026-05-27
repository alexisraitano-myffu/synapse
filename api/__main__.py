"""Run the Synapse API: `python -m api` (uvicorn on 0.0.0.0:8000)."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=os.environ.get("SYNAPSE_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("SYNAPSE_API_PORT", "8000")),
        log_level="info",
    )
