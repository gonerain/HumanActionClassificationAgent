from __future__ import annotations

from pathlib import Path
import html

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

_BACKEND_FILE = Path(__file__).resolve().parent / "backend_service.py"

@app.get("/", response_class=HTMLResponse)
async def show_backend_source() -> str:
    """Return a simple page showing ``backend_service.py``."""
    code = _BACKEND_FILE.read_text(encoding="utf-8")
    escaped = html.escape(code)
    return (
        "<html><body><h1>backend_service.py</h1><pre>"
        f"{escaped}"
        "</pre></body></html>"
    )
