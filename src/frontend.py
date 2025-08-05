"""Minimal frontend to display backend state."""

from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


app = FastAPI()

BACKEND_URL = "http://localhost:8000/inference/report"


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Return a simple page that shows the backend report."""
    return (
        "<html><body><h1>Backend report</h1>"
        "<pre id='data'>Loading...</pre>"
        "<script>"
        "async function fetchData(){"
        "const res=await fetch('/api/report');"
        "const data=await res.json();"
        "document.getElementById('data').textContent=JSON.stringify(data,null,2);"
        "}"
        "fetchData();setInterval(fetchData,1000);"
        "</script></body></html>"
    )


@app.get("/api/report")
async def proxy_report() -> dict:
    """Proxy request to the backend service."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(BACKEND_URL)
        resp.raise_for_status()
        return resp.json()

