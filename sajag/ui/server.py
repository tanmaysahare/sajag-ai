"""Local dashboard: FastAPI + WebSocket, bound to 127.0.0.1 only.

Nothing is served beyond the loopback interface and no third-party assets
are loaded, so the dashboard works offline and leaks nothing.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

STATIC = Path(__file__).resolve().parent / "static"


def create_app(guardian) -> FastAPI:
    from ..demo import SCENARIOS, run_scenario
    from ..npu import device_summary

    app = FastAPI(title="Sajag AI", docs_url=None, redoc_url=None)
    clients: set[WebSocket] = set()
    loop_holder: dict = {}

    def broadcast(ev: dict) -> None:
        loop = loop_holder.get("loop")
        if not loop:
            return
        msg = json.dumps(ev, ensure_ascii=False, default=str)
        for ws in list(clients):
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)

    guardian.subscribe(broadcast)

    @app.on_event("startup")
    async def _startup() -> None:
        loop_holder["loop"] = asyncio.get_running_loop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse({
            "state": guardian.state.as_dict(),
            "stats": guardian.stats,
            "device": device_summary(),
            "sessions": guardian.factory.report(),
            "events": list(guardian.events)[-60:],
            "scenarios": {k: v["title"] for k, v in SCENARIOS.items()},
        })

    @app.post("/api/demo/{name}")
    def demo(name: str) -> JSONResponse:
        if name not in SCENARIOS:
            return JSONResponse({"error": "unknown scenario"}, status_code=404)
        threading.Thread(target=run_scenario, args=(name, guardian), kwargs={"realtime": True}, daemon=True).start()
        return JSONResponse({"started": name})

    @app.post("/api/reset")
    def reset() -> JSONResponse:
        guardian.reset()
        return JSONResponse({"ok": True})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            clients.discard(websocket)

    return app


def serve(guardian, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(guardian), host=host, port=port, log_level="warning")
