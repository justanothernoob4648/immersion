from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod
from typing import Any, Dict, List, Optional, TypedDict, Union

import uvicorn
from bot import run_bot
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

load_dotenv(override=True)

app = FastAPI()
app.mount("/prebuilt", SmallWebRTCPrebuiltUI)
small_webrtc_handler: SmallWebRTCRequestHandler = SmallWebRTCRequestHandler()
active_sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/prebuilt/")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest):
    async def on_connection(connection):
        asyncio.create_task(run_bot(connection))

    return await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    logger.debug(f"Received patch request: {request}")
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.post("/start")
async def rtvi_start(request: Request):
    class IceServer(TypedDict, total=False):
        urls: Union[str, List[str]]

    class IceConfig(TypedDict):
        iceServers: List[IceServer]

    class StartBotResult(TypedDict, total=False):
        sessionId: str
        iceConfig: Optional[IceConfig]

    try:
        payload = await request.json()
        logger.debug(f"Received request: {payload}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to parse request body: {exc}")
        payload = {}

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = payload

    result: StartBotResult = {"sessionId": session_id}
    if payload.get("enableDefaultIceServers"):
        result["iceConfig"] = IceConfig(
            iceServers=[IceServer(urls=["stun:stun.l.google.com:19302"])]
        )

    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_request(
    session_id: str,
    path: str,
    request: Request,
):
    active_session = active_sessions.get(session_id)
    if active_session is None:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    if path.endswith("api/offer"):
        try:
            request_data = await request.json()
            if request.method == HTTPMethod.POST.value:
                webrtc_request = SmallWebRTCRequest(
                    sdp=request_data["sdp"],
                    type=request_data["type"],
                    pc_id=request_data.get("pc_id"),
                    restart_pc=request_data.get("restart_pc"),
                    request_data=request_data,
                )
                return await offer(webrtc_request)
            if request.method == HTTPMethod.PATCH.value:
                patch_request = SmallWebRTCPatchRequest(
                    pc_id=request_data["pc_id"],
                    candidates=[IceCandidate(**c) for c in request_data.get("candidates", [])],
                )
                return await ice_candidate(patch_request)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to parse WebRTC request: {exc}")
            return Response(content="Invalid WebRTC request", status_code=400)

    logger.info(f"Received request for path: {path}")
    return Response(status_code=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmallWebRTC demo server")
    parser.add_argument("--host", default="localhost", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    logger.add(sys.stderr, level="TRACE" if args.verbose else "INFO")

    uvicorn.run(app, host=args.host, port=args.port)
