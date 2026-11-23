"""FastAPI backend for the medicine recognition mobile app.

There is no web interface: the app uploads an image, shows the suggestion to
the user, and posts the confirmation or a manual name back.

Run:
    uvicorn src.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import Config
from src.decision.confirmation import ConfirmationService, UnknownRequestError
from src.pipeline import MedicineRecognizer
from src.preprocessing import image_processor as ip

MAX_RAW_UPLOAD_BYTES = 25 * 1024 * 1024


class ConfirmRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    accepted: bool
    medicine_name: str | None = Field(default=None, max_length=120)


class ManualInputRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    medicine_name: str = Field(min_length=1, max_length=120)


def create_app(config: Config | None = None, recognizer: MedicineRecognizer | None = None) -> FastAPI:
    cfg = config or Config()
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rec = recognizer or MedicineRecognizer(cfg)
        rec.load()  # OCR model + dataset loaded exactly once
        app.state.recognizer = rec
        app.state.confirmations = ConfirmationService()
        yield

    app = FastAPI(
        title=cfg.service_title,
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    def _log(entry: dict) -> None:
        if not cfg.log_requests:
            return
        cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.outputs_dir / "requests.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @app.get("/api/health")
    def health():
        rec = getattr(app.state, "recognizer", None)
        return {
            "status": "ok",
            "ocr_loaded": bool(rec and getattr(rec.extractor, "loaded", False)),
            "labels_in_dataset": len(rec.dataset) if rec else 0,
            "ocr_model_load_ms": round(rec.model_load_ms, 1) if rec and rec.model_load_ms else None,
            "threshold": cfg.match_threshold,
        }

    @app.post("/api/recognize")
    async def recognize(file: UploadFile = File(...)):
        started = time.perf_counter()
        try:
            ip.validate_extension(file.filename, cfg.allowed_extensions)
        except ValueError as error:
            raise HTTPException(status_code=415, detail=str(error))

        payload = await file.read(MAX_RAW_UPLOAD_BYTES + 1)
        try:
            ip.validate_size(payload, cfg.max_upload_mb)
        except ValueError as error:
            raise HTTPException(status_code=413, detail=str(error))

        rec = app.state.recognizer
        try:
            with inference_lock:  # Paddle inference is not re-entrant
                # The pipeline decodes and validates the image so that image
                # loading is measured as its own stage.
                result = rec.recognize(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except Exception as error:  # OCR failure is a server-side problem
            raise HTTPException(status_code=500, detail=f"Recognition failed: {error}") from error

        request_id = uuid.uuid4().hex[:12]
        result["request_id"] = request_id
        app.state.confirmations.remember(request_id, result)

        _log(
            {
                "request_id": request_id,
                "image_name": file.filename,
                "status": result["status"],
                "suggested_medicine": result["suggested_medicine"],
                "match_score": result["match_score"],
                "detected_texts": result["detected_texts"],
                "timings_ms": result["timings_ms"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        return result

    @app.post("/api/confirm")
    def confirm(body: ConfirmRequest):
        try:
            return app.state.confirmations.confirm(body.request_id, body.accepted, body.medicine_name)
        except UnknownRequestError:
            raise HTTPException(status_code=404, detail="Unknown request_id; run /api/recognize first")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

    @app.post("/api/manual-input")
    def manual_input(body: ManualInputRequest):
        try:
            return app.state.confirmations.manual_input(body.request_id, body.medicine_name)
        except UnknownRequestError:
            raise HTTPException(status_code=404, detail="Unknown request_id; run /api/recognize first")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

    return app


app = create_app()
