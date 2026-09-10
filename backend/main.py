"""
main.py — API REST FruitGuard v2.1
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from predictor import predict_fruit

app = FastAPI(
    title="FruitGuard API",
    description="Clasificador de frutas y frescura en tiempo real.",
    version="2.1.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    return RedirectResponse(url="/app/")


@app.get("/api")
async def api_info():
    return {
        "service": "FruitGuard API",
        "version": "2.1.2",
        "frontend": "/app/",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg", "image/x-png"}
    ctype = (file.content_type or "").lower()
    if ctype and ctype not in allowed and not ctype.startswith("image/"):
        raise HTTPException(415, detail=f"Tipo no soportado: {file.content_type}")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        raise HTTPException(400, detail="No se pudo leer la imagen.")

    try:
        result = predict_fruit(pil_image, top_k=3)
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Error en predicción: {e}")

    return result


if FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
