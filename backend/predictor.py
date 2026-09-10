"""
predictor.py — Punto de entrada de inferencia (CNN MobileNet).
"""

from __future__ import annotations

from cnn_predictor import available, predict_cnn


def predict_fruit(pil_image, top_k: int = 3) -> dict:
    if not available():
        raise RuntimeError(
            "Modelo CNN no encontrado. Coloca backend/models/fruit_cnn.pt"
        )
    return predict_cnn(pil_image, top_k=top_k)
