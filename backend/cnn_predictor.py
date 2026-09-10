"""
cnn_predictor.py — MobileNetV3 + ensemble robusto (web + cámara).

Mejoras vs v2:
- EXIF / orientación
- Varias vistas (raw, letterbox, centro, iso, flip, multi-crop) → promedio
- Primero fruta (suma fresh+rotten), luego estado
- Penaliza clases raras (azufaifo, sandía…) que “roban” predicciones
- Priors de color / confusiones conocidas (mango↔naranja, manzana↔papa)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from preprocess import isolate_fruit_on_white

MODELS_DIR = Path(__file__).parent / "models"
ONNX = MODELS_DIR / "fruit_cnn.onnx"
META = MODELS_DIR / "fruit_cnn_meta.json"
CKPT = MODELS_DIR / "fruit_cnn.pt"  # legacy; prefer ONNX en prod

_session = None
_classes: list[str] = []
_img_size = 224
_class_index: dict[str, int] = {}
_input_name = "input"

# Clases con pocas muestras en Freshness44 → suelen disparar falsos positivos en fotos web
RARE_FRUITS = {
    "jujube": 0.35,
    "watermelon": 0.40,
    "papaya": 0.55,
    "pear": 0.70,  # rotten_pear muy escaso
    "kaki": 0.75,
    "bittergroud": 0.70,
}

FRUIT_ES = {
    "apple": "Manzana",
    "banana": "Banana",
    "bittergroud": "Cundeamor",
    "capsicum": "Pimiento",
    "carrot": "Zanahoria",
    "cucumber": "Pepino",
    "grape": "Uva",
    "guava": "Guayaba",
    "jujube": "Azufaifo",
    "kaki": "Caqui",
    "lime": "Limón",
    "mango": "Mango",
    "okra": "Okra",
    "orange": "Naranja",
    "papaya": "Papaya",
    "peach": "Durazno",
    "pear": "Pera",
    "pomegranate": "Granada",
    "potato": "Papa",
    "strawberry": "Fresa",
    "tomato": "Tomate",
    "watermelon": "Sandía",
}
FRUIT_EMOJI = {
    "apple": "🍎",
    "banana": "🍌",
    "bittergroud": "🥒",
    "capsicum": "🫑",
    "carrot": "🥕",
    "cucumber": "🥒",
    "grape": "🍇",
    "guava": "🍈",
    "jujube": "🍒",
    "kaki": "🟠",
    "lime": "🍋",
    "mango": "🥭",
    "okra": "🌿",
    "orange": "🍊",
    "papaya": "🍈",
    "peach": "🍑",
    "pear": "🍐",
    "pomegranate": "🍎",
    "potato": "🥔",
    "strawberry": "🍓",
    "tomato": "🍅",
    "watermelon": "🍉",
}


def available() -> bool:
    return ONNX.exists() or CKPT.exists()


def _load():
    global _session, _classes, _img_size, _class_index, _input_name
    if _session is not None:
        return
    if not ONNX.exists():
        raise RuntimeError(
            "Falta fruit_cnn.onnx. Exporta el modelo o vuelve a desplegar con el ONNX incluido."
        )
    if META.exists():
        import json

        meta = json.loads(META.read_text())
        _classes = list(meta["classes"])
        _img_size = int(meta.get("img_size", 224))
    else:
        raise RuntimeError("Falta fruit_cnn_meta.json con las clases del modelo.")
    _class_index = {c: i for i, c in enumerate(_classes)}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _session = ort.InferenceSession(str(ONNX), sess_options=so, providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name


def prepare_image(pil_image: Image.Image) -> Image.Image:
    """Corrige orientación EXIF (muy común en fotos de Google/móvil)."""
    try:
        return ImageOps.exif_transpose(pil_image.convert("RGB"))
    except Exception:
        return pil_image.convert("RGB")


def _preprocess(img: Image.Image) -> np.ndarray:
    """RGB PIL → NCHW float32 ImageNet-normalized."""
    size = _img_size
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float64)


def _forward(img: Image.Image) -> np.ndarray:
    x = _preprocess(img)
    logits = _session.run(None, {_input_name: x})[0][0]
    return _softmax(logits)


def _letterbox(pil: Image.Image, size: int) -> Image.Image:
    img = pil.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _center_crop_square(pil: Image.Image, frac: float = 0.82) -> Image.Image:
    w, h = pil.size
    side = int(min(w, h) * frac)
    left = (w - side) // 2
    top = (h - side) // 2
    return pil.crop((left, top, left + side, top + side))


def _crop_instances(pil: Image.Image, max_n: int = 4) -> list[Image.Image]:
    rgb = np.asarray(pil.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    scale = 480 / max(h0, w0)
    if scale < 1:
        rgb_s = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb_s = rgb
    bgr = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 40, 30), (180, 255, 255))
    m[hsv[:, :, 2] > 245] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    h, w = m.shape
    min_a = 0.012 * h * w
    cnts = [c for c in sorted(cnts, key=cv2.contourArea, reverse=True) if cv2.contourArea(c) >= min_a][:max_n]
    crops = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        pad = int(0.12 * max(bw, bh))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = rgb_s[y0:y1, x0:x1].copy()
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        c2 = c.copy()
        c2[:, 0, 0] -= x0
        c2[:, 0, 1] -= y0
        cv2.drawContours(mask, [c2], -1, 255, -1)
        white = np.full_like(crop, 255)
        isolated = np.where(mask[:, :, None] > 0, crop, white)
        side = max(isolated.shape[0], isolated.shape[1])
        canvas = np.full((side, side, 3), 255, dtype=np.uint8)
        oy, ox = (side - isolated.shape[0]) // 2, (side - isolated.shape[1]) // 2
        canvas[oy : oy + isolated.shape[0], ox : ox + isolated.shape[1]] = isolated
        crops.append(Image.fromarray(canvas))
    return crops


def _hue_stats(pil: Image.Image) -> dict[str, float]:
    """Estadísticas HSV suaves para priors (no sustituyen al CNN)."""
    rgb = np.asarray(pil.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    scale = 320 / max(h0, w0)
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # máscara de píxeles “fruta” (saturados / no blancos)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    m = (sat > 40) & (val > 35) & (val < 250)
    if m.sum() < 80:
        m = (sat > 25) & (val > 25)
    if m.sum() < 40:
        return {
            "mean_h": 0,
            "mean_s": 0,
            "mean_v": 0,
            "orange_ratio": 0,
            "yellow_ratio": 0,
            "green_ratio": 0,
            "red_ratio": 0,
            "brown_ratio": 0,
        }

    hues = hsv[:, :, 0][m].astype(np.float32)
    s = sat[m].astype(np.float32)
    v = val[m].astype(np.float32)
    # OpenCV H: 0-179
    orange = ((hues >= 8) & (hues <= 22)).mean()
    yellow = ((hues >= 20) & (hues <= 38) & (s > 70) & (v > 90)).mean()
    green = ((hues >= 40) & (hues <= 85)).mean()
    red = ((hues <= 8) | (hues >= 160)).mean()
    brown = ((hues >= 8) & (hues <= 30) & (s < 90) & (v < 140)).mean()
    return {
        "mean_h": float(hues.mean()),
        "mean_s": float(s.mean()),
        "mean_v": float(v.mean()),
        "orange_ratio": float(orange),
        "yellow_ratio": float(yellow),
        "green_ratio": float(green),
        "red_ratio": float(red),
        "brown_ratio": float(brown),
    }


def _apply_rare_penalty(proba: np.ndarray) -> np.ndarray:
    out = proba.copy()
    for i, label in enumerate(_classes):
        _, fruit = label.split("_", 1)
        factor = RARE_FRUITS.get(fruit)
        if factor is not None:
            out[i] *= factor
    s = out.sum()
    return out / s if s > 0 else proba


def _apply_color_priors(proba: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    out = proba.copy()

    def bump(fruit: str, factor: float):
        for estado in ("fresh", "rotten"):
            key = f"{estado}_{fruit}"
            if key in _class_index:
                out[_class_index[key]] *= factor

    # Mango: mezcla verde + naranja/rojo, o naranja con verde residual
    if stats["green_ratio"] > 0.12 and (stats["orange_ratio"] > 0.15 or stats["red_ratio"] > 0.12):
        bump("mango", 1.55)
        bump("orange", 0.72)
        bump("jujube", 0.55)
        bump("kaki", 0.70)

    # Naranja “pura”: mucho naranja, poco verde/amarillo limón
    if stats["orange_ratio"] > 0.45 and stats["green_ratio"] < 0.08 and stats["yellow_ratio"] < 0.25:
        bump("orange", 1.25)
        bump("mango", 0.85)

    # Limón / lima amarilla (muy confundida con pimiento / tomate amarillo)
    if stats["yellow_ratio"] > 0.28 and stats["green_ratio"] < 0.35:
        bump("lime", 3.5)
        bump("capsicum", 0.18)
        bump("tomato", 0.15)
        bump("banana", 0.55)
        bump("orange", 0.45)
        bump("mango", 0.45)
        bump("potato", 0.35)
        # Si el CNN casi no votó limón, inyéctale masa (color inequívoco)
        for estado in ("fresh", "rotten"):
            key = f"{estado}_lime"
            if key in _class_index:
                i = _class_index[key]
                out[i] = max(float(out[i]), 0.22 if estado == "fresh" else 0.05)
    elif stats["yellow_ratio"] > 0.18 and stats["mean_s"] > 90:
        bump("lime", 2.0)
        bump("capsicum", 0.35)
        bump("tomato", 0.30)


    # Pimiento verde: mucho verde vegetal, poco amarillo limón
    if stats["green_ratio"] > 0.40 and stats["yellow_ratio"] < 0.12:
        bump("capsicum", 1.25)
        bump("lime", 0.80)

    # Manzana podrida vs papa: papas = marrón mate / baja sat
    if stats["mean_s"] > 70 and stats["red_ratio"] > 0.18:
        bump("apple", 1.35)
        bump("potato", 0.55)
    if stats["brown_ratio"] > 0.25 and stats["mean_s"] < 70 and stats["green_ratio"] < 0.08:
        bump("potato", 1.2)
        bump("apple", 0.85)

    # Azufaifo casi nunca es lo correcto en fotos stock de mesa
    if stats["orange_ratio"] > 0.2 or stats["green_ratio"] > 0.1 or stats["yellow_ratio"] > 0.15:
        bump("jujube", 0.45)

    s = out.sum()
    return out / s if s > 0 else proba


def _resolve_fruit_then_state(proba: np.ndarray) -> np.ndarray:
    """
    Reordena: gana la fruta con más masa (fresh+rotten),
    y dentro de ella el estado más probable. Evita 'naranja fresca'
    cuando 'mango' (fresh+rotten) suma más.
    """
    fruit_mass: dict[str, float] = {}
    for i, label in enumerate(_classes):
        _, fruit = label.split("_", 1)
        fruit_mass[fruit] = fruit_mass.get(fruit, 0.0) + float(proba[i])

    best_fruit = max(fruit_mass, key=fruit_mass.get)
    fresh_k = f"fresh_{best_fruit}"
    rotten_k = f"rotten_{best_fruit}"
    fresh_p = float(proba[_class_index[fresh_k]]) if fresh_k in _class_index else 0.0
    rotten_p = float(proba[_class_index[rotten_k]]) if rotten_k in _class_index else 0.0
    best_label = fresh_k if fresh_p >= rotten_p else rotten_k
    if best_label not in _class_index:
        return proba

    # Refuerza el label ganador sin aplastar el ranking del top
    out = proba.copy()
    winner = _class_index[best_label]
    boost = 1.0 + 0.35 * (fruit_mass[best_fruit] / max(proba.max(), 1e-6))
    out[winner] *= min(boost, 1.8)
    # suaviza otras frutas si la masa de la ganadora es clara
    runner = sorted(fruit_mass.values(), reverse=True)
    if len(runner) >= 2 and runner[0] >= runner[1] * 1.15:
        for i, label in enumerate(_classes):
            _, fruit = label.split("_", 1)
            if fruit != best_fruit:
                out[i] *= 0.82
    s = out.sum()
    return out / s if s > 0 else proba


def _pack(proba: np.ndarray, top_k: int, engine: str) -> dict:
    idxs = proba.argsort()[::-1][:top_k]
    best = int(idxs[0])
    label_raw = _classes[best]
    confidence = float(proba[best])
    estado, fruta = label_raw.split("_", 1)
    top = []
    for i in idxs:
        raw = _classes[int(i)]
        e, f = raw.split("_", 1)
        top.append(
            {
                "label_raw": raw,
                "fruit": FRUIT_ES.get(f, f.capitalize()),
                "fruit_key": f,
                "estado": "Fresca" if e == "fresh" else "Podrida",
                "emoji": FRUIT_EMOJI.get(f, "🍎"),
                "confidence": float(proba[int(i)]),
                "confidence_pct": f"{proba[int(i)] * 100:.1f}%",
            }
        )
    fruit_name = FRUIT_ES.get(fruta, fruta.capitalize())
    emoji = FRUIT_EMOJI.get(fruta, "🍎")
    estado_str = "Fresca" if estado == "fresh" else "Podrida"
    return {
        "fruit": fruit_name,
        "fruit_key": fruta,
        "estado": estado_str,
        "emoji": emoji,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "label_raw": label_raw,
        "message": f"{emoji} {fruit_name} — {estado_str}",
        "top": top,
        "engine": engine,
    }


def predict_cnn(pil_image: Image.Image, top_k: int = 3) -> dict:
    _load()
    img = prepare_image(pil_image)
    size = _img_size

    views: list[Image.Image] = []
    # Vistas “fieles” a la foto (clave en imágenes de Google)
    views.append(img.resize((size, size), Image.BILINEAR))
    views.append(_letterbox(img, size))
    views.append(_center_crop_square(img, 0.88).resize((size, size), Image.BILINEAR))
    views.append(_center_crop_square(img, 0.70).resize((size, size), Image.BILINEAR))
    views.append(ImageOps.mirror(views[0]))

    # Aislamiento (ayuda en mesa/cámara; a veces empeora stock photos → peso menor)
    try:
        iso = isolate_fruit_on_white(img, size=size)
        views.append(iso)
        views.append(ImageOps.mirror(iso))
    except Exception:
        pass

    crops = _crop_instances(img)
    for c in crops[:3]:
        views.append(_letterbox(c, size))

    probs = [_forward(v) for v in views]
    # Peso: raw/letterbox/center más altos que iso
    weights = []
    for i, _ in enumerate(views):
        if i <= 4:
            weights.append(1.25)  # raw-ish
        elif i <= 6:
            weights.append(0.65)  # iso
        else:
            weights.append(1.0)  # instance crops
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    avg = np.tensordot(w, np.stack(probs, axis=0), axes=(0, 0))

    stats = _hue_stats(img)
    avg = _apply_rare_penalty(avg)
    avg = _apply_color_priors(avg, stats)
    avg = _resolve_fruit_then_state(avg)

    return _pack(avg, top_k, engine="cnn+ensemble")
