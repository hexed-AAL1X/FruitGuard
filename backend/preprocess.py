"""
preprocess.py — Aísla fruta(s) del fondo (mesa, madera, varias piezas).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def isolate_fruit_on_white(pil_image: Image.Image, size: int = 224) -> Image.Image:
    rgb = np.asarray(pil_image.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    scale = 480 / max(h0, w0)
    if scale < 1:
        rgb_s = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb_s = rgb.copy()
    bgr = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    mask = _best_mask(bgr)
    mask = _keep_large_blobs(mask, min_frac=0.008)

    coverage = float(mask.mean() / 255.0)
    if coverage < 0.025 or coverage > 0.93:
        mask = _center_ellipse(h, w, 0.80)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pil_image.convert("RGB").resize((size, size))

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = int(0.1 * max(x1 - x0, y1 - y0, 1))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)

    crop_rgb = rgb_s[y0 : y1 + 1, x0 : x1 + 1]
    crop_m = mask[y0 : y1 + 1, x0 : x1 + 1]
    white = np.full_like(crop_rgb, 255)
    isolated = np.where(crop_m[:, :, None] > 0, crop_rgb, white)

    ch, cw = isolated.shape[:2]
    side = max(ch, cw)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy : oy + ch, ox : ox + cw] = isolated
    return Image.fromarray(canvas).resize((size, size), Image.BILINEAR)


def _best_mask(bgr: np.ndarray) -> np.ndarray:
    """Combina GrabCut + saturación (frutas coloridas sobre mesa)."""
    gc = _grabcut_mask(bgr)
    sat = _saturation_mask(bgr)
    # unión si ambos aportan
    if sat.mean() > 5:
        combo = cv2.bitwise_or(gc, sat)
        # si combo es absurdo, preferir el más razonable
        cov = combo.mean() / 255.0
        if 0.04 <= cov <= 0.85:
            return combo
        if 0.04 <= sat.mean() / 255.0 <= 0.85:
            return sat
    return gc


def _saturation_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # frutas suelen ser más saturadas que madera/gris
    m = cv2.inRange(hsv, (0, 50, 40), (180, 255, 255))
    # quitar bordes muy claros (blanco)
    v = hsv[:, :, 2]
    m[v > 245] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return m


def _grabcut_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mx, my = max(int(w * 0.06), 4), max(int(h * 0.06), 4)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return _center_ellipse(h, w, 0.75)


def _keep_large_blobs(mask: np.ndarray, min_frac: float = 0.008) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    h, w = mask.shape
    min_area = min_frac * h * w
    # quedarnos con los top blobs (hasta 5 frutas)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    out = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out if out.any() else mask


def _center_ellipse(h: int, w: int, ratio: float = 0.75) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * ratio / 2), int(h * ratio / 2)), 0, 0, 360, 255, -1)
    return m
