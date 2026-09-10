<a id="readme-top"></a>

<div align="center">
  <h1>🍎 FruitGuard</h1>
  <p><strong>Clasificador inteligente de frutas y estado de frescura</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/GitHub-hexed--AAL1X-181717?style=for-the-badge&logo=github" />
  </p>

  <p>
    <a href="#sobre-el-proyecto">Sobre el proyecto</a> ·
    <a href="#mejoras-v2">Mejoras v2</a> ·
    <a href="#datasets">Datasets</a> ·
    <a href="#estructura">Estructura</a> ·
    <a href="#empezar">Empezar</a> ·
    <a href="#contributors">Contributors</a>
  </p>
</div>

---

## Sobre el proyecto

FruitGuard es un sistema automático de **control de calidad alimentaria** que clasifica frutas por tipo y estado (fresca o podrida) a partir de imágenes digitales, usando visión por computadora y aprendizaje automático.

**Objetivo:** reemplazar la inspección manual —lenta y subjetiva— con un pipeline automatizado que funcione en tiempo real sobre cualquier imagen.

---

## Mejoras v2

La versión original presentaba un problema crítico: **solo detectaba frutas correctamente sobre fondos blancos**. Esto hacía al sistema ineficiente en condiciones reales.

### 🔧 Problema raíz
Las características (color, textura, forma) se extraían de **toda la imagen**, incluyendo el fondo, lo que contaminaba el vector de features.

### ✅ Solución implementada — Segmentación GrabCut

| Aspecto                  | v1 (original)                          | v2 (mejorado)                                 |
|--------------------------|----------------------------------------|-----------------------------------------------|
| Segmentación             | `adaptiveThreshold` (asume fondo blanco) | **GrabCut** (cualquier fondo)                |
| Features de color        | Media/std de toda la imagen            | Solo píxeles de la fruta (máscara aplicada)  |
| Features de contorno     | Threshold binario                      | Máscara GrabCut directamente                 |
| Robustez                 | Falla con fondos oscuros/coloridos     | Fallback elíptico si GrabCut falla           |
| Interface                | Gradio (local)                         | **FastAPI REST + Frontend web**              |

### Otros problemas corregidos
- **`ZeroDivisionError`** en circularidad cuando el perímetro era 0
- **Feature vector inestable**: tamaño variable ahora manejado con padding/truncado
- **Sin API**: ahora tiene un backend REST con endpoints documentados (`/docs`)
- **Sin frontend desacoplado**: interfaz web independiente del backend

---

## Datasets

El modelo actual (`fruit_cnn.pt`) se entrenó con datos públicos de Kaggle. **Créditos a sus autores:**

| Dataset | Autor | Uso en FruitGuard | Enlace |
|---------|--------|-------------------|--------|
| **Freshness44** | [siavash93](https://www.kaggle.com/siavash93) | Entrenamiento principal del CNN (≈53k imágenes, 22 frutas/verduras × fresca/podrida, fondos variados) | [kaggle.com/datasets/siavash93/freshness44](https://www.kaggle.com/datasets/siavash93/freshness44) |
| **Fresh and Stale Classification** | [swoyam2609](https://www.kaggle.com/swoyam2609) | Prototipo inicial y experimentos previos de clasificación fresca/podrida | [kaggle.com/datasets/swoyam2609/fresh-and-stale-classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) |

> Los datasets **no** se incluyen en este repositorio (solo el modelo ya entrenado). Si reentrenas, descárgalos desde Kaggle y respeta sus términos de uso / licencia.

---

## Estructura

```
FruitGuard/
├── backend/
│   ├── main.py              ← API FastAPI (/predict, /health, sirve /app)
│   ├── cnn_predictor.py     ← MobileNetV3 + ensemble
│   ├── preprocess.py        ← Aislamiento de fruta
│   ├── predictor.py
│   ├── requirements.txt
│   └── models/
│       └── fruit_cnn.pt     ← Modelo Freshness44
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── icon.png
│
└── README.md
```

---

## Empezar

### Prerrequisitos

```bash
python --version  # >= 3.10
```

### Instalación

```bash
cd backend/
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar

```bash
cd backend/
uvicorn main:app --host 0.0.0.0 --port 8000
```

Abre:
- **App**: http://127.0.0.1:8000/app/
- **API docs**: http://127.0.0.1:8000/docs

---

## API Reference

```
POST /predict
  Body: multipart/form-data  { file: <imagen> }
  Response: {
    "fruit": "Manzana",
    "estado": "Fresca",
    "emoji": "🍎",
    "confidence": 0.94,
    "confidence_pct": "94.0%",
    "message": "🍎 Manzana — Fresca",
    "top": [...]
  }

GET /health   → { "status": "ok" }
GET /docs     → Swagger UI
```

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ZtanQ">
        <img src="https://github.com/ZtanQ.png" width="80" alt="ZtanQ" /><br/>
        <sub><b>Gabriel Reyna</b></sub><br/>
        <sub>ZtanQ</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Dreelliot">
        <img src="https://github.com/Dreelliot.png" width="80" alt="Dreelliot" /><br/>
        <sub><b>André Elliot</b></sub><br/>
        <sub>Dreelliot</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hexed-AAL1X">
        <img src="https://github.com/hexed-AAL1X.png" width="80" alt="aal1x" /><br/>
        <sub><b>Leonardo Bravo</b></sub><br/>
        <sub>aal1x</sub>
      </a>
    </td>
  </tr>
</table>

---

<p align="right"><a href="#readme-top">↑ Volver arriba</a></p>
