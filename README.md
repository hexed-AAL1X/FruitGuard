<a id="readme-top"></a>
<!-- SHIELDS -->
<img src="https://github.com/AnderMendoza/AnderMendoza/raw/main/assets/line-neon.gif" width="100%">
<p align='center'> 
  <img alt="GitHub Repo contributors" src="https://img.shields.io/github/contributors/hexed-AAL1X/FruitGuard?style=for-the-badge">&nbsp;
  <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/hexed-AAL1X/FruitGuard?style=for-the-badge">&nbsp;
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/hexed-AAL1X/FruitGuard?style=for-the-badge">&nbsp;
  <img alt="GitHub Repo issues" src="https://img.shields.io/github/issues/hexed-AAL1X/FruitGuard?style=for-the-badge">&nbsp;
</p>

<!-- PROJECT LOGO -->
<br>
<div align="center">
   <img src="assets/images/logo.png" alt="FruitGuard Logo" width="120">
   <h3 align="center">🍎 FruitGuard</h3>
   <p align="center">
     Clasificador inteligente de frutas y frescura en tiempo real
     <br>
     <a href="https://github.com/hexed-AAL1X/FruitGuard"><strong>Explore the docs »</strong></a>
     <br>
     <br>
     <a href="https://github.com/hexed-AAL1X/FruitGuard">View Demo</a>
     ·
     <a href="https://github.com/hexed-AAL1X/FruitGuard/issues/new?labels=bug">Report Bug</a>
     ·
     <a href="https://github.com/hexed-AAL1X/FruitGuard/issues/new?labels=enhancement">Request Feature</a>
   </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#important-notices">Important Notices</a></li>
    <li><a href="#datasets">Datasets</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#contributing">Contributing</a>
      <ul>
        <li><a href="#top-contributors">Top Contributors</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>
<br>

<!-- ABOUT THE PROJECT -->
<a id="about-the-project"></a>***About The Project***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<div align="center">
  <img src="assets/images/dashboard.png" alt="FruitGuard dashboard — detección real de banana fresca" width="900">
  <p><em>Captura real de la app: banana clasificada como fresca</em></p>
</div>

**FruitGuard** es un sistema de control de calidad alimentaria que clasifica frutas/verduras por **tipo** y **estado** (fresca o podrida) a partir de imágenes, con cámara en vivo o foto subida.

Nació para reemplazar la inspección manual —lenta y subjetiva— con un pipeline de visión por computadora usable en condiciones reales (mesa, fondos variados, varias piezas).

Here's why:

* Detecta en **tiempo real** desde la webcam o desde una imagen.
* Usa un **CNN (MobileNetV3 → ONNX)** entrenado con Freshness44 + ensemble en inferencia.
* Backend **FastAPI** + frontend web (sin Gradio).
* Pensado para desplegarse en un solo servicio (p. ej. Render).

<a id="built-with"></a>
### Built With
* ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)&nbsp;
* ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)&nbsp;
* ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)&nbsp;
* ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)&nbsp;
* ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)&nbsp;
* ![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)&nbsp;
* ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)&nbsp;
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- IMPORTANT NOTICES -->
<a id="important-notices"></a>***Important Notices***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

> [!NOTE]  
> Para instalar y ejecutar FruitGuard necesitas:

> | Requirement | Description |
> |-------------|-------------|
> | Python | ![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white&color=black) |
> | API | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white&color=black) |
> | ML | ![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white&color=black) |

> [!IMPORTANT]  
> El modelo ya viene entrenado en `backend/models/fruit_cnn.onnx`. **No hace falta** el dataset para usar la app; solo si quieres reentrenar.

> [!WARNING]  
> No es un modelo tipo GPT/Gemini: puede fallar en fotos difíciles (pantallas, fondos raros, clases muy parecidas). Sigue mejorando con priors + ensemble.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASETS -->
<a id="datasets"></a>***Datasets***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

Créditos a los autores de los datasets públicos usados para entrenar / prototipar:

| Dataset | Autor | Uso | Enlace |
|---------|--------|-----|--------|
| **Freshness44** | [siavash93](https://www.kaggle.com/siavash93) | Entrenamiento principal del CNN (~53k imágenes, fresh/rotten, fondos variados) | [Kaggle · Freshness44](https://www.kaggle.com/datasets/siavash93/freshness44) |
| **Fresh and Stale Classification** | [swoyam2609](https://www.kaggle.com/swoyam2609) | Prototipo inicial fresca/podrida | [Kaggle · Fresh and Stale](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) |

> Los datasets **no** se incluyen en este repo. Descárgalos en Kaggle si reentrenas y respeta su licencia.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
<a id="getting-started"></a>***Getting Started***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

Instrucciones para levantar FruitGuard en local.

<a id="prerequisites"></a>
### Prerequisites
* Python **3.10+**
* `pip` / `venv`
* Cámara (opcional, para modo en vivo)

<a id="installation"></a>
### Installation

1. Clona el repositorio
   ```sh
   git clone https://github.com/hexed-AAL1X/FruitGuard.git
   ```
2. Entra al proyecto
   ```sh
   cd FruitGuard
   ```
3. Crea el entorno e instala dependencias
   ```sh
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Arranca la API (también sirve el frontend)
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
5. Abre la app
   - App: http://127.0.0.1:8000/app/
   - Docs: http://127.0.0.1:8000/docs

#### Estructura

```
FruitGuard/
├── backend/          # FastAPI + CNN
│   └── models/fruit_cnn.onnx
├── frontend/         # UI web
├── assets/images/    # Logo README
└── README.md
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
<a id="contributing"></a>***Contributing***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

Las contribuciones hacen grande al open source. ¡Cualquier PR es bienvenida!

1. Fork del proyecto
2. Crea una rama (`git checkout -b feature/Mejora`)
3. Commit (`git commit -m 'Add Mejora'`)
4. Push (`git push origin feature/Mejora`)
5. Abre un Pull Request

<a id="top-contributors"></a>
### Top contributors

<div align="center">

<table>
  <tr>
    <td align="center" width="160">
      <a href="https://github.com/ZtanQ">
        <img src="https://github.com/ZtanQ.png" width="88" height="88" alt="Gabriel Reyna" style="border-radius:50%;" /><br />
        <b>Gabriel Reyna</b><br />
        <sub>@ZtanQ</sub>
      </a>
    </td>
    <td align="center" width="160">
      <a href="https://github.com/Dreelliot">
        <img src="https://github.com/Dreelliot.png" width="88" height="88" alt="André Elliot" style="border-radius:50%;" /><br />
        <b>André Elliot</b><br />
        <sub>@Dreelliot</sub>
      </a>
    </td>
  </tr>
</table>

</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
<a id="contact"></a>***Contact***
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">
<p align="center">
  <a href="mailto:hexed_aal1x.ops@proton.me"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white&color=black" /></a>
  <a href="https://www.instagram.com/hexed_aal1x"><img src="https://img.shields.io/badge/instagram-%2312100E.svg?&style=for-the-badge&logo=instagram&logoColor=white&color=black" /></a>
  <a href="https://www.linkedin.com/in/leonardo-bravo-4120b8228/"><img src="https://img.shields.io/badge/linkedin-%2312100E.svg?&style=for-the-badge&logo=linkedin&logoColor=white&color=black" /></a>
</p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>
