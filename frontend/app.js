/**
 * FruitGuard — cámara en vivo + upload
 * v3.0 — splash + layout sin scroll
 */

function resolveApiBase() {
  // file:// → hay que pegarle al backend directo
  if (window.location.protocol === "file:") {
    return "http://127.0.0.1:8000";
  }
  // Cualquier http(s): mismo host (sirve para localhost, IP y port-forward)
  return window.location.origin;
}

const API_BASE = resolveApiBase();
console.info("[FruitGuard] API_BASE =", API_BASE);

const qs = (id) => document.getElementById(id);

const video = qs("video");
const snap = qs("snap");
const camBtn = qs("camBtn");
const liveBadge = qs("liveBadge");
const liveResult = qs("liveResult");
const liveEmoji = qs("liveEmoji");
const liveFruit = qs("liveFruit");
const liveEstado = qs("liveEstado");
const liveConf = qs("liveConf");
const topList = qs("topList");
const speedSelect = qs("speedSelect");
const errorCard = qs("errorCard");
const errorMsg = qs("errorMsg");
const livePanel = qs("livePanel");
const uploadPanel = qs("uploadPanel");

let stream = null;
let loopTimer = null;
let busy = false;
let camOn = false;
let failStreak = 0;

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const mode = btn.dataset.mode;
    livePanel.classList.toggle("hidden", mode !== "live");
    uploadPanel.classList.toggle("hidden", mode !== "upload");
    if (mode !== "live") stopCamera();
  });
});

camBtn.addEventListener("click", async () => {
  if (camOn) stopCamera();
  else await startCamera();
});

async function pingBackend() {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 2500);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal, cache: "no-store" });
    clearTimeout(t);
    if (!res.ok) throw new Error(`health ${res.status}`);
    return true;
  } catch (e) {
    clearTimeout(t);
    return false;
  }
}

async function startCamera() {
  const ok = await pingBackend();
  if (!ok) {
    showError(
      `No hay backend en ${API_BASE}. Abre exactamente: http://127.0.0.1:8000/app/`
    );
    return;
  }
  hideError();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    // fallback webcam
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err2) {
      showError("No se pudo abrir la cámara. Permite el acceso en el navegador.");
      return;
    }
  }

  video.srcObject = stream;
  await video.play().catch(() => {});
  camOn = true;
  failStreak = 0;
  camBtn.textContent = "Apagar cámara";
  liveBadge.textContent = "● En vivo";
  liveBadge.classList.add("on");
  scheduleLoop();
}

function stopCamera() {
  if (loopTimer) clearTimeout(loopTimer);
  loopTimer = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  camOn = false;
  camBtn.textContent = "Encender cámara";
  liveBadge.textContent = "Cámara apagada";
  liveBadge.classList.remove("on");
}

function scheduleLoop() {
  if (!camOn) return;
  const ms = Number(speedSelect.value) || 400;
  loopTimer = setTimeout(tick, ms);
}

async function tick() {
  if (!camOn) return;
  if (!busy) {
    busy = true;
    try {
      const blob = await captureFrame();
      if (blob && blob.size > 0) {
        const data = await predictBlob(blob);
        failStreak = 0;
        renderLive(smoothPredict(data));
        hideError();
      }
    } catch (e) {
      failStreak += 1;
      console.warn("[FruitGuard] tick error", e);
      // Solo mostrar error tras varios fallos seguidos (evita spam)
      if (failStreak >= 3) {
        showError(String(e.message || e));
      }
    } finally {
      busy = false;
    }
  }
  scheduleLoop();
}

function captureFrame() {
  if (!video.videoWidth || video.readyState < 2) {
    return Promise.resolve(null);
  }
  const w = 320;
  const h = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * w));
  snap.width = w;
  snap.height = h;
  const ctx = snap.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  return new Promise((resolve) => {
    snap.toBlob((b) => resolve(b), "image/jpeg", 0.8);
  });
}

async function predictBlob(blobOrFile) {
  const fd = new FormData();
  const file =
    blobOrFile instanceof File
      ? blobOrFile
      : new File([blobOrFile], "frame.jpg", { type: "image/jpeg" });
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: fd,
    cache: "no-store",
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function renderLive(data) {
  liveEmoji.textContent = data.emoji || "🍎";
  liveFruit.textContent = data.fruit || "—";
  liveEstado.textContent = data.estado || "";
  liveConf.textContent = `Confianza: ${data.confidence_pct || "—"}`;
  liveResult.classList.remove("fresh", "rotten", "uncertain");
  if (data.uncertain) liveResult.classList.add("uncertain");
  else liveResult.classList.add(data.estado === "Fresca" ? "fresh" : "rotten");
  renderTop(topList, data.top || []);
}

/** Suavizado: vota las últimas N predicciones para no parpadear. */
const VOTE_N = 4;
const voteBuf = [];

function smoothPredict(data) {
  const conf = Number(data.confidence) || 0;
  // Confianza alta: limpia el buffer (evita quedar pegado a "pimiento")
  if (conf >= 0.55) {
    voteBuf.length = 0;
  }
  voteBuf.push({ key: data.label_raw, data, conf });
  if (voteBuf.length > VOTE_N) voteBuf.shift();

  const scores = {};
  for (const v of voteBuf) {
    scores[v.key] = (scores[v.key] || 0) + 0.5 + v.conf;
  }
  let bestKey = data.label_raw;
  let bestScore = -1;
  for (const [k, s] of Object.entries(scores)) {
    if (s > bestScore) {
      bestScore = s;
      bestKey = k;
    }
  }
  const picked = voteBuf.filter((v) => v.key === bestKey).at(-1)?.data || data;
  const avgConf =
    voteBuf.filter((v) => v.key === bestKey).reduce((a, v) => a + v.conf, 0) /
    Math.max(1, voteBuf.filter((v) => v.key === bestKey).length);

  const uncertain = avgConf < 0.45;
  return {
    ...picked,
    confidence: avgConf,
    confidence_pct: `${(avgConf * 100).toFixed(1)}%`,
    uncertain,
    estado: uncertain ? `${picked.estado} (poco seguro)` : picked.estado,
  };
}

function renderTop(el, items) {
  if (!items || !items.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = items
    .slice(0, 2)
    .map(
      (t) => `
      <div class="top-item">
        <span>${t.emoji || ""} <strong>${t.fruit}</strong> · ${t.estado}</span>
        <span class="pct">${t.confidence_pct || ""}</span>
      </div>`
    )
    .join("");
}

/* ── Splash ──────────────────────────────────────────────────── */
async function runSplash() {
  const splash = qs("splash");
  const fill = qs("splashFill");
  const status = qs("splashStatus");
  const shell = qs("appShell");
  if (!splash || !fill || !shell) return;

  const steps = [
    { pct: 18, msg: "Calibrando sensores…" },
    { pct: 42, msg: "Cargando modelo Freshness44…" },
    { pct: 68, msg: "Conectando API…" },
    { pct: 88, msg: "Preparando cámara…" },
    { pct: 100, msg: "Listo" },
  ];

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const start = performance.now();

  for (const step of steps) {
    if (step.pct === 68) await pingBackend().catch(() => false);
    fill.style.width = `${step.pct}%`;
    fill.parentElement?.setAttribute("aria-valuenow", String(step.pct));
    if (status) status.textContent = step.msg;
    await wait(step.pct === 100 ? 280 : 420 + Math.random() * 180);
  }

  const elapsed = performance.now() - start;
  if (elapsed < 2200) await wait(2200 - elapsed);

  splash.classList.add("is-done");
  shell.classList.remove("is-locked");
  shell.classList.add("is-ready");
  setTimeout(() => splash.remove(), 800);
}

// ── Upload ────────────────────────────────────────────────────
const dropZone = qs("dropZone");
const fileInput = qs("fileInput");
const previewZone = qs("previewZone");
const previewImg = qs("previewImg");
const changeBtn = qs("changeBtn");
const analyzeBtn = qs("analyzeBtn");
const btnText = qs("btnText");
const btnSpinner = qs("btnSpinner");
const uploadResult = qs("uploadResult");
const upTopList = qs("upTopList");

let currentFile = null;

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});
changeBtn.addEventListener("click", () => fileInput.click());
analyzeBtn.addEventListener("click", analyzeUpload);

function setFile(file) {
  currentFile = file;
  previewImg.src = URL.createObjectURL(file);
  dropZone.classList.add("hidden");
  previewZone.classList.remove("hidden");
  analyzeBtn.disabled = false;
  uploadResult.classList.add("hidden");
  upTopList.innerHTML = "";
}

async function analyzeUpload() {
  if (!currentFile) return;
  analyzeBtn.disabled = true;
  btnText.textContent = "Analizando...";
  btnSpinner.classList.remove("hidden");
  try {
    const ok = await pingBackend();
    if (!ok) throw new Error(`Backend caído en ${API_BASE}`);
    const data = await predictBlob(currentFile);
    qs("upEmoji").textContent = data.emoji;
    qs("upFruit").textContent = data.fruit;
    qs("upEstado").textContent = data.estado;
    qs("upConf").textContent = `Confianza: ${data.confidence_pct}`;
    uploadResult.classList.remove("hidden", "fresh", "rotten");
    uploadResult.classList.add(data.estado === "Fresca" ? "fresh" : "rotten");
    renderTop(upTopList, data.top || []);
    hideError();
  } catch (e) {
    showError(e.message || "Error al analizar");
  } finally {
    analyzeBtn.disabled = false;
    btnText.textContent = "Analizar fruta";
    btnSpinner.classList.add("hidden");
  }
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.classList.remove("hidden");
}
function hideError() {
  errorCard.classList.add("hidden");
}

window.addEventListener("beforeunload", stopCamera);

qs("errorClose")?.addEventListener("click", hideError);

// Aviso si abren el HTML como archivo
if (window.location.protocol === "file:") {
  showError("Estás abriendo el HTML como archivo. Usa http://127.0.0.1:8000/app/");
}

runSplash();
