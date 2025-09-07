import React, { useState, useEffect, useRef } from "react";

const BACKEND_URL = "http://10.44.246.170:5000";
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:5000";


/* ================== Fokus-Score Tuning ================== */
const ROI_SCALE = 0.55;          // 0.4–0.6 testen
const W_TENENGRAD = 0.40;
const W_LAPLACE   = 0.35;
const W_HF        = 0.25;
const DOG_SIGMA_SMALL = 0.8;
const DOG_SIGMA_LARGE = 1.6;

/* ================== Kleine UI-Bausteine ================== */
function Panel({ title, right, children }) {
  return (
    <section className="bg-white/70 backdrop-blur border rounded-2xl shadow p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}

function Badge({ tone = "info", children }) {
  const map = {
    info: "bg-blue-50 text-blue-700 border-blue-200",
    ok: "bg-emerald-50 text-emerald-700 border-emerald-200",
    warn: "bg-amber-50 text-amber-700 border-amber-200",
    err: "bg-rose-50 text-rose-700 border-rose-200",
    mute: "bg-gray-50 text-gray-700 border-gray-200",
  };
  return (
    <span className={`inline-block px-2 py-0.5 text-xs rounded border ${map[tone]}`}>{children}</span>
  );
}
function Btn({
  children,
  onClick,
  disabled,
  tone = "primary",
  className = "",
  type = "button",
  title,
}) {
  const map = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white",
    success: "bg-emerald-600 hover:bg-emerald-700 text-white",
    danger: "bg-rose-600 hover:bg-rose-700 text-white",
    neutral: "bg-gray-700 hover:bg-gray-800 text-white",
    soft: "bg-gray-200 hover:bg-gray-300 text-gray-800",
    amber: "bg-amber-600 hover:bg-amber-700 text-white",
    teal: "bg-teal-600 hover:bg-teal-700 text-white",
  };
  const base =
    "px-3 py-2 rounded-lg text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed transition";
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${map[tone]} ${className}`}
    >
      {children}
    </button>
  );
}

/* ================== App ================== */
function App() {
  const imgRef = useRef(null);

  // Stream & Canvas
  const [streamReady, setStreamReady] = useState(false);
  const afCanvasRef = useRef(null);
  const afCtxRef = useRef(null);

  // UI / State
  const [autofocusScore, setAutofocusScore] = useState(0);
  const [posResponse, setPosResponse] = useState("");
  const [coords, setCoords] = useState({ x: "", y: "", z: "" });
  const [stepSize, setStepSize] = useState(100);
  const [isLoading, setIsLoading] = useState(false);

  // Stitching
  const [stitchStatus, setStitchStatus] = useState("");
  const [stitchingActive, setStitchingActive] = useState(false);
  const [imageDims, setImageDims] = useState(null);
  const [stepX, setStepX] = useState(0);
  const [stepY, setStepY] = useState(0);

  // Smart/NextGen AF
  const [smartAfBusy, setSmartAfBusy] = useState(false);
  const [smartAf, setSmartAf] = useState(false);
  const [smartAfStatus, setSmartAfStatus] = useState(null);
  const [nextgenBusy, setNextgenBusy] = useState(false);
  const [nextgenStatus, setNextgenStatus] = useState(null);

  // Video Sweep
  const sweepCancelRef = useRef(false);

  // "photo" oder "move"
  const [stitchTurn, setStitchTurn] = useState("photo"); 


  /* ================== Helpers ================== */
  const nowMs = () => (performance?.now?.() ?? Date.now());
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ================== Effects ================== */
  // Stream initialisieren
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    img.crossOrigin = "anonymous";
    const onLoad = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) setStreamReady(true);
    };
    img.addEventListener("load", onLoad);
    img.src = `${BACKEND_URL}/video_feed`;
    return () => img.removeEventListener("load", onLoad);
  }, []);

  // Bilddimensionen holen (Schrittweiten berechnen)
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/image_dimensions`);
        const data = await res.json();
        setImageDims(data);
        const overlap = 0.3;
        setStepX(Math.round(data.width_um * 2 * (1 - overlap)));
        setStepY(Math.round(data.height_um * 2 * (1 - overlap)));
      } catch (e) {
        console.error("image_dimensions Fehler:", e);
      }
    })();
  }, []);

  // Tastatursteuerung (W/A/S/D, Q/E; Shift = 5× Schrittweite)
  useEffect(() => {
    const onKey = (ev) => {
      const k = ev.key.toLowerCase();
      const mult = ev.shiftKey ? 5 : 1;
      const s = stepSize * mult;
      if (["w","a","s","d","q","e"].includes(k)) ev.preventDefault();
      if (k === "w") moveAxis(0,  s, 0, true);
      if (k === "s") moveAxis(0, -s, 0, true);
      if (k === "a") moveAxis( s, 0, 0, true);
      if (k === "d") moveAxis( -s, 0, 0, true);
      if (k === "q") moveAxis(0, 0,  s, true);
      if (k === "e") moveAxis(0, 0, -s, true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepSize]);

  /* ================== Fokus-Scoring ================== */
  function tenengradScore(imageData, w, h) {
    const p = imageData.data;
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < p.length; i += 4, j++) {
      gray[j] = 0.299 * p[i] + 0.587 * p[i + 1] + 0.114 * p[i + 2];
    }
    const kx = [-1,0,1,-2,0,2,-1,0,1];
    const ky = [-1,-2,-1,0,0,0,1,2,1];
    let sum = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let gx = 0, gy = 0, kv = 0;
        for (let yy = -1; yy <= 1; yy++) {
          const row = (y + yy) * w;
          for (let xx = -1; xx <= 1; xx++) {
            const gi = row + (x + xx);
            gx += gray[gi] * kx[kv];
            gy += gray[gi] * ky[kv];
            kv++;
          }
        }
        sum += gx * gx + gy * gy;
      }
    }
    return sum / ((w - 2) * (h - 2));
  }

  function laplacianVarScore(imageData, w, h) {
    const p = imageData.data;
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < p.length; i += 4, j++) {
      gray[j] = 0.299 * p[i] + 0.587 * p[i + 1] + 0.114 * p[i + 2];
    }
    const kernel = [0,1,0, 1,-4,1, 0,1,0];
    const resp = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let v = 0, kv = 0;
        for (let yy = -1; yy <= 1; yy++) {
          const row = (y + yy) * w;
          for (let xx = -1; xx <= 1; xx++) {
            v += gray[row + (x + xx)] * kernel[kv++];
          }
        }
        resp[y * w + x] = v;
      }
    }
    let sum = 0, sumSq = 0, n = (w - 2) * (h - 2);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const v = resp[y * w + x];
        sum += v; sumSq += v * v;
      }
    }
    const mean = sum / n;
    return Math.max(0, (sumSq / n) - mean * mean);
  }

  function gaussianKernel1D(sigma) {
    const radius = Math.max(1, Math.floor(sigma * 3));
    const k = new Float32Array(radius * 2 + 1);
    const s2 = 2 * sigma * sigma;
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      const v = Math.exp(-(i * i) / s2);
      k[i + radius] = v; sum += v;
    }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    return { k, radius };
  }

  function blurSeparable(gray, w, h, sigma) {
    const { k, radius } = gaussianKernel1D(sigma);
    const tmp = new Float32Array(w * h);
    const out = new Float32Array(w * h);
    // horizontal
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let t = -radius; t <= radius; t++) {
          const xx = Math.min(w - 1, Math.max(0, x + t));
          acc += gray[y * w + xx] * k[t + radius];
        }
        tmp[y * w + x] = acc;
      }
    }
    // vertikal
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let acc = 0;
        for (let t = -radius; t <= radius; t++) {
          const yy = Math.min(h - 1, Math.max(0, y + t));
          acc += tmp[yy * w + x] * k[t + radius];
        }
        out[y * w + x] = acc;
      }
    }
    return out;
  }

  function highFreqEnergy_DoG(imageData, w, h, sSmall = DOG_SIGMA_SMALL, sLarge = DOG_SIGMA_LARGE) {
    const p = imageData.data;
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < p.length; i += 4, j++) {
      gray[j] = 0.299 * p[i] + 0.587 * p[i + 1] + 0.114 * p[i + 2];
    }
    const gSmall = blurSeparable(gray, w, h, sSmall);
    const gLarge = blurSeparable(gray, w, h, sLarge);
    let sum = 0;
    for (let i = 0; i < gray.length; i++) {
      const band = gSmall[i] - gLarge[i];
      sum += band * band;
    }
    return sum / (w * h);
  }

  function captureScoreFromImg(img, roiScale = ROI_SCALE) {
    const w = img.naturalWidth | 0;
    const h = img.naturalHeight | 0;
    if (!w || !h || !img.complete) return null;

    let canvas = afCanvasRef.current;
    let ctx = afCtxRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      afCanvasRef.current = canvas;
    }
    if (!ctx) {
      ctx = canvas.getContext("2d", { willReadFrequently: true });
      afCtxRef.current = ctx;
    }

    const rw = Math.max(16, Math.floor(w * roiScale));
    const rh = Math.max(16, Math.floor(h * roiScale));
    const rx = Math.floor((w - rw) / 2);
    const ry = Math.floor((h - rh) / 2);

    if (canvas.width !== rw) canvas.width = rw;
    if (canvas.height !== rh) canvas.height = rh;

    ctx.drawImage(img, rx, ry, rw, rh, 0, 0, rw, rh);

    try {
      const imageData = ctx.getImageData(0, 0, rw, rh);
      const sTen = tenengradScore(imageData, rw, rh);
      const sLap = laplacianVarScore(imageData, rw, rh);
      const sHf  = highFreqEnergy_DoG(imageData, rw, rh);
      const nTen = Math.log10(sTen + 1);
      const nLap = Math.log10(sLap + 1);
      const nHf  = Math.log10(sHf  + 1);
      return W_TENENGRAD * nTen + W_LAPLACE * nLap + W_HF * nHf;
    } catch (e) {
      console.warn("captureScore error:", e?.message || e);
      return null;
    }
  }

  function parabolicFitZ(frames, bestIdx) {
    const a = frames[bestIdx - 1];
    const b = frames[bestIdx];
    const c = frames[bestIdx + 1];
    if (!a || !b || !c) return b?.z;
    const x1 = a.z, y1 = a.score;
    const x2 = b.z, y2 = b.score;
    const x3 = c.z, y3 = c.score;
    const denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    if (denom === 0) return x2;
    const A = (x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3 - y2)) / denom;
    const B = (x3*x3*(y1 - y2) + x2*x2*(y3 - y1) + x1*x1*(y2 - y3)) / denom;
    const xv = -B / (2 * A);
    return isFinite(xv) ? xv : x2;
  }

  /* ================== Backend Calls ================== */
  const sendZoom = async (direction) => {
    try {
      await fetch(`${BACKEND_URL}/zoom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ direction }),
      });
    } catch (err) {
      console.error("Zoom-Fehler:", err);
    }
  };

  const moveAxis = async (x = 0, y = 0, z = 0, showStatus = false) => {
    try {
      const res = await fetch(`${BACKEND_URL}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x, y, z }),
      });
      const data = await res.json();
      if (showStatus) {
        setStitchStatus(
          data.status === "moving" || data.status === "moved" ? "Bewegt!" : "Fehler bei Bewegung!"
        );
      }
    } catch (err) {
      if (showStatus) setStitchStatus("Fehler bei Bewegung!");
      console.error("Move-Fehler:", err);
    }
  };

  const runAutofocus = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/autofocus`, { method: "POST" });
      const data = await res.json();
      setAutofocusScore(data?.autofocus_score || 0);
    } catch (err) {
      console.error("Autofokus-Fehler:", err);
      setAutofocusScore(0);
    } finally {
      setIsLoading(false);
    }
  };

  async function handleSmartAutofocus() {
    setSmartAfBusy(true);
    setSmartAfStatus(null);
    try {
      const res = await fetch(`${API_BASE}/autofocus/smart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (!res.ok || data.status !== "ok") {
        setSmartAfStatus({ error: data?.message || `HTTP ${res.status}` });
        return;
      }
      setSmartAfStatus({
        best_z_um: data.best_z_um,
        score: data.score,
        window: data.window,
        clipped: data.clipped,
      });
    } catch (e) {
      setSmartAfStatus({ error: String(e) });
    } finally {
      setSmartAfBusy(false);
    }
  }

  async function handleNextGenAutofocus() {
    setNextgenBusy(true);
    setNextgenStatus(null);
    try {
      const res = await fetch(`${API_BASE}/autofocus/nextgen`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (!res.ok || data.status !== "ok") {
        setNextgenStatus({ error: data?.message || `HTTP ${res.status}` });
        return;
      }
      setNextgenStatus({
        best_z_um: data.best_z_um,
        score: data.score,
        coverage: data.coverage,
        coarse_points: data.coarse_points,
        fine_points: data.fine_points,
      });
    } catch (e) {
      setNextgenStatus({ error: String(e) });
    } finally {
      setNextgenBusy(false);
    }
  }

  const getPosition = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/position`);
      const data = await res.json();
      setPosResponse(data.position);
    } catch (err) {
      console.error("Positionsabfrage fehlgeschlagen", err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCoords((prev) => ({ ...prev, [name]: value }));
  };

  const handleMoveInput = () => {
    moveAxis(Number(coords.x), Number(coords.y), Number(coords.z), true);
  };

  // Stitching API
  const handleStitchingStart = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/stitching/start`, { method: "POST" });
      const data = await res.json();
      if (data.status === "started") {
        setStitchStatus("Stitching gestartet!");
        setStitchingActive(true);
      } else {
        setStitchStatus("Fehler beim Starten!");
      }
    } catch {
      setStitchStatus("Fehler beim Starten!");
    }
  };
  const handleStitchingFinish = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/stitching/finish`, { method: "POST" });
      const data = await res.json();
      if (data.status === "finished") {
        setStitchingActive(false);
        if (data.stitched) {
          setStitchStatus(
            <>
              Stitching beendet!
              <br />
              <a
                href={`${BACKEND_URL}/${data.stitched}`}
                target="_blank"
                rel="noopener noreferrer"
                className="underline text-blue-700"
              >
                Gestitchtes Bild anzeigen
              </a>
            </>
          );
        } else {
          setStitchStatus("Stitching beendet!");
        }
      } else {
        setStitchStatus("Fehler beim Beenden!");
      }
    } catch {
      setStitchStatus("Fehler beim Beenden!");
    }
  };
  const handleStitchingCapture = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/stitching/capture`, { method: "POST" });
      const data = await res.json();
      setStitchStatus(data.status === "captured" ? "Foto aufgenommen!" : "Fehler beim Aufnehmen!");
    } catch {
      setStitchStatus("Fehler beim Aufnehmen!");
    }
  };
  const handleMoveStitching = async (x, y, z) => {
    await moveAxis(x, y, z, true);
  };

  /* ================== Video-AF Sweep ================== */
const startVideoAF_simple = async () => {
  if (!imgRef.current || !streamReady) {
    console.warn("[Video-AF] Stream noch nicht bereit…");
    return;
  }
  const img = imgRef.current;

  const zStart = -10000;   // Startposition
  const zEnd   = +10000;   // Endposition
  const totalMove = zEnd - zStart;  // 20000 µm
  const duration  = 30000;          // Sweep-Dauer nach vorne (25 s)
  const fps       = 9;
  const frameInterval = 1000 / fps;

  console.group("[Video-AF] Sweep gestartet");

  // 1. Vorpositionieren auf -10000
  console.log("➡️ Fahre zurück auf Start:", zStart, "µm");
  await moveAxis(0, 0, zStart);

  // 10 Sekunden warten, bis Motor sicher steht
  console.log("⏳ Warte 15 s, bis Motor stabil ist…");
  await sleep(15000);

  // 2. Sweep starten (20000 µm nach vorne in 25 s)
  console.log("➡️ Starte Sweep:", totalMove, "µm vorwärts in ~25 s");
  const t0 = nowMs();
  moveAxis(0, 0, totalMove);  // nicht awaiten

  const frames = [];
  let frameIdx = 0;

  while (!sweepCancelRef.current) {
    const t = nowMs() - t0;

    if (t >= duration) {
      console.log("⏹ Sweep beendet. Samples:", frames.length);
      if (!frames.length) return;

      // 3. Bestes Sample suchen
      let bestIdx = 0;
      for (let i = 1; i < frames.length; i++) {
        if (frames[i].score > frames[bestIdx].score) bestIdx = i;
      }
      const bestZ = frames[bestIdx].z;

      // 4. Rückfahrt von Endposition (+10000) zurück zum besten
      const deltaBack = bestZ - zEnd;
      console.log(`🟢 Bester Fokus bei Z=${bestZ.toFixed(1)} µm (Frame #${bestIdx})`);
      console.log(`↩️ Rückfahrt ΔZ=${deltaBack.toFixed(1)} µm`);

      await moveAxis(0, 0, deltaBack);
      await sleep(500);

      setAutofocusScore(frames[bestIdx].score);
      console.log("✅ Autofokus abgeschlossen. Score=", frames[bestIdx].score.toFixed(3));
      console.groupEnd();
      return;
    }

    // fixes Sampling
    const nextPlanned = t0 + (frameIdx + 1) * frameInterval;
    const delay = Math.max(0, nextPlanned - nowMs());
    if (delay > 0) await sleep(delay);

    const score = captureScoreFromImg(img, ROI_SCALE);
    if (score != null) {
      const z = zStart + (totalMove * (t / duration));
      frames.push({ z, score });
      setAutofocusScore(score);

      if (frameIdx % 20 === 0) {
        console.log(`[Video-AF] f#${frameIdx} t=${Math.round(t)}ms z≈${z.toFixed(1)}µm score=${score.toFixed(3)}`);
      }
    }
    frameIdx++;
  }
};

  /* ================== UI ================== */
  return (
    <div className="h-screen w-full overflow-hidden bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="mx-auto max-w-7xl h-full p-6 grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6">
        
        {/* Sidebar – nur hier scrollt */}
        <div className="h-full overflow-y-auto pr-0 space-y-4">
          
          <Panel
            title="Macroscope Steuerung"
            right={<Badge tone={streamReady ? "ok" : "warn"}>{streamReady ? "Stream bereit" : "Warte auf Stream…"}</Badge>}
          >
            <div className="grid grid-cols-2 gap-2">
              <Btn onClick={() => sendZoom("in")}  tone="primary">Zoom +</Btn>
              <Btn onClick={() => sendZoom("out")} tone="primary">Zoom –</Btn>

              <Btn onClick={runAutofocus} tone="success" disabled={isLoading}>
                {isLoading ? "AF läuft…" : "Autofokus"} 
              </Btn>
              <Btn onClick={startVideoAF_simple} tone="neutral" disabled={smartAf}>
                {smartAf ? "Video AF läuft…" : "Video Autofokus"}
              </Btn>
              <Btn onClick={handleSmartAutofocus} tone="teal" disabled={smartAfBusy}>
                {smartAfBusy ? "Smart AF läuft…" : "Smart Autofocus"}
              </Btn>
              <Btn onClick={handleNextGenAutofocus} tone="amber" disabled={nextgenBusy}>
                {nextgenBusy ? "NextGen AF läuft…" : "NextGen Autofocus"}
              </Btn>
            </div>
          </Panel>

          <Panel title="Manuelle Bewegung">
            <div className="grid grid-cols-2 gap-2">
              <Btn onClick={() => moveAxis(stepSize, 0, 0)} tone="primary">X +</Btn>
              <Btn onClick={() => moveAxis(-stepSize, 0, 0)} tone="primary">X –</Btn>
      
              <Btn onClick={() => moveAxis(0, stepSize, 0)} tone="success">Y +</Btn>
              <Btn onClick={() => moveAxis(0, -stepSize, 0)} tone="success">Y –</Btn>

              <Btn onClick={() => moveAxis(0, 0, stepSize)} tone="amber">Z +</Btn>
              <Btn onClick={() => moveAxis(0, 0, -stepSize)} tone="amber">Z –</Btn>
            </div>

            <div className="mt-3 flex items-center gap-2">
              <label className="text-sm">Schrittweite:</label>
              <input
                type="number"
                value={stepSize}
                onChange={(e) => setStepSize(Number(e.target.value))}
                className="border rounded-lg px-2 py-1 w-28 text-sm text-center"
              />
            </div>

            <div className="mt-3 flex items-center gap-2">
              <input
                type="number"
                name="x"
                value={coords.x}
                onChange={handleInputChange}
                placeholder="X"
                className="border rounded-lg px-2 py-1 w-24 text-center"
              />
              <input
                type="number"
                name="y"
                value={coords.y}
                onChange={handleInputChange}
                placeholder="Y"
                className="border rounded-lg px-2 py-1 w-24 text-center"
              />
              <input
                type="number"
                name="z"
                value={coords.z}
                onChange={handleInputChange}
                placeholder="Z"
                className="border rounded-lg px-2 py-1 w-24 text-center"
              />
            </div>

            <div className="mt-3 flex items-center gap-2">
              <Btn onClick={handleMoveInput} tone="primary">Bewegen</Btn>
            </div>

            <p className="mt-4 text-sm text-gray-700">
              <strong>Shortcuts:</strong>&nbsp;
              <span className="font-mono bg-gray-200 px-2 py-0.5 rounded">W/S</span> → Y,&nbsp;
              <span className="font-mono bg-gray-200 px-2 py-0.5 rounded">A/D</span> → X,&nbsp;
              <span className="font-mono bg-gray-200 px-2 py-0.5 rounded">Q/E</span> → Z
            </p>

          </Panel>

          <Panel
  title="Stitching"
  right={
    stitchingActive ? <Badge tone="ok">Aktiv</Badge> : <Badge tone="mute">Inaktiv</Badge>
  }
>
  {/* Start / Stop */}
  <div className="flex gap-2 mb-4">
    <Btn
      onClick={() => {
        handleStitchingStart();
        setStitchTurn("photo"); // beim Start immer mit Foto beginnen
      }}
      tone="success"
      disabled={stitchingActive}
      className="flex-1 h-12"
    >
      ▶ Start
    </Btn>
    <Btn
      onClick={handleStitchingFinish}
      tone="danger"
      disabled={!stitchingActive}
      className="flex-1 h-12"
    >
      ⏹ Beenden
    </Btn>
  </div>

  {/* D-Pad */}
  <div className="flex flex-col items-center gap-2">
    {/* Y+ */}
    <Btn
      onClick={async () => {
        await handleMoveStitching(0, stepY, 0);
        setStitchTurn("photo");
      }}
      tone="success"
      disabled={!stitchingActive || stitchTurn !== "move"}
      className="h-12 w-12"
      title="Y+"
    >
      ↑
    </Btn>

    {/* X- / Foto / X+ */}
    <div className="flex items-center gap-3">
      <Btn
        onClick={async () => {
          await handleMoveStitching(stepX, 0, 0);
          setStitchTurn("photo");
        }}
        tone="primary"
        disabled={!stitchingActive || stitchTurn !== "move"}
        className="h-12 w-12"
        title="X+"
      >
        ←
      </Btn>

      <Btn
        onClick={async () => {
          await handleStitchingCapture();
          setStitchTurn("move");
        }}
        tone="amber"
        disabled={!stitchingActive || stitchTurn !== "photo"}
        className="h-14 w-14 rounded-full text-lg"
        title="Foto aufnehmen"
      >
        📸
      </Btn>

      <Btn
        onClick={async () => {
          await handleMoveStitching(-stepX, 0, 0);
          setStitchTurn("photo");
        }}
        tone="primary"
        disabled={!stitchingActive || stitchTurn !== "move"}
        className="h-12 w-12"
        title="X–"
      >
        →
      </Btn>
    </div>

    {/* Y- */}
    <Btn
      onClick={async () => {
        await handleMoveStitching(0, -stepY, 0);
        setStitchTurn("photo");
      }}
      tone="success"
      disabled={!stitchingActive || stitchTurn !== "move"}
      className="h-12 w-12"
      title="Y–"
    >
      ↓
    </Btn>
  </div>

  {/* Status */}
  <div className="mt-4 min-h-[1.5em] text-sm text-blue-700 text-center font-medium">
    {stitchStatus}
  </div>
</Panel>


        </div>

        {/* Rechte Spalte – Live-Stream immer sichtbar */}
        <div className="h-full flex flex-col">
          <Panel title="Live-Stream" className="h-full flex flex-col !p-3">
            <div className="flex-1 overflow-hidden rounded-2xl">
              <img
                id="stream"
                ref={imgRef}
                alt="Live"
                className="w-full h-full object-contain border rounded-2xl shadow"
              />
            </div>
          </Panel>
        </div>

      </div>
    </div>
  );

}

export default App;
