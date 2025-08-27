import React, { useState, useEffect, useRef } from "react";

const BACKEND_URL = "http://192.168.0.12:5000";
// If you don't already have this:
const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

// === Fokus-Score Konfiguration ===
const ROI_SCALE = 0.55;          // 40–60% ausprobieren
const W_TENENGRAD = 0.40;
const W_LAPLACE   = 0.35;
const W_HF        = 0.25;
const DOG_SIGMA_SMALL = 0.8;     // DoG Bandpass (schneller FFT-Proxy)
const DOG_SIGMA_LARGE = 1.6;

function App() {
  const imgRef = useRef(null);
  const [streamReady, setStreamReady] = useState(false);

  const afCanvasRef = useRef(null);
  const afCtxRef = useRef(null);

  const [autofocusScore, setAutofocusScore] = useState(0);
  const [posResponse, setPosResponse] = useState("");
  const [coords, setCoords] = useState({ x: "", y: "", z: "" });
  const [stepSize, setStepSize] = useState(100);
  const [isLoading, setIsLoading] = useState(false);
  const [stitchStatus, setStitchStatus] = useState("");
  const [stitchingActive, setStitchingActive] = useState(false);
  const [imageDims, setImageDims] = useState(null);
  const [stepX, setStepX] = useState(0);
  const [stepY, setStepY] = useState(0);

  const [smartAfBusy, setSmartAfBusy] = React.useState(false);
  const [smartAfStatus, setSmartAfStatus] = React.useState(null);

  // NextGen AF UI state
  const [nextgenBusy, setNextgenBusy] = React.useState(false);
  const [nextgenStatus, setNextgenStatus] = React.useState(null);

  // Cancel-Ref für den Video-Sweep
  const sweepCancelRef = useRef(false);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    // wichtig für Canvas readback
    img.crossOrigin = "anonymous";

    const onLoad = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        setStreamReady(true);
      }
    };
    img.addEventListener("load", onLoad);

    // MJPEG-Stream setzen (einmal reicht)
    img.src = `${BACKEND_URL}/video_feed`;

    return () => {
      img.removeEventListener("load", onLoad);
    };
  }, []);

  useEffect(() => {
    const fetchDimensions = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/image_dimensions`);
        const data = await res.json();
        setImageDims(data);

        const overlap = 0.3;
        const sx = Math.round(data.width_um  * 2 * (1 - overlap));
        const sy = Math.round(data.height_um * 2 * (1 - overlap));
        setStepX(sx);
        setStepY(sy);
      } catch (err) {
        console.error("Fehler beim Laden der Bilddimensionen:", err);
      }
    };
    fetchDimensions();
  }, []);

  const nowMs = () => (performance?.now?.() ?? Date.now());
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // =================== Fokus-Score Einzelmetriken ===================

  // 3x3 Sobel, Tenengrad-Energie
  function tenengradScore(imageData, w, h) {
    const p = imageData.data;
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < p.length; i += 4, j++) {
      gray[j] = 0.299*p[i] + 0.587*p[i+1] + 0.114*p[i+2];
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

  // Laplacian-Varianz
  function laplacianVarScore(imageData, w, h) {
    const p = imageData.data;
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < p.length; i += 4, j++) {
      gray[j] = 0.299*p[i] + 0.587*p[i+1] + 0.114*p[i+2];
    }
    const kernel = [0,1,0, 1,-4,1, 0,1,0];
    const resp = new Float32Array(w * h);

    for (let y=1; y<h-1; y++) {
      for (let x=1; x<w-1; x++) {
        let v=0, kv=0;
        for (let yy=-1; yy<=1; yy++) {
          const row=(y+yy)*w;
          for (let xx=-1; xx<=1; xx++) {
            v += gray[row + (x+xx)] * kernel[kv++];
          }
        }
        resp[y*w + x] = v;
      }
    }

    let sum=0, sumSq=0, n=(w-2)*(h-2);
    for (let y=1; y<h-1; y++) {
      for (let x=1; x<w-1; x++) {
        const v = resp[y*w + x];
        sum += v; sumSq += v*v;
      }
    }
    const mean = sum/n;
    return Math.max(0, (sumSq/n) - mean*mean);
  }

  // Gaussian 1D Kernel
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

  // High-Frequency Energie via DoG
  function highFreqEnergy_DoG(imageData, w, h, sSmall = DOG_SIGMA_SMALL, sLarge = DOG_SIGMA_LARGE) {
    const p = imageData.data;
    const gray = new Float32Array(w * h);
    for (let i=0, j=0; i<p.length; i+=4, j++) {
      gray[j] = 0.299*p[i] + 0.587*p[i+1] + 0.114*p[i+2];
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

  // =================== ROI & Misch-Score ===================

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

      // Einzel-Scores
      const sTen = tenengradScore(imageData, rw, rh);
      const sLap = laplacianVarScore(imageData, rw, rh);
      const sHf  = highFreqEnergy_DoG(imageData, rw, rh);

      // Heuristische Log-Skalierung für vergleichbare Skalen
      const nTen = Math.log10(sTen + 1);
      const nLap = Math.log10(sLap + 1);
      const nHf  = Math.log10(sHf  + 1);

      const mixed = W_TENENGRAD*nTen + W_LAPLACE*nLap + W_HF*nHf;
      return mixed;
    } catch (e) {
      console.warn("captureScore error:", e?.message || e);
      return null;
    }
  }

  // Parabel-Fit um bestIdx
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
    const xv = -B / (2*A);
    return isFinite(xv) ? xv : x2;
  }

  // =================== Backend Calls ===================

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

  const runAutofocus = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/autofocus`, { method: "POST" });
      const data = await res.json();
      const score = data?.autofocus_score || 0;
      setAutofocusScore(score);
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
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}) // optionally pass tuning params here
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'ok') {
        const msg = data?.message || `HTTP ${res.status}`;
        setSmartAfStatus({ error: msg });
        return;
      }
      setSmartAfStatus({
        best_z_um: data.best_z_um,
        score: data.score,
        window: data.window,
        clipped: data.clipped
      });
    } catch (e) {
      setSmartAfStatus({ error: String(e) });
    } finally {
      setSmartAfBusy(false);
    }
  }

  // NextGen Autofocus handler
  async function handleNextGenAutofocus() {
    setNextgenBusy(true);
    setNextgenStatus(null);
    try {
      const res = await fetch(`${API_BASE}/autofocus/nextgen`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Optionally: pass field_id/neighbor_focus if you manage tiles
        body: JSON.stringify({
          // field_id: 'tile_42',
          // neighbor_focus: 1234.0,
          // start_pos_um: 0,
          // coarse_step_um: 15, coarse_n: 5, fine_step_um: 3, fine_n: 5, settle_s: 0.25
        })
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'ok') {
        const msg = data?.message || `HTTP ${res.status}`;
        setNextgenStatus({ error: msg });
        return;
      }
      setNextgenStatus({
        best_z_um: data.best_z_um,
        score: data.score,
        coverage: data.coverage,
        coarse_points: data.coarse_points,
        fine_points: data.fine_points
      });
    } catch (e) {
      setNextgenStatus({ error: String(e) });
    } finally {
      setNextgenBusy(false);
    }
  }

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
          data.status === "moving" || data.status === "moved"
            ? "Bewegt!"
            : "Fehler bei Bewegung!"
        );
      }
    } catch (err) {
      if (showStatus) setStitchStatus("Fehler bei Bewegung!");
      console.error("Move-Fehler:", err);
    }
  };

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
    moveAxis(Number(coords.x), Number(coords.y), Number(coords.z));
  };

  // --- Stitching Controls ---
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
    } catch (err) {
      setStitchStatus("Fehler beim Starten!");
    }
  };

  const handleStitchingFinish = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/stitching/finish`, { method: "POST" });
      const data = await res.json();
      if (data.status === "finished") {
        setStitchStatus("Stitching beendet!");
        setStitchingActive(false);
        if (data.stitched) {
          setStitchStatus(
            <>
              Stitching beendet!<br />
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
        }
      } else {
        setStitchStatus("Fehler beim Beenden!");
      }
    } catch (err) {
      setStitchStatus("Fehler beim Beenden!");
    }
  };

  const handleStitchingCapture = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/stitching/capture`, { method: "POST" });
      const data = await res.json();
      setStitchStatus(data.status === "captured" ? "Foto aufgenommen!" : "Fehler beim Aufnehmen!");
    } catch (err) {
      setStitchStatus("Fehler beim Aufnehmen!");
    }
  };

  const handleMoveStitching = async (x, y, z) => {
    await moveAxis(x, y, z, true);
  };

  // =================== Video-Autofokus Sweep ===================

  const startVideoAF_simple = async () => {
    if (!imgRef.current || !streamReady) {
      console.warn("Stream noch nicht bereit…");
      return;
    }
    const img = imgRef.current;

    // Parameter
    const zStart = -10000;         // µm
    const zEnd   = +10000;         // µm
    const totalMove = zEnd - zStart;   // 20000 µm
    const duration = 25000;        // ms (geschätzte Fahrzeit)
    const fps = 9;                 // Messrate
    const frameInterval = 1000 / fps; // <<< FIX

    // evtl. laufenden Sweep abbrechen
    sweepCancelRef.current = true;
    await new Promise(r => setTimeout(r, 10));
    sweepCancelRef.current = false;

    console.log(`➡️ Schritt 1: Z ${zStart} (relativ) vorpositionieren`);
    await moveAxis(0, 0, zStart);
    await sleep(300); // kurze Beruhigung

    console.log(`➡️ Schritt 2: Starte Vorwärtsfahrt ${totalMove} µm in ~${(duration/1000)|0}s`);
    const t0 = nowMs();
    // Start Bewegung ohne await, damit parallel gemessen werden kann
    moveAxis(0, 0, totalMove);

    const frames = [];
    let frameIdx = 0;

    while (!sweepCancelRef.current) {
      const t = nowMs() - t0;

      if (t >= duration) {
        console.log(`⏹ Sweep fertig. Samples: ${frames.length}`);
        if (!frames.length) return;

        // Bestes Sample bestimmen
        let bestIdx = 0;
        for (let i = 1; i < frames.length; i++) {
          if (frames[i].score > frames[bestIdx].score) bestIdx = i;
        }

        const bestZ = parabolicFitZ(frames, bestIdx);
        const deltaBack = bestZ - zEnd;

        console.log(`🟢 Peak bei Z≈${bestZ.toFixed(1)} µm (Best-Frame #${bestIdx})`);
        console.log(`↩️ Rückfahrt ΔZ=${deltaBack.toFixed(1)} µm`);
        await moveAxis(0, 0, deltaBack);
        await sleep(200);
        setAutofocusScore(frames[bestIdx].score);
        return;
      }

      // fixes Sampling-Intervall
      const nextPlanned = t0 + (frameIdx + 1) * frameInterval;
      const delay = Math.max(0, nextPlanned - nowMs());
      if (delay > 0) await sleep(delay);

      const score = captureScoreFromImg(img, ROI_SCALE);
      if (score != null) {
        const z = zStart + (totalMove * ((nowMs() - t0) / duration));
        frames.push({ z, score });
        setAutofocusScore(score);
      }

      frameIdx++;
    }
  };

  // Optional: Cancel-Funktion, falls du einen Button ergänzen willst
  const cancelVideoAF = () => {
    sweepCancelRef.current = true;
  };

  // =================== UI ===================

  return (
    <div className="h-screen flex flex-row items-start justify-center p-6 space-x-6">
      <div className="space-y-4 w-1/3 text-left">
        <h1 className="text-2xl font-bold mb-2 text-center">Macroscope Steuerung</h1>

        <div className="flex flex-col space-y-2">
          <button onClick={() => sendZoom("in")} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Zoom +</button>
          <button onClick={() => sendZoom("out")} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Zoom –</button>
          <button onClick={runAutofocus} className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600" disabled={isLoading}>
            {isLoading ? "Kalibriert..." : "Autofokus"}
          </button>
          <button onClick={startVideoAF_simple} className="bg-gray-700 text-white px-4 py-2 rounded hover:bg-gray-800">
            🎥 Video-Autofokus
          </button>
          <div className="flex flex-col">
            <button
              onClick={handleSmartAutofocus}
              disabled={smartAfBusy}
              className="bg-teal-600 text-white px-4 py-2 rounded hover:bg-teal-700 disabled:opacity-60 w-full"
            >
              {smartAfBusy ? 'Smart AF läuft…' : 'Smart Autofocus'}
            </button>
            {smartAfStatus && (
              <span className="text-sm mt-1">
                {smartAfStatus.error
                  ? `Fehler: ${smartAfStatus.error}`
                  : `Z=${smartAfStatus.best_z_um.toFixed(1)} µm (Score≈${smartAfStatus.score.toFixed(2)})`}
              </span>
            )}
          </div>
          
          {/* NextGen Autofocus */}
          <div className="flex flex-col mt-2">
            <button
              onClick={handleNextGenAutofocus}
              disabled={nextgenBusy}
              className="bg-amber-600 text-white px-4 py-2 rounded hover:bg-amber-700 disabled:opacity-60 w-full"
            >
              {nextgenBusy ? 'NextGen AF läuft…' : 'NextGen Autofocus'}
            </button>
            {nextgenStatus && (
              <span className="text-sm mt-1">
                {nextgenStatus.error
                  ? `Fehler: ${nextgenStatus.error}`
                  : `Z=${nextgenStatus.best_z_um.toFixed(1)} µm (Score≈${nextgenStatus.score.toFixed(2)} | cov≈${(nextgenStatus.coverage ?? 0).toFixed(2)})`}
              </span>
            )}
          </div>
          {/* <button onClick={cancelVideoAF} className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600">AF abbrechen</button> */}
        </div>

        {autofocusScore !== null && (
          <div className="text-sm text-gray-700">
            Autofokus-Schärfewert: {autofocusScore?.toFixed?.(3) ?? "N/A"}
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 w-full mt-4">
          <button onClick={() => moveAxis(stepSize, 0, 0)} className="bg-indigo-500 text-white px-2 py-1 rounded hover:bg-indigo-600">X+</button>
          <button onClick={() => moveAxis(-stepSize, 0, 0)} className="bg-indigo-500 text-white px-2 py-1 rounded hover:bg-indigo-600">X–</button>
          <button onClick={() => moveAxis(0, stepSize, 0)} className="bg-indigo-500 text-white px-2 py-1 rounded hover:bg-indigo-600">Y+</button>
          <button onClick={() => moveAxis(0, -stepSize, 0)} className="bg-indigo-500 text-white px-2 py-1 rounded hover:bg-indigo-600">Y–</button>
          <button onClick={() => moveAxis(0, 0, stepSize)} className="bg-indigo-500 text-white px-2 py-1 rounded hover:bg-indigo-600">Z+</button>
          <button onClick={() => moveAxis(0, 0, -stepSize)} className="bg-indigo-500 text-white px-2 py-1 rounded hover:bg-indigo-600">Z–</button>
        </div>

        <div className="mt-6 space-y-3">
          <div className="space-x-2">
            <input type="number" name="x" value={coords.x} onChange={handleInputChange} placeholder="X" className="border p-1 w-20 text-center rounded" />
            <input type="number" name="y" value={coords.y} onChange={handleInputChange} placeholder="Y" className="border p-1 w-20 text-center rounded" />
            <input type="number" name="z" value={coords.z} onChange={handleInputChange} placeholder="Z" className="border p-1 w-20 text-center rounded" />
            <button onClick={handleMoveInput} className="bg-yellow-500 text-white px-3 py-1 rounded hover:bg-yellow-600">Bewegen</button>
          </div>

          <div className="flex items-center space-x-2">
            <label className="text-sm">Schrittweite:</label>
            <input
              type="number"
              value={stepSize}
              onChange={(e) => setStepSize(Number(e.target.value))}
              className="border p-1 w-24 text-center rounded"
            />
          </div>

          <button onClick={getPosition} className="bg-purple-500 text-white px-3 py-1 rounded hover:bg-purple-600">Position abfragen</button>
          {posResponse && (
            <div className="mt-2 p-2 border rounded bg-gray-100 text-sm text-gray-800">
              Position: {posResponse}
            </div>
          )}
        </div>
      </div>

      <div className="w-2/3 mx-auto flex flex-col items-center">
        <img
          id="stream"
          ref={imgRef}
          alt="Live"
          className="border rounded shadow w-full max-w-[640px]"
          style={{ width: '100%' }}
        />

        {/* Stitching Controls */}
        <div className="flex flex-col items-center mt-4 space-y-2">
          <div className="flex space-x-2 mb-2">
            <button
              onClick={handleStitchingStart}
              className="bg-green-500 text-white px-3 py-1 rounded"
              disabled={stitchingActive}
            >
              Stitching Start
            </button>
            <button
              onClick={handleStitchingFinish}
              className="bg-red-500 text-white px-3 py-1 rounded hover:bg-red-700"
              disabled={!stitchingActive}
            >
              Stitching Beenden
            </button>
          </div>
          <div className="flex flex-col items-center">
            <div className="flex space-x-2">
              <button
                onClick={() => handleMoveStitching(0, stepY, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >
                ↑
              </button>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handleMoveStitching(stepX, 0, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >←</button>
              <button
                onClick={handleStitchingCapture}
                className="bg-blue-500 text-white px-3 py-1 rounded"
                disabled={!stitchingActive}
              >
                Foto aufnehmen
              </button>
              <button
                onClick={() => handleMoveStitching(-stepX, 0, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >→</button>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handleMoveStitching(0, -stepY, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >↓</button>
            </div>
          </div>
          <div className="mt-2 text-center text-sm text-blue-700 min-h-[1.5em]">{stitchStatus}</div>
        </div>

        {imageDims && (
          <div className="text-sm text-gray-600 mt-2">
            Bildgröße: {imageDims.width_px}×{imageDims.height_px} px<br />
            (ca. {imageDims.width_um} × {imageDims.height_um} µm)
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
