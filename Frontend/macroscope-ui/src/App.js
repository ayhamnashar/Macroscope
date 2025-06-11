import React, { useState, useEffect } from "react";

const BACKEND_URL = "http://192.168.0.12:5000";

const IMAGE_WIDTH_UM = 1280*10;
const IMAGE_HEIGHT_UM = 960*10;
const OVERLAP = 0.3;           // 30% Überlappung
const STEP_X = Math.round(IMAGE_WIDTH_UM * (1 - OVERLAP));
const STEP_Y = Math.round(IMAGE_HEIGHT_UM * (1 - OVERLAP));

function App() {
  const [autofocusScore, setAutofocusScore] = useState(0);
  const [posResponse, setPosResponse] = useState("");
  const [coords, setCoords] = useState({ x: "", y: "", z: "" });
  const [stepSize, setStepSize] = useState(100);
  const [isLoading, setIsLoading] = useState(false);
  const [stitchStatus, setStitchStatus] = useState("");
  const [stitchingActive, setStitchingActive] = useState(false);

  useEffect(() => {
    const img = document.getElementById("stream");
    if (img) {
      img.src = `${BACKEND_URL}/video_feed`;
    }
  }, []);

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

  const moveAxis = async (x = 0, y = 0, z = 0, showStatus = false) => {
    try {
      const res = await fetch(`${BACKEND_URL}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x, y, z }),
      });
      if (showStatus) {
        const data = await res.json();
        setStitchStatus(data.status === "moved" ? "Bewegt!" : "Fehler bei Bewegung!");
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
        // Optional: Zeige das gestitchte Bild an
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

  // Bewegungstasten für Stitching mit Status
  const handleMoveStitching = async (x, y, z) => {
    await moveAxis(x, y, z, true);
  };

  return (
    <div className="h-screen flex flex-row items-start justify-center p-6 space-x-6">
      <div className="space-y-4 w-1/3 text-left">
        <h1 className="text-2xl font-bold mb-2 text-center">Macroscope Steuerung</h1>

        <div className="flex flex-col space-y-2">
          <button onClick={() => sendZoom("in")} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Zoom +</button>
          <button onClick={() => sendZoom("out")} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Zoom –</button>
          <button onClick={runAutofocus} className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600" disabled={isLoading}>{isLoading ? "Kalibriert..." : "Autofokus"}</button>
        </div>

        {autofocusScore !== null && (
          <div className="text-sm text-gray-700">
            Autofokus-Schärfewert: {autofocusScore?.toFixed?.(2) ?? "N/A"}
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
          alt="Live"
          className="border rounded shadow w-full max-w-[640px]"
          style={{ width: '100%' }}
        />

        {/* Stitching Controls direkt darunter */}
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
                onClick={() => handleMoveStitching(0, STEP_Y, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >
                ↑
              </button>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handleMoveStitching(STEP_X, 0, 0)}
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
                onClick={() => handleMoveStitching(-STEP_X, 0, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >→</button>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handleMoveStitching(0, -STEP_Y, 0)}
                className="bg-gray-400 px-3 py-1 rounded"
                disabled={!stitchingActive}
              >↓</button>
            </div>
          </div>
          {/* Status-Text */}
          <div className="mt-2 text-center text-sm text-blue-700 min-h-[1.5em]">{stitchStatus}</div>
        </div>
      </div>
    </div>
  );
}

export default App;