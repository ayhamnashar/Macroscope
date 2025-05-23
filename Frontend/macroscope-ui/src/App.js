import React, { useState, useEffect } from "react";

const BACKEND_URL = "http://192.168.253.170:5000";


function App() {
  const [autofocusScore, setAutofocusScore] = useState(null);
  const [posResponse, setPosResponse] = useState("");
  const [coords, setCoords] = useState({ x: "", y: "", z: "" });
  const [stepSize, setStepSize] = useState(100);


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
    try {
      const res = await fetch(`${BACKEND_URL}/autofocus`, { method: "POST" });
      const data = await res.json();
      setAutofocusScore(data.autofocus_score);
    } catch (err) {
      console.error("Autofokus-Fehler:", err);
    }
  };

  const moveAxis = async (x = 0, y = 0, z = 0) => {
    try {
      await fetch(`${BACKEND_URL}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x, y, z }),
      });
    } catch (err) {
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

  return (
    <div className="h-screen flex flex-row items-start justify-center p-6 space-x-6">
      <div className="space-y-4 w-1/3 text-left">
        <h1 className="text-2xl font-bold mb-2 text-center">Macroscope Steuerung</h1>

        <div className="flex flex-col space-y-2">
          <button onClick={() => sendZoom("in")} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Zoom +</button>
          <button onClick={() => sendZoom("out")} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">Zoom –</button>
          <button onClick={runAutofocus} className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">Autofokus</button>
        </div>

        {autofocusScore !== null && (
          <div className="text-sm text-gray-700">
            Autofokus-Schärfewert: {autofocusScore.toFixed(2)}
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

      <div className="w-2/3">
        <img
          id="stream"
          alt="Live"
          className="border rounded shadow w-full max-w-[640px]"
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}

export default App;
