#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from motorController import MotorController
from config import *
import RPi.GPIO as GPIO
import time
import threading

# =========================
# Einstellungen
# =========================
# Schrittweite und Verzögerung kommen aus config.py:
# STEP_SIZE, MOVEMENT_DELAY
# Pin-Definitionen in config.py:
# ENCODER_X_CLK, ENCODER_X_DT, ENCODER_X_SW, ...
# ENCODER_Y_CLK, ENCODER_Y_DT, ENCODER_Y_SW
# ENCODER_Z_CLK, ENCODER_Z_DT, ENCODER_Z_SW

DEBOUNCE_S = 0.2  # Entprellzeit nur für Taster (optional)

# =========================
# Globale Zustände
# =========================
motor = MotorController()

# Dreh-Delta-Puffer pro Achse (wie viele "Ticks" akkumuliert wurden)
deltas = {"x": 0, "y": 0, "z": 0}

# Letzte Zeit, zu der eine Drehung registriert wurde (für MOVEMENT_DELAY)
last_rot = {"x": 0.0, "y": 0.0, "z": 0.0}

# Letzte gelesene Pegel an CLK/BTN (für Flankenerkennung)
last_clk = {}
last_btn = {}
last_btn_time = {"x": 0.0, "y": 0.0, "z": 0.0}

# Thread-Schutz
lock = threading.Lock()

# Pin-Mapping kompakt
AXES = {
    "x": {"CLK": ENCODER_X_CLK, "DT": ENCODER_X_DT, "SW": ENCODER_X_SW},
    "y": {"CLK": ENCODER_Y_CLK, "DT": ENCODER_Y_DT, "SW": ENCODER_Y_SW},
    "z": {"CLK": ENCODER_Z_CLK, "DT": ENCODER_Z_DT, "SW": ENCODER_Z_SW},
}


# =========================
# Worker: führt verzögerte Bewegungen aus
# =========================
def move_worker():
    """
    Liest regelmäßig die akkumulierten Dreh-Impulse und triggert Motorbewegungen,
    wenn seit der letzten Drehung mehr als MOVEMENT_DELAY vergangen ist.
    """
    while True:
        time.sleep(0.05)
        with lock:
            now = time.time()
            for axis in ("x", "y", "z"):
                if deltas[axis] != 0 and (now - last_rot[axis]) > MOVEMENT_DELAY:
                    steps = deltas[axis] * STEP_SIZE
                    if steps != 0:
                        print(f"➡️ Starte Bewegung: {axis.upper()} {steps:+d}")
                        try:
                            motor.move(**{axis: steps})
                        except Exception as e:
                            print(f"❌ Motorbewegung {axis.upper()} fehlgeschlagen: {e}")
                    deltas[axis] = 0


# =========================
# Haupt-Loop: Polling von Encodern + Tastern
# =========================
def run_drehgeber():
    GPIO.setmode(GPIO.BCM)

    # Pins setzen und Startpegel merken
    for a, p in AXES.items():
        GPIO.setup(p["CLK"], GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(p["DT"],  GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(p["SW"],  GPIO.IN, pull_up_down=GPIO.PUD_UP)

        last_clk[a] = GPIO.input(p["CLK"])
        last_btn[a] = GPIO.input(p["SW"])

    # Worker starten
    threading.Thread(target=move_worker, daemon=True).start()

    try:
        while True:
            time.sleep(0.001)
            now = time.time()

            # Alle Achsen abfragen
            for axis, pins in AXES.items():
                clk = GPIO.input(pins["CLK"])
                dt  = GPIO.input(pins["DT"])
                btn = GPIO.input(pins["SW"])  # PUD_UP => gedrückt == LOW

                # --- Dreh-Impuls: auf fallender CLK-Flanke auswerten
                if last_clk[axis] == GPIO.HIGH and clk == GPIO.LOW:
                    # Drehen funktioniert OHNE Tastendruck:
                    direction = 1 if dt == GPIO.HIGH else -1
                    with lock:
                        deltas[axis] += direction
                        last_rot[axis] = now
                        print(f"↻↺ Drehung {axis.upper()}: Δ={deltas[axis]}")
                last_clk[axis] = clk

                # --- Taster optional (z. B. Kurz-Druck-Aktion)
                # Flanken + Debounce
                if btn == GPIO.LOW and last_btn[axis] == GPIO.HIGH:
                    # falling edge (pressed)
                    if (now - last_btn_time[axis]) > DEBOUNCE_S:
                        last_btn_time[axis] = now
                        print(f"🔘 {axis.upper()} gedrückt")
                        # TODO: Hier optional eine Aktion beim Druck starten
                        # z.B.: if axis == "z": autofocus()
                elif btn == GPIO.HIGH and last_btn[axis] == GPIO.LOW:
                    # rising edge (released)
                    pass  # aktuell keine Release-Aktion
                last_btn[axis] = btn

    except KeyboardInterrupt:
        print("🛑 Beendet durch Benutzer.")
    finally:
        GPIO.cleanup()
        print("🧹 GPIO sauber freigegeben.")

def diagnose_pins():
    GPIO.setmode(GPIO.BCM)
    for a, p in AXES.items():
        GPIO.setup(p["CLK"], GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(p["DT"],  GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(p["SW"],  GPIO.IN, pull_up_down=GPIO.PUD_UP)
        last_clk[a] = GPIO.input(p["CLK"])

    print("Drehe SLOW, NICHT drücken. Erwartet: CLK toggelt, DT bestimmt Richtung.")
    try:
        while True:
            time.sleep(0.001)
            for axis, pins in AXES.items():
                clk = GPIO.input(pins["CLK"])
                dt  = GPIO.input(pins["DT"])
                if last_clk[axis] == GPIO.HIGH and clk == GPIO.LOW:
                    dirn = 1 if dt == GPIO.HIGH else -1
                    print(f"{axis.upper()} Flanke, DT={dt}, Richtung={'+' if dirn>0 else '-'}")
                last_clk[axis] = clk
    except KeyboardInterrupt:
        GPIO.cleanup()

# =========================
# Script-Entry
# =========================
if __name__ == "__main__":
    print("🟢 Drehgeber-Reader startet (Free-Turn ohne Tasterpflicht)...")
    print(f"STEP_SIZE={STEP_SIZE}, MOVEMENT_DELAY={MOVEMENT_DELAY}s")
    run_drehgeber()
