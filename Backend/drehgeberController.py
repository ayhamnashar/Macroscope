from motorController import MotorController
from config import *
import RPi.GPIO as GPIO
import time
import threading

motor = MotorController()

z_delta = 0
last_rotation_time = 0
lock = threading.Lock()

def move_worker():
    global z_delta, last_rotation_time
    while True:
        time.sleep(0.05)
        lock.acquire()
        idle_time = time.time() - last_rotation_time
        if z_delta != 0 and idle_time > MOVEMENT_DELAY:
            steps = z_delta * STEP_SIZE
            print(f"➡️ Starte Bewegung: Z {steps}")
            motor.move(z=steps)
            z_delta = 0
        lock.release()

def run_drehgeber():
    global z_delta, last_rotation_time
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ENCODER_Z_CLK, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(ENCODER_Z_DT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(ENCODER_Z_SW, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    last_clk_state = GPIO.input(ENCODER_Z_CLK)
    last_button_state = GPIO.input(ENCODER_Z_SW)
    last_button_time = 0

    threading.Thread(target=move_worker, daemon=True).start()

    try:
        while True:
            clk_state = GPIO.input(ENCODER_Z_CLK)
            dt_state = GPIO.input(ENCODER_Z_DT)

            if last_clk_state == 1 and clk_state == 0:
                lock.acquire()
                z_delta += 1 if dt_state == 1 else -1
                last_rotation_time = time.time()
                print(f"↻↺ Drehung Z: {z_delta}")
                lock.release()

            last_clk_state = clk_state

            button_state = GPIO.input(ENCODER_Z_SW)
            now = time.time()
            if button_state == GPIO.LOW and last_button_state == GPIO.HIGH:
                if now - last_button_time > 0.2:
                    print("🔘 Button gedrückt - Autofokus")
                    last_button_time = now

            last_button_state = button_state
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("🛑 Beendet.")
    finally:
        GPIO.cleanup()