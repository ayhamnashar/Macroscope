import serial
import time

class MotorController:
    def __init__(self, port='/dev/serial0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        time.sleep(2)  # warten bis bereit

    def send_command(self, command):
        with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
            time.sleep(2)  # Warte, bis Verbindung bereit ist
            ser.write((command + '\n').encode())
            print(f"Gesendet: {command}")
            response = ser.readline().decode().strip()
            print(f"Antwort: {response}")
            return response

    def move(self, x=0, y=0, z=0):
        command = f"mr {x} {y} {z}"
        return self.send_command(command)

    def get_position(self):
        return self.send_command("p?")

    def release_motors(self):
        return self.send_command("release")
