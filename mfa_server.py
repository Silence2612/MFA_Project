import serial
import requests
import time

SERIAL_PORT = "COM8"   # Change to your COM port
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print("[*] Listening on", SERIAL_PORT)

while True:
    if ser.in_waiting:
        line = ser.readline().decode().strip()
        print("[Arduino] ", line)

        # Example command: ENROLL|username=vansh|mode=face
        parts = line.split("|")

        if len(parts) < 3:
            continue

        command = parts[0]
        username = parts[1].split("=")[1]
        mode = parts[2].split("=")[1]

        if command == "ENROLL":
            print("[*] Sending ENROLL to server...")
            r = requests.post("http://127.0.0.1:8000/api/enroll",
                              data={"username": username, "mode": mode})

            ser.write((r.text + "\n").encode())  # Send result back to Arduino
            print("[Server] ", r.text)

        elif command == "VERIFY":
            print("[*] Sending VERIFY to server...")
            r = requests.post("http://127.0.0.1:8000/api/verify",
                              data={"username": username, "mode": mode})

            ser.write((r.text + "\n").encode())
            print("[Server] ", r.text)
