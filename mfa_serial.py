import serial
import requests
import time

ser = serial.Serial("COM8", 9600, timeout=0.5)
print("[*] Listening for commands...")

while True:
    raw = ser.readline()
    if not raw:
        continue

    try:
        line = raw.decode(errors="ignore").strip()
    except:
        continue

    if not line:
        continue

    print("[Arduino]", repr(line))

    # Normalize
    line = line.replace("  ", " ").strip()

    # Must contain ":"
    if ":" not in line:
        print("Skipping invalid line")
        continue

    action, username = [x.strip() for x in line.split(":", 1)]

    # Normalize action
    action = action.lower()
    username = username.strip()

    # small debounce
    time.sleep(0.1)

    # ----------- HANDLE COMMANDS -----------
    if action == "enroll":
        print("[*] Calling ENROLL endpoint...")
        r = requests.post("http://127.0.0.1:8000/api/enroll",
                          data={"username": username})

    elif action in ("authenticate", "verify", "auth"):
        print("[*] Calling VERIFY endpoint...")
        r = requests.post("http://127.0.0.1:8000/api/verify",
                          data={"username": username})
    else:
        print("Invalid command:", action)
        continue

    print("[FastAPI]", r.text)

    # Send result back to Arduino
    ser.write((r.text + "\n").encode())

    # ----------- REFRESH FOR NEXT COMMAND -----------
    ser.reset_input_buffer()    # Clear any unread serial data
    ser.reset_output_buffer()   # Ensure clean port state
    time.sleep(0.2)             # Give Arduino time to send next
