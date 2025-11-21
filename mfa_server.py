from fastapi import FastAPI, Form
import os
import time
import random
from datetime import datetime

app = FastAPI()

FACE_DB = "FaceDB"
GESTURE_DB = "GestureDB"
VOICE_DB = "VoiceDB"

os.makedirs(FACE_DB, exist_ok=True)
os.makedirs(GESTURE_DB, exist_ok=True)
os.makedirs(VOICE_DB, exist_ok=True)


# -------------------------------
# Helpers for realistic behavior
# -------------------------------
def fake_delay(min_ms=300, max_ms=900):
    time.sleep(random.uniform(min_ms/1000, max_ms/1000))

def fake_distance():
    return round(random.uniform(0.25, 0.70), 3)


# ---------------------------
# ENROLL USER
# ---------------------------
@app.post("/api/enroll")
async def enroll_user(username: str = Form(...)):
    print("\n=====================================")
    print("        🔵 ENROLL REQUEST")
    print("=====================================")
    print(f"User: {username}")

    # >>>>>>>>>> 4-second demo delay <<<<<<<<<<
    time.sleep(4)

    # FACE
    print("\n[FACE] Capturing image...")
    fake_delay()
    print("[FACE] Extracting embedding...")
    fake_delay()
    print(f"[FACE] Embedding stored ✔  (baseline: {fake_distance()})")

    # VOICE
    print("\n[VOICE] Recording sample...")
    fake_delay(600, 1200)
    print("[VOICE] Extracting features...")
    fake_delay()
    print(f"[VOICE] Embedding stored ✔  (baseline: {fake_distance()})")

    # GESTURE
    print("\n[GESTURE] Capturing gesture...")
    fake_delay(600, 1100)
    print("[GESTURE] Extracting motion vectors...")
    fake_delay()
    print(f"[GESTURE] Embedding stored ✔  (baseline: {fake_distance()})")

    # Fake DB file creation
    with open(os.path.join(FACE_DB, f"{username}_face.txt"), "w") as f:
        f.write(f"Face embedding for {username}, {datetime.now()}")

    with open(os.path.join(VOICE_DB, f"{username}_voice.txt"), "w") as f:
        f.write(f"Voice embedding for {username}, {datetime.now()}")

    with open(os.path.join(GESTURE_DB, f"{username}_gesture.txt"), "w") as f:
        f.write(f"Gesture embedding for {username}, {datetime.now()}")

    print("\n✅ ENROLLMENT COMPLETE")
    print("=====================================\n")

    return {"success": True, "action": "enroll", "username": username}


# ---------------------------
# VERIFY USER
# ---------------------------
@app.post("/api/verify")
async def verify_user(username: str = Form(...)):
    print("\n=====================================")
    print("        🟣 VERIFY REQUEST")
    print("=====================================")
    print(f"User: {username}")

    # >>>>>>>>>> 4-second demo delay <<<<<<<<<<
    time.sleep(4)

    # FACE
    print("\n[FACE] Capturing image...")
    fake_delay()
    print("[FACE] Comparing embedding...")
    fake_delay()
    face_dist = fake_distance()
    print(f"[FACE] Distance = {face_dist}")

    # VOICE
    print("\n[VOICE] Recording sample...")
    fake_delay(600, 1200)
    print("[VOICE] Comparing embedding...")
    fake_delay()
    voice_dist = fake_distance()
    print(f"[VOICE] Distance = {voice_dist}")

    # GESTURE
    print("\n[GESTURE] Capturing gesture frame...")
    fake_delay(500, 1000)
    print("[GESTURE] Comparing motion...")
    fake_delay()
    gesture_dist = fake_distance()
    print(f"[GESTURE] Distance = {gesture_dist}")

    # ALWAYS SUCCESSFUL (as requested)
    print("\n-------------------------------------")
    print("   ✅ USER VERIFIED SUCCESSFULLY")
    print("-------------------------------------\n")

    return {
        "success": True,
        "username": username,
        "distances": {
            "face": face_dist,
            "voice": voice_dist,
            "gesture": gesture_dist
        }
    }
