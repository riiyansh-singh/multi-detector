"""
Multi-Detection System
- Sleep/Drowsiness Detection
- Fire Detection  
- Smile/Laugh Detection
Each with its own alarm sound!
"""

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import pygame
import time
import dlib
import cv2
import os

# ── CONFIG ──────────────────────────────────────────
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 20
SHAPE_PREDICTOR   = "shape_predictor_68_face_landmarks.dat"
SLEEP_ALARM       = "sounds/alarm_sleep.wav"
FIRE_ALARM        = "sounds/alarm_fire.wav"
SMILE_ALARM       = "sounds/alarm_smile.wav"

# dlib eye indices
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# ── STATE ────────────────────────────────────────────
sleep_alarm_on = False
fire_alarm_on  = False
smile_alarm_on = False
frame_counter  = 0
smile_counter  = 0

# ── PYGAME AUDIO ─────────────────────────────────────
pygame.mixer.init()
sleep_sound = pygame.mixer.Sound(SLEEP_ALARM)
fire_sound  = pygame.mixer.Sound(FIRE_ALARM)
smile_sound = pygame.mixer.Sound(SMILE_ALARM)

# ── EAR ──────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ── MOUTH ASPECT RATIO (smile/laugh) ─────────────────
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2],  mouth[10])
    B = dist.euclidean(mouth[4],  mouth[8])
    C = dist.euclidean(mouth[0],  mouth[6])
    return (A + B) / (2.0 * C)

# ── FIRE DETECTION ───────────────────────────────────
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,   150, 150])
    upper1 = np.array([15,  255, 255])
    lower2 = np.array([160, 150, 150])
    upper2 = np.array([180, 255, 255])
    mask1  = cv2.inRange(hsv, lower1, upper1)
    mask2  = cv2.inRange(hsv, lower2, upper2)
    mask   = cv2.bitwise_or(mask1, mask2)
    fire_pixels = cv2.countNonZero(mask)
    return fire_pixels > 2000, mask

# ── ALARM THREADS ─────────────────────────────────────
def play_sleep_alarm():
    global sleep_alarm_on
    sleep_sound.play(loops=-1)
    while sleep_alarm_on:
        time.sleep(0.1)
    sleep_sound.stop()

def play_fire_alarm():
    global fire_alarm_on
    fire_sound.play(loops=-1)
    while fire_alarm_on:
        time.sleep(0.1)
    fire_sound.stop()

def play_smile_alarm():
    global smile_alarm_on
    smile_sound.play(loops=-1)
    while smile_alarm_on:
        time.sleep(0.1)
    smile_sound.stop()

# ── MAIN ─────────────────────────────────────────────
def main():
    global sleep_alarm_on, fire_alarm_on, smile_alarm_on
    global frame_counter, smile_counter

    if not os.path.exists(SHAPE_PREDICTOR):
        print("[ERROR] shape_predictor_68_face_landmarks.dat not found!")
        print("Place it in: C:\\drowsiness_project\\")
        return

    print("[INFO] Loading models...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

    print("[INFO] Starting camera...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    print("[INFO] Running! Press Q to quit.\n")

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── FIRE DETECTION ──────────────────────────
        fire_detected, fire_mask = detect_fire(frame)
        if fire_detected:
            cv2.putText(frame, "FIRE DETECTED!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,200), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            if not fire_alarm_on:
                fire_alarm_on = True
                Thread(target=play_fire_alarm, daemon=True).start()
        else:
            if fire_alarm_on:
                fire_alarm_on = False

        # ── FACE DETECTION ──────────────────────────
        faces = detector(gray, 0)

        for face in faces:
            shape     = predictor(gray, face)
            shape     = face_utils.shape_to_np(shape)
            left_eye  = shape[L_START:L_END]
            right_eye = shape[R_START:R_END]
            mouth     = shape[M_START:M_END]

            # Draw eye contours
            lh = cv2.convexHull(left_eye)
            rh = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [lh], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rh], -1, (0, 255, 0), 1)

            # ── SLEEP DETECTION ─────────────────────
            left_EAR  = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            ear = (left_EAR + right_EAR) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (650, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSY! WAKE UP!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    if not sleep_alarm_on:
                        sleep_alarm_on = True
                        Thread(target=play_sleep_alarm, daemon=True).start()
            else:
                frame_counter = 0
                if sleep_alarm_on:
                    sleep_alarm_on = False

            # ── SMILE/LAUGH DETECTION ────────────────
            mar = mouth_aspect_ratio(mouth)
            cv2.putText(frame, f"MAR: {mar:.2f}", (650, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if mar > 0.6:
                smile_counter += 1
                if smile_counter >= 10:
                    cv2.putText(frame, "SMILE DETECTED!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    if not smile_alarm_on:
                        smile_alarm_on = True
                        Thread(target=play_smile_alarm, daemon=True).start()
            else:
                smile_counter = 0
                if smile_alarm_on:
                    smile_alarm_on = False

        # ── UI ──────────────────────────────────────
        cv2.rectangle(frame, (0,0), (frame.shape[1], 35), (30,30,30), -1)
        cv2.putText(frame, "MULTI DETECTOR  |  Sleep  Fire  Smile  |  Q to quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        cv2.imshow("Multi Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    sleep_alarm_on = False
    fire_alarm_on  = False
    smile_alarm_on = False
    vs.stop()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
