import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Load known faces
known_face_encodings = []
known_face_names = []

image_extensions = {".jpg", ".jpeg", ".png"}

print("[INFO] Loading known faces...")

for file in os.listdir("known_faces"):
    ext = os.path.splitext(file)[1].lower()
    if ext not in image_extensions:
        print(f"[WARNING] Skipping non-image file: {file}")
        continue
    try:
        img_path = os.path.join("known_faces", file)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(file)[0])
            print(f"[INFO] Loaded encoding for {file}")
        else:
            print(f"[WARNING] No faces found in {file}")
    except Exception as e:
        print(f"[ERROR] Processing {file}: {e}")

# Function to mark attendance
def mark_attendance(name):
    with open("attendance.csv", "a+") as f:
        f.seek(0)
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]
        if name not in names:
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{dt}\n")

# Start webcam and process frames
print("[INFO] Starting camera. Press ESC to stop...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

        # Scale face locations back to original frame size
        top, right, bottom, left = [v * 4 for v in location]

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display output frame
    cv2.imshow("AI Attendance System", frame)

    # Exit on ESC key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
