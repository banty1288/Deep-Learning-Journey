from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO(r"C:\Users\banty\runs\detect\train-11\weights\best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Predict
    results = model(frame, conf=0.5, verbose=False)

    # Draw detections
    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        gesture = model.names[cls]

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{gesture}: {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Hand Gesture Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press Q or ESC to exit
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()