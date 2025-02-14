import os
import cv2
from flask import Flask, render_template, request, Response
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
LATEST_VIDEO = "uploads/latest_video.mp4"  # Only store the latest video
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano model

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Delete previous file if it exists
        if os.path.exists(LATEST_VIDEO):
            os.remove(LATEST_VIDEO)

        # Save the latest video as "latest_video.mp4"
        file.save(LATEST_VIDEO)

        return render_template("video.html")  # Redirect to video streaming page

    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    if not os.path.exists(LATEST_VIDEO):
        return "No video uploaded", 400
    return Response(generate_frames(LATEST_VIDEO), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
