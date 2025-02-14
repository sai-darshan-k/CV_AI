import os
import cv2
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
LATEST_VIDEO = "uploads/latest_video.mp4"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model only when needed
model = None

def load_model():
    global model
    if model is None:
        model = YOLO("yolov8n.pt")
    return model

def process_frame(frame, target_size=(640, 480)):
    # Resize frame to reduce memory usage
    frame = cv2.resize(frame, target_size)
    return frame

def generate_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        model = load_model()
        
        # Process every nth frame to reduce memory usage
        frame_skip = 2
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Resize and process frame
            frame = process_frame(frame)
            
            # Run detection with smaller confidence threshold
            results = model(frame, conf=0.5)  # Increased confidence threshold
            
            # Draw only the most confident detections
            for result in results:
                boxes = result.boxes
                # Sort boxes by confidence and only show top 5
                conf_sorted_idx = boxes.conf.argsort(descending=True)[:5]
                
                for idx in conf_sorted_idx:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to JPEG with lower quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Clear some memory
            del results
            del buffer
            del frame_bytes
            
    except Exception as e:
        print(f"Error in generate_frames: {str(e)}")
    finally:
        if cap is not None:
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

        # Save the latest video
        file.save(LATEST_VIDEO)
        return render_template("video.html")

    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    if not os.path.exists(LATEST_VIDEO):
        return "No video uploaded", 400
    return Response(generate_frames(LATEST_VIDEO),
                   mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)  # Disable debug mode for production