import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import threading
import queue
import time
import os

app = Flask(__name__)

# Global variables
camera = None
frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
model = None
camera_thread = None
is_running = False

def load_model():
    global model
    if model is None:
        model = YOLO("yolov8n.pt", task='detect')
    return model

def process_frame(frame, model):
    # Reduce frame size significantly
    frame = cv2.resize(frame, (320, 240))
    
    results = model(frame, conf=0.6, verbose=False)  # Increased confidence threshold
    
    # Draw only top 3 most confident detections
    for result in results:
        boxes = result.boxes
        conf_sorted_idx = boxes.conf.argsort(descending=True)[:3]
        
        for idx in conf_sorted_idx:
            box = boxes[idx]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = f"{model.names[cls]}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

def camera_stream():
    global is_running
    model = load_model()
    frame_skip = 3  # Process every 3rd frame
    frame_count = 0
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    try:
        while is_running:
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            try:
                processed_frame = process_frame(frame, model)
                
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(processed_frame)
                
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue
                
            time.sleep(0.1)  # Reduce CPU usage
            
    finally:
        cap.release()
        is_running = False

def generate_frames():
    timeout_counter = 0
    max_timeouts = 5  # Maximum number of consecutive timeouts
    
    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
            timeout_counter = 0  # Reset counter on successful frame
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except queue.Empty:
            timeout_counter += 1
            if timeout_counter >= max_timeouts:
                print("Too many consecutive timeouts, stopping stream")
                break
            continue
        except Exception as e:
            print(f"Frame generation error: {str(e)}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream')
def start_stream():
    global is_running, camera_thread
    if not is_running:
        is_running = True
        camera_thread = threading.Thread(target=camera_stream)
        camera_thread.daemon = True
        camera_thread.start()
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream')
def stop_stream():
    global is_running
    is_running = False
    return '', 204

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)