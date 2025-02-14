import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import threading
import queue
import time

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
frame_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)  # Limit queue size
model = None
camera_thread = None

def load_model():
    global model
    if model is None:
        model = YOLO("yolov8n.pt")  # Load the smallest YOLOv8 model
    return model

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Use default camera (0)
        # Set lower resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def process_frame(frame):
    # Ensure frame size is consistent
    frame = cv2.resize(frame, (640, 480))
    
    model = load_model()
    results = model(frame, conf=0.5)  # Higher confidence threshold
    
    # Draw only top 5 most confident detections
    for result in results:
        boxes = result.boxes
        # Sort by confidence and get top 5
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
    
    return frame

def camera_stream():
    global output_frame, frame_lock
    
    frame_skip = 2  # Process every nth frame
    frame_count = 0
    
    while True:
        camera = get_camera()
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        try:
            # Process frame and update the global frame
            processed_frame = process_frame(frame)
            
            with frame_lock:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Remove old frame
                    except queue.Empty:
                        pass
                frame_queue.put(processed_frame)
                
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
            
        time.sleep(0.01)  # Small delay to prevent overwhelming the system

def generate_frames():
    while True:
        try:
            # Get the latest frame
            frame = frame_queue.get(timeout=1.0)
            
            # Convert to JPEG with lower quality for streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            break

def start_camera():
    global camera_thread
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=camera_stream)
        camera_thread.daemon = True
        camera_thread.start()

@app.route('/')
def index():
    start_camera()  # Start camera when first accessing the page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Cleanup when the app is shutting down
@app.teardown_appcontext
def cleanup(exception=None):
    release_camera()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)