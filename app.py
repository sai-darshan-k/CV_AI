from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import io
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Will auto-download pretrained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get image from mobile camera
    file = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    
    # Perform detection
    results = model(img)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            detections.append({
                'label': label,
                'confidence': conf,
                'coordinates': [x1, y1, x2, y2]
            })
    
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)