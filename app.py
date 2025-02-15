from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from flask import request, jsonify

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = model(img, conf=0.25)
        
        # Get detection results
        detections = []
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': int(cls),
                'name': results[0].names[int(cls)]
            })
        
        return jsonify({'detections': detections})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')