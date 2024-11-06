from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)

# Load your trained YOLOv11 model
model = YOLO("train_model_yolo_detection-2.pt")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize a counter for labels
label_counter = Counter()

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not grabbed correctly, break the loop
        if not ret:
            break

        # Use the model to perform inference on the frame
        results = model(frame)

        # Update the label counter
        for result in results:
            label_counter.update(result.names)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/label_data')
def label_data():
    return jsonify(label_counter)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)