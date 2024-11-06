from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load your trained YOLOv11 model
model = YOLO("train_model_yolo_detection-3.pt")

# Initialize the camera
cap = cv2.VideoCapture(1)


def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not grabbed correctly, break the loop
        if not ret:
            break

        # Use the model to perform inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
