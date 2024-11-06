from flask import Flask, request, render_template
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from Models import LuggageProcessor as lp

app = Flask(__name__)
model = lp.LuggageProcessor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        image_path = f"uploads/{file.filename}"
        file.save(image_path)
        model.depth_test(image_path=image_path)
        return "Image processed successfully"


if __name__ == "__main__":
    app.run(debug=True)
