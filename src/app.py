from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
)
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "upload"
RESULT_FOLDER = "Models/ResultImages"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # 呼叫 main.py 並傳遞圖片路徑
        subprocess.run(["python", "main.py", file_path])

        # 假設結果圖片的名稱是 depth_output_原始圖片名稱.png
        result_image_path = f"depth_output_{file.filename}.png"

        return redirect(url_for("display_result", filename=result_image_path))


@app.route("/waiting", methods=["POST"])
def waiting():
    if "file" not in request.files:
        return redirect(url_for("upload_form"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("upload_form"))
    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # 呼叫 main.py 並傳遞圖片路徑
        subprocess.run(["python", "main.py", file_path])

        # 假設結果圖片的名稱是 depth_output_原始圖片名稱.png
        result_image_path = f"depth_output_{file.filename}.png"

        return redirect(url_for("display_result", filename=result_image_path))


@app.route("/result/<filename>")
def display_result(filename):
    return render_template("result.html", filename=filename)


@app.route("/uploads/<filename>")
def send_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
