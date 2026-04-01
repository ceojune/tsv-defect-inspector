import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from analyzer import TSVDefectAnalyzer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB 제한

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

analyzer = TSVDefectAnalyzer()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "지원하지 않는 파일 형식입니다. (PNG, JPG, TIFF, BMP 지원)"}), 400

    # 파일을 메모리에서 바로 읽기 (Vercel은 디스크 쓰기 불가)
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "이미지를 읽을 수 없습니다."}), 400

    # 메모리 기반 분석 실행
    result = analyzer.analyze_in_memory(image)

    return jsonify(result)


if __name__ == "__main__":
    # 로컬 실행용
    app.run(debug=True, port=5000)
