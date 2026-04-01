"""
TSV 결함 탐지 엔진 v2
=====================
SEM/FIB 단면(cross-section) 이미지에서 4가지 TSV 결함을 탐지하는 모듈.

결함 유형:
  1. Open Defect (TSV-RDL)  — TSV-RDL 접합부의 갭
  2. Open Defect (Bump)     — 범프-다이 접합부의 갭
  3. Short Defect (Bump)    — 인접 범프 간 브릿지(단락)
  4. Void Formation (TSV)   — TSV 내부 공동

OpenCV 기반 규칙 탐지이며, 나중에 YOLO 등 딥러닝 모델로 교체 가능.
"""

import os
import cv2
import numpy as np


# 결함 종류별 시각화 색상 (BGR)
DEFECT_COLORS = {
    "Open Defect (TSV-RDL)": (0, 0, 255),      # 빨강
    "Open Defect (Bump)": (0, 165, 255),        # 주황
    "Short Defect (Bump)": (0, 255, 255),       # 노랑
    "Void Formation (TSV)": (255, 0, 255),      # 보라
}

DEFECT_DESCRIPTIONS = {
    "Open Defect (TSV-RDL)": (
        "TSV와 RDL 사이 구리 충전이 불완전해 갭이 발생한 상태예요. "
        "직렬 커패시턴스와 고저항 경로가 형성되어 신호 전달에 문제가 생겨요."
    ),
    "Open Defect (Bump)": (
        "범프 높이 불균일로 범프가 상대 다이에 닿지 못해 물리적 접촉이 끊긴 상태예요. "
        "신호 경로가 전기적으로 단절돼요."
    ),
    "Short Defect (Bump)": (
        "본딩 시 범프가 수평 팽창해 인접 범프와 맞닿아 의도치 않은 전기적 경로가 형성된 상태예요. "
        "임피던스 저하와 불필요한 발열이 생겨요."
    ),
    "Void Formation (TSV)": (
        "구리 충전 속도 불량이나 열응력으로 TSV 내부에 빈 공간이 생긴 상태예요. "
        "저항값 변화를 유발하고 기계적·열적 스트레스에서 치명적이에요."
    ),
}


class TSVDefectAnalyzer:
    """SEM/FIB 단면 이미지에서 TSV 결함을 탐지하는 분석기."""

    def __init__(self):
        self.min_defect_area = 30
        self.max_defect_area = 80000

    # ================================================================
    # 전처리
    # ================================================================

    def preprocess(self, image):
        """그레이스케일 변환 → CLAHE 대비 향상 → 노이즈 제거."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # CLAHE — SEM 이미지의 낮은 대비를 향상
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 가우시안 블러 — 노이즈 제거
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

        return gray, enhanced, denoised

    def _find_layer_regions(self, gray):
        """
        SEM 단면 이미지에서 수평 레이어 경계를 추정.
        밝기 프로파일의 변화로 TSV/RDL/Bump/Si Sub 영역을 대략 나눈다.
        """
        h, w = gray.shape
        # 수평 밝기 프로파일 (각 행의 평균 밝기)
        row_profile = np.mean(gray, axis=1)

        # 프로파일을 스무딩
        kernel_size = max(h // 20, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(row_profile.reshape(-1, 1), (1, kernel_size), 0).flatten()

        # 밝기 변화가 큰 지점 = 레이어 경계
        gradient = np.abs(np.diff(smoothed))
        threshold = np.mean(gradient) + 1.5 * np.std(gradient)
        boundaries = np.where(gradient > threshold)[0]

        # 경계를 클러스터링 (가까운 경계 합치기)
        if len(boundaries) == 0:
            return {"top": 0, "upper_third": h // 3, "lower_third": 2 * h // 3, "bottom": h}

        clustered = [boundaries[0]]
        for b in boundaries[1:]:
            if b - clustered[-1] > h * 0.05:
                clustered.append(b)

        # 상/중/하 영역으로 나누기
        if len(clustered) >= 2:
            return {
                "top": 0,
                "upper_third": clustered[0],
                "lower_third": clustered[-1],
                "bottom": h,
            }
        else:
            return {"top": 0, "upper_third": h // 3, "lower_third": 2 * h // 3, "bottom": h}

    # ================================================================
    # 결함 ① Open Defect (TSV-RDL)
    # ================================================================

    def detect_open_defect_tsv_rdl(self, gray, denoised, layers):
        """
        TSV 상단-RDL 접합부에서 어두운 갭(gap)을 탐지.
        - 위치: 이미지 상단 1/3 영역
        - 특징: TSV 구조(세로로 밝은 기둥) 바로 위에 있는 어두운 틈
        """
        defects = []
        h, w = gray.shape

        # 상단 영역 (RDL-TSV 접합부)
        y_start = layers["top"]
        y_end = layers["upper_third"] + int(h * 0.1)
        y_end = min(y_end, h)
        roi = denoised[y_start:y_end, :]

        if roi.size == 0:
            return defects

        # 적응형 이진화 — 주변보다 어두운 영역(갭) 탐지
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 12
        )

        # 세로 방향 구조 강조 (TSV는 세로 기둥)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area or area > self.max_defect_area * 0.3:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)
            y_global = y_local + y_start

            # 종횡비: 갭은 보통 가로로 넓거나 정사각형에 가까움
            aspect = cw / (ch + 1e-5)
            if aspect < 0.3 or aspect > 8.0:
                continue

            # 영역 내 밝기 — 갭은 어두움
            roi_region = gray[y_global:y_global + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)
            global_mean = np.mean(gray)

            if mean_val < global_mean * 0.75:
                darkness_score = 1 - (mean_val / global_mean)
                confidence = min(0.95, 0.40 + darkness_score * 0.4 + min(area / 2000, 0.15))
                defects.append({
                    "type": "Open Defect (TSV-RDL)",
                    "bbox": [int(x), int(y_global), int(cw), int(ch)],
                    "confidence": round(float(confidence), 2),
                    "contour": cnt,
                })

        return defects

    # ================================================================
    # 결함 ② Open Defect (Bump)
    # ================================================================

    def detect_open_defect_bump(self, gray, denoised, layers):
        """
        범프-다이 접합부에서 어두운 갭을 탐지.
        - 위치: 이미지 하단 1/3 영역
        - 특징: 범프(밝은 블록) 사이/위에 있는 어두운 틈
        """
        defects = []
        h, w = gray.shape

        # 하단 영역 (범프-다이 접합부)
        y_start = max(layers["lower_third"] - int(h * 0.1), 0)
        y_end = layers["bottom"]
        roi = denoised[y_start:y_end, :]

        if roi.size == 0:
            return defects

        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # 가로 방향 구조 강조 (범프 사이 갭은 가로로 이어질 수 있음)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area or area > self.max_defect_area * 0.3:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)
            y_global = y_local + y_start

            # 가로로 넓은 갭 형태
            aspect = cw / (ch + 1e-5)
            if aspect < 0.5:
                continue

            roi_region = gray[y_global:y_global + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)
            global_mean = np.mean(gray)

            if mean_val < global_mean * 0.70:
                darkness_score = 1 - (mean_val / global_mean)
                confidence = min(0.95, 0.38 + darkness_score * 0.4 + min(area / 3000, 0.17))
                defects.append({
                    "type": "Open Defect (Bump)",
                    "bbox": [int(x), int(y_global), int(cw), int(ch)],
                    "confidence": round(float(confidence), 2),
                    "contour": cnt,
                })

        return defects

    # ================================================================
    # 결함 ③ Short Defect (Bump)
    # ================================================================

    def detect_short_defect_bump(self, gray, denoised, layers):
        """
        인접 범프 사이 밝은 브릿지(단락)를 탐지.
        - 위치: 이미지 하단 1/3 영역
        - 특징: 범프 사이에 있어야 할 어두운 갭 대신 밝은 연결부
        """
        defects = []
        h, w = gray.shape

        # 하단 영역
        y_start = max(layers["lower_third"] - int(h * 0.1), 0)
        y_end = layers["bottom"]
        roi = denoised[y_start:y_end, :]

        if roi.size == 0:
            return defects

        # 밝은 영역 탐지 (브릿지는 범프와 비슷하게 밝음)
        # Otsu로 밝은 영역 추출
        _, binary_bright = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 가로로 연결된 밝은 영역 강조
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        connected = cv2.morphologyEx(binary_bright, cv2.MORPH_CLOSE, kernel_h, iterations=3)

        # 큰 범프 영역을 제거하고 가로로 이어진 좁은 브릿지만 남김
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        eroded = cv2.erode(connected, kernel_erode, iterations=2)
        bridges = cv2.bitwise_and(connected, cv2.bitwise_not(eroded))

        # 가로 방향 필터
        kernel_h2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        bridges = cv2.morphologyEx(bridges, cv2.MORPH_OPEN, kernel_h2, iterations=1)

        contours, _ = cv2.findContours(bridges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area * 0.5 or area > self.max_defect_area * 0.2:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)
            y_global = y_local + y_start

            # 가로로 넓은 형태 (브릿지)
            aspect = cw / (ch + 1e-5)
            if aspect < 1.5:
                continue

            # 밝기 확인 — 브릿지는 밝아야 함
            roi_region = gray[y_global:y_global + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)
            global_mean = np.mean(gray)

            if mean_val > global_mean * 0.9:
                brightness_score = mean_val / 255
                confidence = min(0.90, 0.35 + brightness_score * 0.3 + min(aspect / 15, 0.25))
                defects.append({
                    "type": "Short Defect (Bump)",
                    "bbox": [int(x), int(y_global), int(cw), int(ch)],
                    "confidence": round(float(confidence), 2),
                    "contour": cnt,
                })

        return defects

    # ================================================================
    # 결함 ④ Void Formation (TSV)
    # ================================================================

    def detect_void_formation(self, gray, denoised, layers):
        """
        TSV 내부의 어두운 공동(void)을 탐지.
        - 위치: 이미지 중앙 영역 (TSV 기둥 내부)
        - 특징: 비교적 둥근 어두운 영역, TSV 내부에 위치
        """
        defects = []
        h, w = gray.shape

        # 중앙 영역 (TSV 본체)
        margin = int(h * 0.05)
        y_start = max(layers["upper_third"] - margin, 0)
        y_end = min(layers["lower_third"] + margin, h)
        roi = denoised[y_start:y_end, :]

        if roi.size == 0:
            return defects

        # 적응형 이진화 — 어두운 영역 탐지
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )

        # 모폴로지로 노이즈 제거, 둥근 형태 강조
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area or area > self.max_defect_area * 0.5:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Void는 비교적 둥근 형태
            if circularity < 0.25:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)
            y_global = y_local + y_start

            # 영역 내 밝기 — 어두울수록 void 가능성 높음
            roi_region = gray[y_global:y_global + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)
            global_mean = np.mean(gray)

            if mean_val < global_mean * 0.65:
                darkness_score = 1 - (mean_val / global_mean)
                circ_score = circularity
                confidence = min(0.95, 0.40 + darkness_score * 0.3 + circ_score * 0.25)
                defects.append({
                    "type": "Void Formation (TSV)",
                    "bbox": [int(x), int(y_global), int(cw), int(ch)],
                    "confidence": round(float(confidence), 2),
                    "contour": cnt,
                })

        return defects

    # ================================================================
    # 중복 제거
    # ================================================================

    def remove_overlapping(self, defects, iou_threshold=0.3):
        """IoU 기반 중복 검출 제거."""
        if len(defects) <= 1:
            return defects

        defects.sort(key=lambda d: d["confidence"], reverse=True)
        kept = []

        for defect in defects:
            x1, y1, w1, h1 = defect["bbox"]
            is_duplicate = False

            for existing in kept:
                x2, y2, w2, h2 = existing["bbox"]
                xi = max(x1, x2)
                yi = max(y1, y2)
                xa = min(x1 + w1, x2 + w2)
                ya = min(y1 + h1, y2 + h2)

                if xi < xa and yi < ya:
                    intersection = (xa - xi) * (ya - yi)
                    union = w1 * h1 + w2 * h2 - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou > iou_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(defect)

        return kept

    # ================================================================
    # 결과 시각화
    # ================================================================

    def draw_results(self, image, defects):
        """결함 위치를 이미지 위에 시각화."""
        result = image.copy()

        for defect in defects:
            color = DEFECT_COLORS[defect["type"]]
            x, y, w, h = defect["bbox"]

            # 바운딩 박스 (두께 2)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # 라벨
            # 짧은 이름 매핑
            short_names = {
                "Open Defect (TSV-RDL)": "Open(TSV-RDL)",
                "Open Defect (Bump)": "Open(Bump)",
                "Short Defect (Bump)": "Short(Bump)",
                "Void Formation (TSV)": "Void(TSV)",
            }
            label = f"{short_names.get(defect['type'], defect['type'])} {defect['confidence']:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            label_y = max(y - 8, th + 4)
            cv2.rectangle(result, (x, label_y - th - 4), (x + tw + 6, label_y + 2), color, -1)
            cv2.putText(result, label, (x + 3, label_y - 1), font, font_scale, (255, 255, 255), thickness)

        return result

    # ================================================================
    # 메인 분석
    # ================================================================

    def _run_detection(self, image):
        """4가지 결함 탐지를 실행하고 결과 리스트를 반환하는 내부 메서드."""
        gray, enhanced, denoised = self.preprocess(image)
        layers = self._find_layer_regions(gray)

        all_defects = []
        all_defects.extend(self.detect_open_defect_tsv_rdl(gray, denoised, layers))
        all_defects.extend(self.detect_open_defect_bump(gray, denoised, layers))
        all_defects.extend(self.detect_short_defect_bump(gray, denoised, layers))
        all_defects.extend(self.detect_void_formation(gray, denoised, layers))

        return self.remove_overlapping(all_defects)

    def _build_result(self, image, all_defects, original_path=None, result_path=None):
        """탐지 결과를 JSON 응답 형식으로 구성."""
        h, w = image.shape[:2]

        # 결과 이미지 생성 → base64
        result_image = self.draw_results(image, all_defects)
        _, buf = cv2.imencode(".png", result_image)
        import base64
        result_b64 = base64.b64encode(buf).decode("utf-8")

        # 원본도 base64
        _, orig_buf = cv2.imencode(".png", image)
        original_b64 = base64.b64encode(orig_buf).decode("utf-8")

        defect_list = []
        for d in all_defects:
            defect_list.append({
                "type": d["type"],
                "description": DEFECT_DESCRIPTIONS[d["type"]],
                "bbox": d["bbox"],
                "confidence": d["confidence"],
            })

        type_counts = {}
        for d in defect_list:
            type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1

        result = {
            "original_image_b64": original_b64,
            "result_image_b64": result_b64,
            "image_size": {"width": w, "height": h},
            "total_defects": len(defect_list),
            "defect_summary": type_counts,
            "defects": defect_list,
        }

        # 로컬 파일 경로도 포함 (로컬 실행 시)
        if original_path:
            result["original_image"] = original_path.replace("\\", "/")
        if result_path:
            result["result_image"] = result_path.replace("\\", "/")

        return result

    def analyze_in_memory(self, image):
        """메모리 기반 분석 (Vercel 배포용) — numpy 배열을 직접 받는다."""
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다."}

        all_defects = self._run_detection(image)
        return self._build_result(image, all_defects)

    def analyze(self, image_path, result_folder, filename):
        """파일 경로 기반 분석 (로컬 실행용)."""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다."}

        all_defects = self._run_detection(image)

        # 결과 이미지를 디스크에도 저장 (로컬용)
        result_image = self.draw_results(image, all_defects)
        result_png = os.path.join(result_folder, f"result_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(result_png, result_image)

        return self._build_result(image, all_defects, image_path, result_png)
