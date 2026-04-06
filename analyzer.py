"""
TSV 결함 탐지 엔진 v2
=====================
SEM/FIB cross-section 이미지에서 4가지 TSV 결함을 탐지하는 모듈.

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
    "Incomplete Fill (TSV)": (255, 128, 0),     # 주황
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
    "Incomplete Fill (TSV)": (
        "electroplating 중 구리가 TSV 내부를 완전히 채우지 못해 세로 방향의 crack "
        "또는 seam이 생긴 상태예요. 도금액 첨가제 비율 불균형이나 과도한 전류밀도로 "
        "bottom-up 충전이 실패할 때 주로 발생해요."
    ),
}

# ● = "high" (해당 공정에서 발생 가능성 높음)
# ○ = "low"  (간접적 원인으로 나타날 수 있음)
# 빈 문자열 = 무관
PROCESS_STAGES = ["S1 Via", "S2 Liner", "S3 Cu Fill", "S4 CMP", "S5 Bond"]

PROCESS_STAGE_DETAILS = {
    "S1 Via": {"name": "Via 형성", "desc": "실리콘 기판에 TSV 홀을 에칭하는 단계"},
    "S2 Liner": {"name": "Liner 증착", "desc": "절연막/배리어 금속을 TSV 내벽에 증착하는 단계"},
    "S3 Cu Fill": {"name": "Cu 충전", "desc": "전해도금으로 구리를 TSV 내부에 채우는 단계"},
    "S4 CMP": {"name": "CMP 평탄화", "desc": "화학적 기계적 연마로 표면을 평탄화하는 단계"},
    "S5 Bond": {"name": "본딩", "desc": "다이 적층 및 범프 접합을 수행하는 단계"},
}

# 결함-공정 매핑 매트릭스 (논문 표 3.2 기준)
# 각 세부 결함이 어떤 공정에서 발생하는지 매핑
DEFECT_PROCESS_MATRIX = {
    "Void":            {"S1 Via": "low",  "S2 Liner": "low",  "S3 Cu Fill": "high", "S4 CMP": "high", "S5 Bond": "low"},
    "Pinch-off":       {"S1 Via": "",     "S2 Liner": "",     "S3 Cu Fill": "high", "S4 CMP": "",     "S5 Bond": ""},
    "Crack":           {"S1 Via": "low",  "S2 Liner": "high", "S3 Cu Fill": "low",  "S4 CMP": "low",  "S5 Bond": "high"},
    "Incomplete fill": {"S1 Via": "",     "S2 Liner": "",     "S3 Cu Fill": "high", "S4 CMP": "",     "S5 Bond": ""},
    "Scalloping":      {"S1 Via": "high", "S2 Liner": "",     "S3 Cu Fill": "",     "S4 CMP": "",     "S5 Bond": ""},
    "Delamination":    {"S1 Via": "",     "S2 Liner": "high", "S3 Cu Fill": "",     "S4 CMP": "low",  "S5 Bond": "high"},
    "Misalignment":    {"S1 Via": "high", "S2 Liner": "",     "S3 Cu Fill": "",     "S4 CMP": "",     "S5 Bond": "high"},
    "Contamination":   {"S1 Via": "low",  "S2 Liner": "low",  "S3 Cu Fill": "low",  "S4 CMP": "high", "S5 Bond": "high"},
}

# 탐지되는 4가지 결함 → 매트릭스의 세부 결함으로 매핑
DEFECT_TO_MATRIX_ROWS = {
    "Open Defect (TSV-RDL)": ["Void", "Incomplete fill", "Crack"],
    "Open Defect (Bump)":    ["Delamination", "Misalignment", "Crack"],
    "Short Defect (Bump)":   ["Misalignment", "Contamination"],
    "Void Formation (TSV)":  ["Void", "Pinch-off", "Incomplete fill"],
    "Incomplete Fill (TSV)": ["Incomplete fill", "Void", "Pinch-off"],
}

# 각 탐지 결함에 대한 엔지니어용 조치 가이드
DEFECT_ACTIONS = {
    "Open Defect (TSV-RDL)": [
        "Cu 도금 전해액의 농도·온도·전류밀도 조건을 점검하세요.",
        "TSV 홀의 aspect ratio가 도금 한계를 초과하는지 확인하세요.",
        "Liner/Seed layer 균일성을 SEM으로 검증하세요.",
    ],
    "Open Defect (Bump)": [
        "bump coplanarity를 측정하세요.",
        "본딩 온도·압력·시간 프로파일을 재검토하세요.",
        "UBM 표면 산화 여부를 확인하세요.",
    ],
    "Short Defect (Bump)": [
        "범프 피치 대비 범프 직경 비율을 확인하세요.",
        "bonding force가 과도하지 않은지 점검하세요.",
        "underfill 유동 특성을 검토하세요.",
    ],
    "Void Formation (TSV)": [
        "도금 전해액의 accelerator/suppressor 비율을 조정하세요.",
        "도금 전류 waveform(DC vs Pulse)을 최적화하세요.",
        "annealing 프로파일에서 급격한 온도 변화를 완화하세요.",
    ],
    "Incomplete Fill (TSV)": [
        "도금 전해액의 Accelerator/Suppressor/Leveler 삼중 첨가제 비율을 재조정하세요.",
        "전류밀도를 낮추고 pulse plating으로 전환해 bottom-up 충전을 유도하세요.",
        "TSV aspect ratio가 도금 공정 한계를 초과하는지 시뮬레이션으로 검증하세요.",
        "도금 전 seed layer 균일도를 SEM으로 확인하세요.",
    ],
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

    def _find_tsv_columns(self, gray, denoised):
        """
        TSV 기둥(구리 충전 영역)을 식별해서 마스크를 반환.
        SEM 이미지에서 TSV 기둥은 기판보다 밝은 세로 구조물이에요.
        냉장고 안에 세워둔 밝은 음료수 캔들을 찾는 것과 비슷해요.

        Returns:
            tsv_mask: TSV 기둥 영역이 255인 바이너리 마스크
        """
        h, w = gray.shape

        # Otsu로 밝은 영역(TSV 구리) vs 어두운 영역(기판) 분리
        _, bright_mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 세로 방향 구조 강조 — TSV는 세로 기둥이므로
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 10, 5)))
        tsv_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_v, iterations=1)

        # 작은 노이즈 제거
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tsv_mask = cv2.morphologyEx(tsv_mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)

        # 형태 필터: 세로 기둥(height > width)만 TSV로 인정
        # 범프·RDL 등 가로로 넓은 구조물은 TSV 마스크에서 제외
        contours_shape, _ = cv2.findContours(tsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(tsv_mask)
        for cnt in contours_shape:
            _, _, cw, ch = cv2.boundingRect(cnt)
            # TSV 기둥 조건: 세로가 가로보다 길고, 이미지 높이의 10% 이상
            col_aspect = ch / (cw + 1e-5)
            if col_aspect >= 1.5 and ch >= h * 0.10:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

        return filtered_mask

    def _find_sem_info_bar_height(self, gray):
        """
        SEM 이미지 하단의 메타데이터 바(눈금·배율 정보가 있는 검정 띠) 높이를 탐지.
        마치 자막 바처럼 이미지 맨 아래에 붙어있는 어두운 영역을 찾아 그 높이를 반환해요.

        주의: 바 안에 흰 텍스트(배율·거리 정보)가 있어서 모든 행이 어둡지는 않아요.
        그래서 밝은 행이 연속 3개 나올 때까지는 바 영역으로 간주해요 (내성 있는 탐색).
        """
        h = gray.shape[0]
        max_bar_height = int(h * 0.15)

        # 빠른 사전 확인: 맨 아래 3행이 어둡지 않으면 정보 바 없음
        if np.mean(gray[max(h - 3, 0):h, :]) > 55:
            return 0

        # 아래서 위로 스캔 — 밝은 행이 연속 3개 이상 나오면 바 영역 종료
        bar_start = h
        bright_streak = 0
        for i in range(h - 1, max(h - max_bar_height, 0) - 1, -1):
            row_mean = np.mean(gray[i, :])
            if row_mean < 70:  # 어두운 행 (바 영역 또는 텍스트 사이 검정)
                bar_start = i
                bright_streak = 0
            else:
                bright_streak += 1
                if bright_streak >= 3:  # 밝은 행 3개 연속 → 실제 이미지 영역
                    break

        bar_height = h - bar_start
        return bar_height if bar_height > 5 else 0

    def _crop_sem_info_bar(self, image):
        """SEM 정보 바를 잘라낸 이미지를 반환. 정보 바가 없으면 원본 반환."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        bar_h = self._find_sem_info_bar_height(gray)
        if bar_h > 0:
            return image[:image.shape[0] - bar_h, :]
        return image

    # ================================================================
    # 결함 ① Open Defect (TSV-RDL)
    # ================================================================

    def detect_open_defect_tsv_rdl(self, gray, denoised, layers, tsv_mask=None):
        """
        TSV 상단-RDL 접합부에서 어두운 gap을 탐지.
        - 위치: 이미지 상단 1/3 영역
        - 특징: TSV 구조(세로로 밝은 기둥) 바로 위에 있는 어두운 틈
        - TSV 기둥 내부의 어두운 점은 Void이므로 제외
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

            # TSV 기둥 내부에 완전히 포함된 영역은 건너뜀 (→ Void가 담당)
            if tsv_mask is not None:
                roi_tsv = tsv_mask[y_global:y_global + ch, x:x + cw]
                if roi_tsv.size > 0 and np.mean(roi_tsv > 0) > 0.7:
                    continue

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

    def detect_open_defect_bump(self, gray, denoised, layers, tsv_mask=None):
        """
        범프-다이 접합부에서 어두운 갭을 탐지.
        - 위치: 이미지 상단 1/3 아래 ~ 하단 전체 (범프가 이미지 중간에 위치할 수 있음)
        - 특징: 범프(밝은 블록) 사이/위에 있는 어두운 틈
        - TSV 기둥 내부의 어두운 점은 제외
        """
        defects = []
        h, w = gray.shape

        # 탐색 영역: upper_third ~ bottom (범프는 이미지 중간~하단에 위치)
        y_start = layers["upper_third"]
        y_end = layers["bottom"]
        roi = denoised[y_start:y_end, :]

        if roi.size == 0:
            return defects

        global_mean = np.mean(gray)

        # ── 방법 A: 적응형 이진화 (작은~중간 크기 gap 탐지) ────────────
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 8
        )
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_a = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── 방법 B: 전역 임계값 (범프 위 큰 어두운 gap 탐지) ──────────
        # 범프 표면의 오목한 dark 영역 — 적응형으로는 놓치는 큰 영역 보완
        void_thresh = int(global_mean * 0.65)
        _, dark_global = cv2.threshold(roi, void_thresh, 255, cv2.THRESH_BINARY_INV)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_b = cv2.morphologyEx(dark_global, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        cleaned_b = cv2.morphologyEx(cleaned_b, cv2.MORPH_OPEN, kernel, iterations=1)

        # 두 방법의 결과를 합산
        combined = cv2.bitwise_or(cleaned_a, cleaned_b)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area or area > self.max_defect_area * 0.5:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)
            y_global = y_local + y_start

            # TSV 기둥 내부에 포함된 영역은 건너뜀
            if tsv_mask is not None:
                roi_tsv = tsv_mask[y_global:y_global + ch, x:x + cw]
                if roi_tsv.size > 0 and np.mean(roi_tsv > 0) > 0.7:
                    continue

            # 종횡비: 가로로 어느 정도 이어진 gap 형태 (너무 세로로 좁은 것 제외)
            aspect = cw / (ch + 1e-5)
            if aspect < 0.5:
                continue

            # 이미지 전체 너비의 95% 이상이면 배경 레이어로 판단 (제외)
            if cw > w * 0.95:
                continue

            roi_region = gray[y_global:y_global + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)

            # 너무 밝으면(범프 자체) 또는 너무 어두우면(완전 배경) 제외
            if mean_val > global_mean * 0.80:
                continue
            if mean_val < global_mean * 0.20:
                continue

            # 인접 영역에 밝은 범프 구조물이 있어야 함
            # gap 아래 또는 가로 양옆에 밝은 금속이 있으면 범프 위 gap으로 판단
            pad = max(ch, 10)
            below_y1 = min(y_global + ch, h)
            below_y2 = min(y_global + ch + pad, h)
            below_region = gray[below_y1:below_y2, x:x + cw]

            side_left = gray[y_global:y_global + ch, max(x - pad, 0):x]
            side_right = gray[y_global:y_global + ch, x + cw:min(x + cw + pad, w)]

            bright_thresh = global_mean * 0.80
            has_bright_neighbor = (
                (below_region.size > 0 and np.mean(below_region) > bright_thresh) or
                (side_left.size > 0 and np.mean(side_left) > bright_thresh) or
                (side_right.size > 0 and np.mean(side_right) > bright_thresh)
            )
            if not has_bright_neighbor:
                continue

            darkness_score = 1 - (mean_val / global_mean)
            confidence = min(0.95, 0.38 + darkness_score * 0.45 + min(area / 4000, 0.17))
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

    def detect_short_defect_bump(self, gray, denoised, layers, tsv_mask=None):
        """
        인접 범프 사이 밝은 브릿지(단락)를 탐지.
        - 위치: 이미지 하단 1/3 영역
        - 특징: 범프 사이에 있어야 할 어두운 갭 대신 밝은 연결부
        - TSV 기둥 자체를 브릿지로 오탐하지 않도록 TSV 마스크로 필터링
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
        _, binary_bright = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # TSV 기둥 영역을 제외 — 기둥 및 기둥 사이 seam 영역까지 넓게 팽창
        tsv_mask_for_short = None
        if tsv_mask is not None:
            kernel_excl = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            tsv_mask_for_short = cv2.dilate(tsv_mask, kernel_excl, iterations=3)
            tsv_roi = tsv_mask_for_short[y_start:y_end, :]
            if tsv_roi.shape == binary_bright.shape:
                binary_bright = cv2.bitwise_and(binary_bright, cv2.bitwise_not(tsv_roi))

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

            # TSV 기둥 내부 또는 인접 영역에 포함된 영역은 건너뜀 (팽창 마스크 사용)
            excl_mask = tsv_mask_for_short if tsv_mask_for_short is not None else tsv_mask
            if excl_mask is not None:
                roi_tsv = excl_mask[y_global:y_global + ch, x:x + cw]
                if roi_tsv.size > 0 and np.mean(roi_tsv > 0) > 0.3:
                    continue

            # 가로로 넓은 형태 (브릿지)
            aspect = cw / (ch + 1e-5)
            if aspect < 1.5:
                continue

            # 이미지 전체 너비의 70% 이상 spanning → 단일 수평 선(RDL/wire/scale bar)으로 판단, 제외
            if cw > w * 0.70:
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

    def detect_void_formation(self, gray, denoised, tsv_mask=None):
        """
        TSV 내부의 void를 탐지.

        핵심 아이디어: TSV 기둥은 구리로 채워져 밝아야 하는데, 그 안에 어두운 점이
        있으면 void예요. 아이스크림 안에 기포가 있는 것처럼, 밝은 구리 기둥 안의
        어두운 빈 공간을 찾아요.

        기존 방식(단순 어두운 영역 찾기)과 달리:
        1. TSV 기둥 마스크로 구리 영역을 먼저 식별
        2. 그 안에서만 어두운 점을 찾음
        3. 기판과 TSV 경계의 밝기 차이를 오탐하지 않음
        """
        defects = []

        if tsv_mask is None:
            tsv_mask = self._find_tsv_columns(gray, denoised)

        # --- 방법 A: TSV 마스크 내부에서 어두운 영역 찾기 ---

        # TSV 영역 내부의 평균 밝기 계산 (구리의 밝기 기준)
        tsv_pixels = gray[tsv_mask > 0]
        if len(tsv_pixels) == 0:
            return defects
        tsv_mean = np.mean(tsv_pixels)

        # TSV 영역 내에서 구리보다 상당히 어두운 픽셀 = void 후보
        # 기판 밝기와도 비교해서, 기판보다도 어두운 영역을 우선 탐지
        substrate_pixels = gray[tsv_mask == 0]
        substrate_mean = np.mean(substrate_pixels) if len(substrate_pixels) > 0 else tsv_mean * 0.6

        # void 임계값: TSV 구리 밝기와 기판 밝기 사이, 또는 기판보다 어두움
        void_threshold = min(tsv_mean * 0.7, (tsv_mean + substrate_mean) / 2)

        # TSV 내부에서 어두운 영역만 추출
        dark_in_tsv = np.zeros_like(gray)
        dark_pixels = (gray < void_threshold).astype(np.uint8) * 255
        dark_in_tsv = cv2.bitwise_and(dark_pixels, tsv_mask)

        # 모폴로지로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_in_tsv = cv2.morphologyEx(dark_in_tsv, cv2.MORPH_OPEN, kernel, iterations=1)
        dark_in_tsv = cv2.morphologyEx(dark_in_tsv, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(dark_in_tsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area * 0.5 or area > self.max_defect_area * 0.5:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)

            # 너무 가느다란 영역 제외 (TSV 경계 아티팩트)
            aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
            if aspect > 8.0:
                continue

            # 이 영역이 TSV 기둥 내부에 충분히 포함되는지 확인
            roi_mask = tsv_mask[y_local:y_local + ch, x:x + cw]
            if roi_mask.size == 0:
                continue
            overlap_ratio = np.mean(roi_mask > 0)
            if overlap_ratio < 0.3:
                continue

            # 영역 밝기 vs TSV 구리 밝기
            roi_region = gray[y_local:y_local + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)

            # void는 TSV 구리보다 상당히 어두워야 함
            darkness_ratio = mean_val / tsv_mean
            if darkness_ratio > 0.8:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            darkness_score = 1 - darkness_ratio
            circ_bonus = min(circularity * 0.15, 0.15)
            size_bonus = min(area / 3000, 0.15)
            confidence = min(0.95, 0.40 + darkness_score * 0.35 + circ_bonus + size_bonus)

            defects.append({
                "type": "Void Formation (TSV)",
                "bbox": [int(x), int(y_local), int(cw), int(ch)],
                "confidence": round(float(confidence), 2),
                "contour": cnt,
            })

        # --- 방법 B: 적응형 이진화 보완 (작은 pin-hole 탐지) ---
        # 방법 A에서 놓칠 수 있는 미세한 void를 추가 탐지

        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        # TSV 내부로 제한
        binary = cv2.bitwise_and(binary, tsv_mask)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        contours_b, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_b:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area * 0.3 or area > self.max_defect_area * 0.3:
                continue

            x, y_local, cw, ch = cv2.boundingRect(cnt)
            aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
            if aspect > 6.0:
                continue

            roi_region = gray[y_local:y_local + ch, x:x + cw]
            if roi_region.size == 0:
                continue
            mean_val = np.mean(roi_region)

            if mean_val / tsv_mean > 0.75:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            darkness_score = 1 - (mean_val / tsv_mean)
            confidence = min(0.90, 0.35 + darkness_score * 0.35 + min(circularity * 0.15, 0.15))

            defects.append({
                "type": "Void Formation (TSV)",
                "bbox": [int(x), int(y_local), int(cw), int(ch)],
                "confidence": round(float(confidence), 2),
                "contour": cnt,
            })

        return defects

    # ================================================================
    # 결함 ⑤ Incomplete Fill (TSV)
    # ================================================================

    def detect_incomplete_fill(self, gray, denoised, tsv_mask=None):
        """
        TSV 내부의 세로 방향 crack/seam을 탐지.

        electroplating 중 bottom-up 충전이 실패하면, 구리가 TSV 측벽에서만 자라다가
        중심부에서 맞닿으면서 세로 seam을 남겨요. 빵 반죽이 오븐에서 중심부까지
        익지 않고 겉만 굳어버리는 것과 비슷해요.

        탐지 전략:
        1. TSV 마스크 내부에서 어두운 픽셀 추출
        2. 세로 방향 모폴로지로 crack/seam 형태 강조
        3. 높이 대비 너비가 큰(세로로 긴) 윤곽만 선택
        4. TSV 전체 높이의 20% 이상에 걸친 구조만 인정
        """
        defects = []
        if tsv_mask is None:
            return defects

        h, w = gray.shape

        # denoised(노이즈 제거 이미지)로 기준 밝기 계산 — 노이즈 픽셀이 평균을 왜곡하지 않아요
        tsv_pixels = denoised[tsv_mask > 0]
        if len(tsv_pixels) == 0:
            return defects
        tsv_mean = np.mean(tsv_pixels)

        # TSV 마스크를 넓게 팽창시켜 경계 부근 및 기둥 사이 seam도 포함
        # (Otsu 임계로 만든 마스크가 seam 영역을 경계에서 잘라낼 수 있어요)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tsv_mask_expanded = cv2.dilate(tsv_mask, kernel_dilate, iterations=3)

        # ── 패스 A: 어두운 seam 탐지 (void/gap형 균열) ──────────────────
        # 임계값 0.82: Cu 평균보다 18% 이상 어두우면 seam 후보
        dark_threshold = tsv_mean * 0.82
        dark_mask = (denoised < dark_threshold).astype(np.uint8) * 255
        dark_in_tsv = cv2.bitwise_and(dark_mask, tsv_mask_expanded)

        # ── 패스 B: 밝은 seam 탐지 (FIB-SEM 2차전자 방출 과다형 균열) ──
        # FIB로 단면을 자르면 seam 표면에서 전자가 더 많이 방출되어
        # Cu 평균보다 오히려 밝게 보여요 — 사과를 칼로 잘랐을 때 단면이 하얗게
        # 빛나는 것과 같은 원리예요
        bright_threshold = tsv_mean * 1.05
        bright_mask = (denoised > bright_threshold).astype(np.uint8) * 255
        bright_in_tsv = cv2.bitwise_and(bright_mask, tsv_mask_expanded)

        v_len = max(h // 8, 15)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        candidate_masks = [
            (dark_in_tsv, "dark"),
            (bright_in_tsv, "bright"),
        ]

        seen_boxes = []

        for anomaly_mask, mode in candidate_masks:
            # 세로 방향 연결 강조 — crack/seam은 세로로 길게 이어져요
            vertical_structure = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel_v, iterations=2)
            vertical_structure = cv2.morphologyEx(vertical_structure, cv2.MORPH_OPEN, kernel_clean, iterations=1)

            contours, _ = cv2.findContours(vertical_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue

                x, y, cw, ch = cv2.boundingRect(cnt)

                # 세로로 길쭉해야 함 (높이/너비 ≥ 1.3)
                aspect = ch / (cw + 1e-5)
                if aspect < 1.3:
                    continue

                # TSV 기둥 세로 범위의 8% 이상을 차지해야 함
                col_slice = tsv_mask[:, max(x, 0):min(x + cw, w)]
                tsv_col_height = int(np.sum(np.any(col_slice > 0, axis=1)))
                if tsv_col_height > 0 and ch < tsv_col_height * 0.08:
                    continue

                # TSV(팽창 마스크) 내부 포함 비율 확인
                roi_mask = tsv_mask_expanded[y:y + ch, x:x + cw]
                if roi_mask.size == 0 or np.mean(roi_mask > 0) < 0.15:
                    continue

                roi_region = gray[y:y + ch, x:x + cw]
                if roi_region.size == 0:
                    continue
                mean_val = np.mean(roi_region)
                brightness_ratio = mean_val / (tsv_mean + 1e-5)

                if mode == "dark":
                    # 어두운 seam: 비율이 0.92 이하여야 함
                    if brightness_ratio > 0.92:
                        continue
                    anomaly_score = 1 - brightness_ratio
                else:
                    # 밝은 seam: 비율이 1.03 이상이어야 함
                    if brightness_ratio < 1.03:
                        continue
                    anomaly_score = brightness_ratio - 1.0

                # 중복 박스 제거 (두 패스에서 같은 위치를 잡을 수 있어요)
                is_dup = False
                for sx, sy, sw, sh in seen_boxes:
                    inter_x = max(0, min(x + cw, sx + sw) - max(x, sx))
                    inter_y = max(0, min(y + ch, sy + sh) - max(y, sy))
                    if inter_x * inter_y > 0.5 * cw * ch:
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_boxes.append((x, y, cw, ch))

                aspect_bonus = min((aspect - 1.3) / 10.0, 0.18)
                size_bonus = min(area / 6000, 0.15)
                confidence = min(0.93, 0.42 + anomaly_score * 0.30 + aspect_bonus + size_bonus)

                defects.append({
                    "type": "Incomplete Fill (TSV)",
                    "bbox": [int(x), int(y), int(cw), int(ch)],
                    "confidence": round(float(confidence), 2),
                    "contour": cnt,
                })

        # ── 패스 C: 에지 기반 세로 seam 탐지 ──────────────────────────
        # 밝기 차이가 미묘해서 패스 A/B가 놓치는 seam을 Canny 에지로 잡아요
        # Cu 내부에 세로 에지가 있으면 → seam/crack 가능성이 높아요
        edges = cv2.Canny(denoised, 30, 100)
        edges_in_tsv = cv2.bitwise_and(edges, tsv_mask_expanded)

        # 세로 방향 연결
        kernel_v_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 10, 10)))
        edge_vertical = cv2.morphologyEx(edges_in_tsv, cv2.MORPH_CLOSE, kernel_v_edge, iterations=2)

        # 가로 에지 제거 (seam은 세로이므로)
        kernel_h_remove = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        edge_vertical = cv2.morphologyEx(edge_vertical, cv2.MORPH_OPEN, kernel_h_remove, iterations=1)
        # 세로 에지만 남기기
        kernel_v_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 15, 8)))
        edge_vertical = cv2.morphologyEx(edge_vertical, cv2.MORPH_OPEN, kernel_v_open, iterations=1)

        # 에지를 약간 팽창시켜 인접 에지를 하나로 합침 (seam 양쪽 에지)
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        edge_merged = cv2.dilate(edge_vertical, kernel_merge, iterations=1)

        contours_edge, _ = cv2.findContours(edge_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_edge:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area or area > self.max_defect_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            aspect = ch / (cw + 1e-5)
            if aspect < 1.3:
                continue

            # TSV 기둥 높이 대비 15% 이상 (패스 C는 에지 기반이라 작은 조각이 많아서 엄격하게)
            col_slice = tsv_mask[:, max(x, 0):min(x + cw, w)]
            tsv_col_height = int(np.sum(np.any(col_slice > 0, axis=1)))
            if tsv_col_height > 0 and ch < tsv_col_height * 0.15:
                continue

            # TSV 내부에 있어야 함 (팽창 마스크)
            roi_mask = tsv_mask_expanded[y:y + ch, x:x + cw]
            if roi_mask.size == 0 or np.mean(roi_mask > 0) < 0.15:
                continue

            # 원본 TSV 마스크 내부 비율 — 경계 에지는 원본 마스크와 겹침이 적어요
            # seam은 Cu 기둥 내부에 있으므로 원본 마스크와 40% 이상 겹쳐야 함
            roi_orig = tsv_mask[y:y + ch, x:x + cw]
            if roi_orig.size == 0 or np.mean(roi_orig > 0) < 0.40:
                continue

            # 기존 박스와 중복 확인
            is_dup = False
            for sx, sy, sw, sh in seen_boxes:
                inter_x = max(0, min(x + cw, sx + sw) - max(x, sx))
                inter_y = max(0, min(y + ch, sy + sh) - max(y, sy))
                if inter_x * inter_y > 0.3 * cw * ch:
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_boxes.append((x, y, cw, ch))

            # 에지 강도 기반 confidence
            edge_roi = edges_in_tsv[y:y + ch, x:x + cw]
            edge_density = np.mean(edge_roi > 0) if edge_roi.size > 0 else 0
            aspect_bonus = min((aspect - 1.3) / 10.0, 0.18)
            size_bonus = min(area / 6000, 0.15)
            confidence = min(0.93, 0.45 + edge_density * 0.25 + aspect_bonus + size_bonus)

            defects.append({
                "type": "Incomplete Fill (TSV)",
                "bbox": [int(x), int(y), int(cw), int(ch)],
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

        # Incomplete Fill이 Short(Bump)보다 우선 — 같은 위치 충돌 시 Incomplete Fill 유지
        _PRIORITY = {
            "Incomplete Fill (TSV)": 0,
            "Void Formation (TSV)": 1,
            "Open Defect (TSV-RDL)": 2,
            "Open Defect (Bump)": 2,
            "Short Defect (Bump)": 3,
        }
        defects.sort(key=lambda d: (_PRIORITY.get(d["type"], 2), -d["confidence"]))
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
        """5가지 결함 탐지를 실행하고 결과 리스트를 반환하는 내부 메서드."""
        gray, enhanced, denoised = self.preprocess(image)
        layers = self._find_layer_regions(gray)

        # TSV 기둥 마스크를 한 번만 생성해서 공유
        tsv_mask = self._find_tsv_columns(gray, denoised)

        all_defects = []
        all_defects.extend(self.detect_open_defect_tsv_rdl(gray, denoised, layers, tsv_mask))
        all_defects.extend(self.detect_open_defect_bump(gray, denoised, layers, tsv_mask))
        all_defects.extend(self.detect_short_defect_bump(gray, denoised, layers, tsv_mask))
        all_defects.extend(self.detect_void_formation(gray, denoised, tsv_mask))
        all_defects.extend(self.detect_incomplete_fill(gray, denoised, tsv_mask))

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

        # 공정 원인 분석 — 탐지된 결함들로부터 공정별 위험도 산출
        process_analysis = self._analyze_process_root_cause(defect_list)

        result = {
            "original_image_b64": original_b64,
            "result_image_b64": result_b64,
            "image_size": {"width": w, "height": h},
            "total_defects": len(defect_list),
            "defect_summary": type_counts,
            "defects": defect_list,
            "process_analysis": process_analysis,
        }

        # 로컬 파일 경로도 포함 (로컬 실행 시)
        if original_path:
            result["original_image"] = original_path.replace("\\", "/")
        if result_path:
            result["result_image"] = result_path.replace("\\", "/")

        return result

    def _analyze_process_root_cause(self, defect_list):
        """탐지된 결함을 기반으로 공정 단계별 원인 분석을 수행."""
        detected_types = set(d["type"] for d in defect_list)

        # 공정별 위험 점수 집계
        stage_scores = {s: 0.0 for s in PROCESS_STAGES}
        # 어떤 세부 결함이 관련되는지 추적
        stage_related = {s: [] for s in PROCESS_STAGES}

        for defect_type in detected_types:
            matrix_rows = DEFECT_TO_MATRIX_ROWS.get(defect_type, [])
            # 해당 결함의 평균 신뢰도
            confs = [d["confidence"] for d in defect_list if d["type"] == defect_type]
            avg_conf = sum(confs) / len(confs) if confs else 0

            for row_name in matrix_rows:
                row = DEFECT_PROCESS_MATRIX.get(row_name, {})
                for stage in PROCESS_STAGES:
                    level = row.get(stage, "")
                    if level == "high":
                        stage_scores[stage] += avg_conf * 1.0
                        if row_name not in stage_related[stage]:
                            stage_related[stage].append(row_name)
                    elif level == "low":
                        stage_scores[stage] += avg_conf * 0.3
                        if row_name not in stage_related[stage]:
                            stage_related[stage].append(row_name)

        # 최대 점수로 정규화 → 0~100%
        max_score = max(stage_scores.values()) if stage_scores else 1
        if max_score == 0:
            max_score = 1

        stages_result = []
        for stage in PROCESS_STAGES:
            score = stage_scores[stage]
            risk_pct = round((score / max_score) * 100) if max_score > 0 else 0
            if risk_pct > 70:
                risk_level = "high"
            elif risk_pct > 30:
                risk_level = "medium"
            elif risk_pct > 0:
                risk_level = "low"
            else:
                risk_level = "none"
            stages_result.append({
                "stage": stage,
                "name": PROCESS_STAGE_DETAILS[stage]["name"],
                "desc": PROCESS_STAGE_DETAILS[stage]["desc"],
                "risk_pct": risk_pct,
                "risk_level": risk_level,
                "related_defects": stage_related[stage],
            })

        # 탐지된 결함별 조치 가이드
        actions = {}
        for defect_type in detected_types:
            actions[defect_type] = DEFECT_ACTIONS.get(defect_type, [])

        # 결함-공정 매트릭스 (프론트에서 표로 그릴 용도)
        matrix_data = {}
        related_rows = set()
        for defect_type in detected_types:
            for row_name in DEFECT_TO_MATRIX_ROWS.get(defect_type, []):
                related_rows.add(row_name)
        for row_name in related_rows:
            matrix_data[row_name] = DEFECT_PROCESS_MATRIX.get(row_name, {})

        return {
            "stages": stages_result,
            "actions": actions,
            "matrix": matrix_data,
        }

    def analyze_in_memory(self, image):
        """메모리 기반 분석 (Vercel 배포용) — numpy 배열을 직접 받는다."""
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다."}

        image = self._crop_sem_info_bar(image)  # 하단 SEM 메타데이터 바 제거
        all_defects = self._run_detection(image)
        return self._build_result(image, all_defects)

    def analyze(self, image_path, result_folder, filename):
        """파일 경로 기반 분석 (로컬 실행용)."""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "이미지를 읽을 수 없습니다."}

        image = self._crop_sem_info_bar(image)  # 하단 SEM 메타데이터 바 제거
        all_defects = self._run_detection(image)

        # 결과 이미지를 디스크에도 저장 (로컬용)
        result_image = self.draw_results(image, all_defects)
        result_png = os.path.join(result_folder, f"result_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(result_png, result_image)

        return self._build_result(image, all_defects, image_path, result_png)
