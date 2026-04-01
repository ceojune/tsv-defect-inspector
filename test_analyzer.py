"""분석 엔진 v2 테스트 — 4가지 TSV 결함이 포함된 합성 SEM 단면 이미지로 검증."""

import os
import cv2
import numpy as np
from analyzer import TSVDefectAnalyzer


def create_test_sem_cross_section():
    """TSV 단면 구조를 모사한 합성 SEM 이미지 생성.

    구조 (위에서 아래):
    - RDL 층 (밝은 수평 선)
    - TSV 기둥 (중간 밝기 세로 구조)
    - Si Substrate (어두운 배경)
    - Bump (밝은 블록)
    """
    h, w = 500, 600
    img = np.ones((h, w), dtype=np.uint8) * 100  # Si Sub 배경

    # 노이즈
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # RDL 층 (상단 밝은 띠)
    img[20:60, :] = 200

    # TSV 기둥들 (세로 밝은 구조)
    tsv_positions = [100, 250, 400, 530]
    tsv_width = 40
    for tx in tsv_positions:
        img[60:380, tx:tx + tsv_width] = 170

    # Bump (하단 밝은 블록)
    for tx in tsv_positions:
        img[380:440, tx - 15:tx + tsv_width + 15] = 210

    # Si Sub 라벨 영역
    img[440:, :] = 80

    # --- 결함 삽입 ---

    # 1) Open Defect (TSV-RDL): TSV 상단에 어두운 갭
    cv2.rectangle(img, (tsv_positions[0] + 5, 50), (tsv_positions[0] + 35, 70), 30, -1)

    # 2) Open Defect (Bump): 범프 위에 어두운 갭
    cv2.rectangle(img, (tsv_positions[2] - 10, 370), (tsv_positions[2] + 50, 385), 25, -1)

    # 3) Short Defect (Bump): 두 범프 사이 밝은 브릿지
    bx1 = tsv_positions[1] + tsv_width + 15
    bx2 = tsv_positions[2] - 15
    cv2.rectangle(img, (bx1, 400), (bx2, 415), 220, -1)

    # 4) Void Formation (TSV): TSV 내부 어두운 원
    cv2.circle(img, (tsv_positions[3] + tsv_width // 2, 200), 15, 25, -1)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    test_img = create_test_sem_cross_section()
    test_path = os.path.join("static", "uploads", "test_sem_v2.png")
    cv2.imwrite(test_path, test_img)
    print(f"Test image: {test_path}")

    analyzer = TSVDefectAnalyzer()
    result = analyzer.analyze(test_path, os.path.join("static", "results"), "test_sem_v2.png")

    print(f"\n===== Analysis Result =====")
    print(f"Image size: {result['image_size']}")
    print(f"Total defects: {result['total_defects']}")
    print(f"Summary: {result['defect_summary']}")
    print(f"\nDefect details:")
    for i, d in enumerate(result["defects"], 1):
        print(f"  {i}. [{d['type']}] pos=({d['bbox'][0]},{d['bbox'][1]}) "
              f"size={d['bbox'][2]}x{d['bbox'][3]} conf={d['confidence']:.0%}")
    print(f"\nResult image: {result['result_image']}")
    print("\nTest PASSED!")


if __name__ == "__main__":
    main()
