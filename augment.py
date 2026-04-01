"""
데이터 증강(Data Augmentation) 유틸리티
======================================
적은 수의 SEM/FIB 이미지를 다양한 변환으로 늘려주는 도구.

사용법:
    python augment.py --input data/raw --output data/augmented --count 10

각 원본 이미지당 count개의 증강 이미지를 생성한다.
나중에 YOLO 등 모델 학습에 사용할 수 있다.
"""

import os
import argparse
import cv2
import numpy as np


def random_brightness(image, low=0.7, high=1.3):
    """밝기를 랜덤하게 조절 — SEM 이미지의 노출 차이를 모사."""
    factor = np.random.uniform(low, high)
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def random_contrast(image, low=0.7, high=1.4):
    """대비를 랜덤하게 조절."""
    factor = np.random.uniform(low, high)
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def add_gaussian_noise(image, mean=0, std_range=(5, 25)):
    """가우시안 노이즈 추가 — SEM 이미지의 촬영 노이즈를 모사."""
    std = np.random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def random_flip(image):
    """좌우/상하 반전."""
    flip_code = np.random.choice([-1, 0, 1])  # -1=both, 0=vertical, 1=horizontal
    return cv2.flip(image, flip_code)


def random_rotation(image, max_angle=15):
    """소각도 회전 — SEM 시료 장착 각도 차이를 모사."""
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_crop_resize(image, min_crop=0.8):
    """랜덤 영역 크롭 후 원본 크기로 리사이즈."""
    h, w = image.shape[:2]
    crop_ratio = np.random.uniform(min_crop, 1.0)
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y = np.random.randint(0, h - new_h + 1)
    x = np.random.randint(0, w - new_w + 1)
    cropped = image[y:y + new_h, x:x + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def random_blur(image, max_kernel=5):
    """랜덤 블러 — 초점 차이를 모사."""
    k = np.random.choice([1, 3, 5])
    if k == 1:
        return image
    return cv2.GaussianBlur(image, (k, k), 0)


def augment_image(image):
    """여러 변환을 랜덤하게 조합해 하나의 증강 이미지를 생성."""
    result = image.copy()

    # 각 변환을 50% 확률로 적용
    transforms = [
        (0.5, random_brightness),
        (0.5, random_contrast),
        (0.5, add_gaussian_noise),
        (0.4, random_flip),
        (0.3, random_rotation),
        (0.4, random_crop_resize),
        (0.3, random_blur),
    ]

    for prob, transform in transforms:
        if np.random.random() < prob:
            result = transform(result)

    return result


def main():
    parser = argparse.ArgumentParser(description="SEM/FIB 이미지 데이터 증강 도구")
    parser.add_argument("--input", required=True, help="원본 이미지 폴더 경로")
    parser.add_argument("--output", required=True, help="증강 이미지 저장 폴더 경로")
    parser.add_argument("--count", type=int, default=10, help="원본 1장당 증강 이미지 수 (기본: 10)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_files = [
        f for f in os.listdir(args.input)
        if f.lower().endswith(extensions)
    ]

    if not image_files:
        print(f"'{args.input}' 폴더에 이미지가 없습니다.")
        return

    total = 0
    for filename in image_files:
        filepath = os.path.join(args.input, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"  [skip] {filename} - 읽기 실패")
            continue

        name, ext = os.path.splitext(filename)

        # 원본도 복사
        cv2.imwrite(os.path.join(args.output, filename), image)

        # 증강 이미지 생성
        for i in range(args.count):
            aug = augment_image(image)
            out_name = f"{name}_aug{i:03d}{ext}"
            cv2.imwrite(os.path.join(args.output, out_name), aug)
            total += 1

        print(f"  {filename} -> {args.count} augmented images")

    print(f"\nDone! {len(image_files)} originals -> {total} augmented images in '{args.output}'")


if __name__ == "__main__":
    main()
