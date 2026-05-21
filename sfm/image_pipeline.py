from pathlib import Path

import numpy as np

from .camera import make_intrinsics
from .io import save_json, save_ply
from .reconstruction import TwoViewReconstructor


def load_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for image-pair reconstruction. Install it with: python -m pip install opencv-python") from exc
    return cv2


def detect_and_match(image_a_path: Path, image_b_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    cv2 = load_cv2()
    image_a = cv2.imread(str(image_a_path), cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(str(image_b_path), cv2.IMREAD_GRAYSCALE)

    if image_a is None or image_b is None:
        raise ValueError("Both image paths must point to readable images.")

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        detector = cv2.ORB_create(nfeatures=4000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    keypoints_a, descriptors_a = detector.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(image_b, None)

    matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [first for first, second in matches if first.distance < 0.75 * second.distance]

    if len(good) < 8:
        raise RuntimeError("Not enough feature matches for two-view reconstruction.")

    points_a = np.float64([keypoints_a[match.queryIdx].pt for match in good])
    points_b = np.float64([keypoints_b[match.trainIdx].pt for match in good])
    height, width = image_a.shape[:2]
    return points_a, points_b, (width, height)


def reconstruct_pair(image_a_path: Path, image_b_path: Path, out_dir: Path, focal: float | None = None) -> dict:
    points_a, points_b, (width, height) = detect_and_match(image_a_path, image_b_path)
    if focal is None:
        focal = 1.2 * max(width, height)

    K = make_intrinsics(width, height, focal)
    reconstruction = TwoViewReconstructor(K).reconstruct(points_a, points_b)
    valid_points = reconstruction.points_3d[reconstruction.inliers]

    save_ply(out_dir / "pair_reconstruction.ply", valid_points)
    report = {
        "matches": int(points_a.shape[0]),
        "triangulatedPoints": int(valid_points.shape[0]),
        "meanReprojectionError": reconstruction.mean_reprojection_error,
        "focal": focal,
        "imageSize": [width, height],
    }
    save_json(out_dir / "pair_report.json", report)
    return report
