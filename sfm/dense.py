from dataclasses import dataclass

import numpy as np

from .geometry import to_homogeneous, triangulate_points


@dataclass
class DenseResult:
    points_3d: np.ndarray
    points_a: np.ndarray
    points_b: np.ndarray
    colors: np.ndarray
    scores: np.ndarray
    candidates: int


def grayscale(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[:, :, 0].astype(float)
        + 0.587 * image[:, :, 1].astype(float)
        + 0.114 * image[:, :, 2].astype(float)
    )


def patch_ncc(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator < 1e-9:
        return -1.0
    return float(np.sum(left_centered * right_centered) / denominator)


def texture_mask(gray: np.ndarray, radius: int, threshold: float) -> np.ndarray:
    height, width = gray.shape
    mask = np.zeros_like(gray, dtype=bool)
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            patch = gray[y - radius : y + radius + 1, x - radius : x + radius + 1]
            mask[y, x] = float(np.std(patch)) >= threshold
    return mask


def epipolar_y(F: np.ndarray, x_a: float, y_a: float, x_b: float) -> float | None:
    line = F @ np.array([x_a, y_a, 1.0])
    if abs(line[1]) < 1e-9:
        return None
    return float(-(line[0] * x_b + line[2]) / line[1])


def dense_match(
    image_a: np.ndarray,
    image_b: np.ndarray,
    F: np.ndarray,
    P_a: np.ndarray,
    P_b: np.ndarray,
    sparse_a: np.ndarray,
    sparse_b: np.ndarray,
    stride: int = 5,
    radius: int = 3,
    search: int = 36,
    texture_threshold: float = 8.0,
    score_threshold: float = 0.82,
    max_points: int = 12000,
) -> DenseResult:
    gray_a = grayscale(image_a)
    gray_b = grayscale(image_b)
    height, width = gray_a.shape
    flow = np.median(sparse_b - sparse_a, axis=0)
    textured = texture_mask(gray_a, radius, texture_threshold)

    matched_a = []
    matched_b = []
    scores = []
    candidates = 0

    for y in range(radius, height - radius, stride):
        for x in range(radius, width - radius, stride):
            if not textured[y, x]:
                continue

            center_x = int(round(x + flow[0]))
            best_score = -1.0
            best_point = None
            patch_a = gray_a[y - radius : y + radius + 1, x - radius : x + radius + 1]

            for candidate_x in range(center_x - search, center_x + search + 1):
                if candidate_x < radius or candidate_x >= width - radius:
                    continue
                candidate_y_float = epipolar_y(F, x, y, candidate_x)
                if candidate_y_float is None:
                    continue
                candidate_y = int(round(candidate_y_float))
                if candidate_y < radius or candidate_y >= height - radius:
                    continue

                patch_b = gray_b[
                    candidate_y - radius : candidate_y + radius + 1,
                    candidate_x - radius : candidate_x + radius + 1,
                ]
                score = patch_ncc(patch_a, patch_b)
                if score > best_score:
                    best_score = score
                    best_point = (candidate_x, candidate_y)

            candidates += 1
            if best_point is not None and best_score >= score_threshold:
                matched_a.append((x, y))
                matched_b.append(best_point)
                scores.append(best_score)
                if len(matched_a) >= max_points:
                    break
        if len(matched_a) >= max_points:
            break

    if not matched_a:
        return DenseResult(
            points_3d=np.empty((0, 3)),
            points_a=np.empty((0, 2)),
            points_b=np.empty((0, 2)),
            colors=np.empty((0, 3), dtype=np.uint8),
            scores=np.empty((0,)),
            candidates=candidates,
        )

    points_a = np.asarray(matched_a, dtype=float)
    points_b = np.asarray(matched_b, dtype=float)
    points_3d = triangulate_points(P_a, P_b, points_a, points_b)
    valid = np.isfinite(points_3d).all(axis=1)
    colors = image_a[points_a[:, 1].astype(int), points_a[:, 0].astype(int)]

    return DenseResult(
        points_3d=points_3d[valid],
        points_a=points_a[valid],
        points_b=points_b[valid],
        colors=colors[valid],
        scores=np.asarray(scores)[valid],
        candidates=candidates,
    )


def depth_percentiles(points_3d: np.ndarray) -> list[float]:
    if points_3d.size == 0:
        return []
    return [float(value) for value in np.percentile(points_3d[:, 2], [5, 50, 95])]
