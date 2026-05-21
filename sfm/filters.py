import numpy as np
from scipy.spatial import cKDTree


def finite_filter(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1)


def depth_percentile_filter(points: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    lower, upper = np.percentile(points[:, 2], [low, high])
    return (points[:, 2] >= lower) & (points[:, 2] <= upper)


def statistical_outlier_filter(points: np.ndarray, k: int = 12, std_ratio: float = 2.0) -> np.ndarray:
    if points.shape[0] <= k:
        return np.ones(points.shape[0], dtype=bool)

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    return mean_distances <= threshold


def combined_outlier_filter(points: np.ndarray, depth_low: float = 1.0, depth_high: float = 99.0, k: int = 12, std_ratio: float = 2.0) -> np.ndarray:
    finite = finite_filter(points)
    valid_indices = np.where(finite)[0]
    valid_points = points[finite]
    keep = np.zeros(points.shape[0], dtype=bool)
    if valid_points.shape[0] == 0:
        return keep
    depth_keep = depth_percentile_filter(valid_points, depth_low, depth_high)
    depth_points = valid_points[depth_keep]
    depth_indices = valid_indices[depth_keep]
    stat_keep = statistical_outlier_filter(depth_points, k, std_ratio)
    keep[depth_indices[stat_keep]] = True
    return keep
