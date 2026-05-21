from dataclasses import dataclass

import numpy as np

from .datasets import load_vgg_dinosaur, sample_colors
from .filters import combined_outlier_filter
from .geometry import reprojection_errors
from .reconstruction import TwoViewReconstructor


@dataclass
class MultiViewResult:
    points_3d: np.ndarray
    colors: np.ndarray
    track_ids: np.ndarray
    pair_reports: list[dict]
    fused_tracks: int
    filtered_tracks: int


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * np.diag(S)) / variance)
    t = target_mean - scale * R @ source_mean
    return scale, R, t


def apply_similarity(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, R, t = transform
    return scale * (R @ points.T).T + t


def reconstruct_pair_tracks(K: np.ndarray, tracks: np.ndarray, images: list[np.ndarray], left: int, right: int, reprojection_threshold: float) -> dict | None:
    visible = np.all(tracks[:, left] >= 0, axis=1) & np.all(tracks[:, right] >= 0, axis=1)
    track_ids = np.where(visible)[0]
    if track_ids.shape[0] < 16:
        return None

    points_left = tracks[track_ids, left]
    points_right = tracks[track_ids, right]
    reconstruction = TwoViewReconstructor(K).reconstruct(points_left, points_right)
    valid = reconstruction.inliers.copy()
    if np.any(valid):
        errors_left = reprojection_errors(reconstruction.camera_a.P, reconstruction.points_3d[valid], points_left[valid])
        errors_right = reprojection_errors(reconstruction.camera_b.P, reconstruction.points_3d[valid], points_right[valid])
        errors = np.full(track_ids.shape[0], np.inf)
        errors[valid] = (errors_left + errors_right) / 2.0
        valid &= errors <= reprojection_threshold

    points = reconstruction.points_3d[valid]
    ids = track_ids[valid]
    colors = np.rint(
        (
            sample_colors(images[left], points_left[valid]).astype(float)
            + sample_colors(images[right], points_right[valid]).astype(float)
        )
        / 2.0
    ).astype(np.uint8)

    return {
        "left": left,
        "right": right,
        "points": points,
        "track_ids": ids,
        "colors": colors,
        "shared_tracks": int(track_ids.shape[0]),
        "kept_tracks": int(points.shape[0]),
        "mean_reprojection_error": reconstruction.mean_reprojection_error,
    }


def reconstruct_vgg_multiview(
    data_dir,
    start_view: int = 20,
    end_view: int = 28,
    focal: float = 850.0,
    reprojection_threshold: float = 1.5,
    depth_low: float = 1.0,
    depth_high: float = 99.0,
    neighbor_k: int = 12,
    std_ratio: float = 2.0,
) -> MultiViewResult:
    data = load_vgg_dinosaur(data_dir, focal)
    tracks = data["tracks"]
    images = data["images"]
    K = data["K"]
    view_indices = list(range(start_view - 1, end_view))
    pair_maps = []

    for left, right in zip(view_indices[:-1], view_indices[1:], strict=True):
        pair = reconstruct_pair_tracks(K, tracks, images, left, right, reprojection_threshold)
        if pair is not None and pair["kept_tracks"] >= 16:
            pair_maps.append(pair)

    if not pair_maps:
        raise RuntimeError("No usable multi-view pairs were reconstructed.")

    fused_points: dict[int, list[np.ndarray]] = {}
    fused_colors: dict[int, list[np.ndarray]] = {}
    pair_reports = []

    for index, pair in enumerate(pair_maps):
        ids = pair["track_ids"]
        points = pair["points"]
        colors = pair["colors"]

        if index > 0:
            common = [track_id for track_id in ids if int(track_id) in fused_points]
            if len(common) >= 6:
                source = np.asarray([points[np.where(ids == track_id)[0][0]] for track_id in common])
                target = np.asarray([np.mean(fused_points[int(track_id)], axis=0) for track_id in common])
                transform = estimate_similarity(source, target)
                points = apply_similarity(points, transform)

        for track_id, point, color in zip(ids, points, colors, strict=True):
            key = int(track_id)
            fused_points.setdefault(key, []).append(point)
            fused_colors.setdefault(key, []).append(color.astype(float))

        pair_reports.append(
            {
                "views": [pair["left"] + 1, pair["right"] + 1],
                "sharedTracks": pair["shared_tracks"],
                "keptTracks": pair["kept_tracks"],
                "meanReprojectionError": pair["mean_reprojection_error"],
            }
        )

    track_ids = np.asarray(sorted(fused_points), dtype=int)
    points = np.asarray([np.mean(fused_points[int(track_id)], axis=0) for track_id in track_ids])
    colors = np.rint(np.asarray([np.mean(fused_colors[int(track_id)], axis=0) for track_id in track_ids])).astype(np.uint8)
    keep = combined_outlier_filter(points, depth_low, depth_high, neighbor_k, std_ratio)

    return MultiViewResult(
        points_3d=points[keep],
        colors=colors[keep],
        track_ids=track_ids[keep],
        pair_reports=pair_reports,
        fused_tracks=int(points.shape[0]),
        filtered_tracks=int(np.sum(keep)),
    )
