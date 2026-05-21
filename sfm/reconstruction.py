from dataclasses import dataclass

import numpy as np

from .bundle_adjustment import refine_two_view
from .camera import Camera
from .geometry import (
    decompose_essential,
    essential_from_fundamental,
    estimate_fundamental,
    estimate_fundamental_ransac,
    points_in_front,
    reprojection_errors,
    triangulate_points,
)


@dataclass
class TwoViewReconstruction:
    camera_a: Camera
    camera_b: Camera
    points_3d: np.ndarray
    inliers: np.ndarray
    mean_reprojection_error: float
    fundamental: np.ndarray
    essential: np.ndarray
    ransac_inliers: np.ndarray
    bundle_adjusted: bool


class TwoViewReconstructor:
    def __init__(self, K: np.ndarray, use_ransac: bool = True, bundle_adjust: bool = True):
        self.K = K
        self.use_ransac = use_ransac
        self.bundle_adjust = bundle_adjust

    def reconstruct(self, points_a: np.ndarray, points_b: np.ndarray) -> TwoViewReconstruction:
        if self.use_ransac:
            F, ransac_inliers = estimate_fundamental_ransac(points_a, points_b)
        else:
            F = estimate_fundamental(points_a, points_b)
            ransac_inliers = np.ones(points_a.shape[0], dtype=bool)

        working_a = points_a[ransac_inliers]
        working_b = points_b[ransac_inliers]
        E = essential_from_fundamental(F, self.K, self.K)

        camera_a = Camera(self.K, np.eye(3), np.zeros(3))
        best = None

        for R, t in decompose_essential(E):
            camera_b = Camera(self.K, R, t)
            points_3d = triangulate_points(camera_a.P, camera_b.P, working_a, working_b)
            front_a = points_in_front(camera_a.R, camera_a.t, points_3d)
            front_b = points_in_front(camera_b.R, camera_b.t, points_3d)
            score = int(np.sum(front_a & front_b))
            if best is None or score > best[0]:
                best = (score, camera_b, points_3d)

        if best is None:
            raise RuntimeError("No valid pose could be selected.")

        _, camera_b, points_3d = best
        inliers = points_in_front(camera_a.R, camera_a.t, points_3d) & points_in_front(camera_b.R, camera_b.t, points_3d)
        adjusted = False

        if self.bundle_adjust and np.sum(inliers) >= 8:
            refined_camera_b, refined_points, refined_error = refine_two_view(
                camera_b,
                points_3d[inliers],
                working_a[inliers],
                working_b[inliers],
            )
            camera_b = refined_camera_b
            points_3d[inliers] = refined_points
            mean_error = refined_error
            adjusted = True
        else:
            errors_a = reprojection_errors(camera_a.P, points_3d[inliers], working_a[inliers])
            errors_b = reprojection_errors(camera_b.P, points_3d[inliers], working_b[inliers])
            mean_error = float(np.mean(np.hstack([errors_a, errors_b]))) if np.any(inliers) else float("inf")

        full_points = np.full((points_a.shape[0], 3), np.nan)
        full_points[ransac_inliers] = points_3d
        full_inliers = np.zeros(points_a.shape[0], dtype=bool)
        full_inliers[ransac_inliers] = inliers

        return TwoViewReconstruction(
            camera_a=camera_a,
            camera_b=camera_b,
            points_3d=full_points,
            inliers=full_inliers,
            mean_reprojection_error=mean_error,
            fundamental=F,
            essential=E,
            ransac_inliers=ransac_inliers,
            bundle_adjusted=adjusted,
        )
