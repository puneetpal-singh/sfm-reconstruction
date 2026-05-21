import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import Camera
from .geometry import project_points, reprojection_errors


def pack_parameters(camera_b: Camera, points_3d: np.ndarray) -> np.ndarray:
    rvec = Rotation.from_matrix(camera_b.R).as_rotvec()
    return np.hstack([rvec, camera_b.t, points_3d.reshape(-1)])


def unpack_parameters(parameters: np.ndarray, K: np.ndarray, n_points: int) -> tuple[Camera, np.ndarray]:
    rvec = parameters[:3]
    t = parameters[3:6]
    points_3d = parameters[6:].reshape(n_points, 3)
    R = Rotation.from_rotvec(rvec).as_matrix()
    return Camera(K, R, t), points_3d


def residuals(parameters: np.ndarray, K: np.ndarray, points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    n_points = points_a.shape[0]
    camera_a = Camera(K, np.eye(3), np.zeros(3))
    camera_b, points_3d = unpack_parameters(parameters, K, n_points)
    projected_a = project_points(camera_a.P, points_3d)
    projected_b = project_points(camera_b.P, points_3d)
    return np.hstack([(projected_a - points_a).reshape(-1), (projected_b - points_b).reshape(-1)])


def refine_two_view(
    camera_b: Camera,
    points_3d: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    max_nfev: int = 80,
) -> tuple[Camera, np.ndarray, float]:
    parameters = pack_parameters(camera_b, points_3d)
    result = least_squares(
        residuals,
        parameters,
        args=(camera_b.K, points_a, points_b),
        loss="huber",
        f_scale=2.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    refined_camera_b, refined_points = unpack_parameters(result.x, camera_b.K, points_a.shape[0])
    camera_a = Camera(camera_b.K, np.eye(3), np.zeros(3))
    errors_a = reprojection_errors(camera_a.P, refined_points, points_a)
    errors_b = reprojection_errors(refined_camera_b.P, refined_points, points_b)
    mean_error = float(np.mean(np.hstack([errors_a, errors_b])))
    return refined_camera_b, refined_points, mean_error
