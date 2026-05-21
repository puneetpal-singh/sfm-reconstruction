import numpy as np

from .camera import Camera, look_at, make_intrinsics
from .geometry import project_points


def create_scene(
    seed: int = 7,
    n_points: int = 180,
    width: int = 1280,
    height: int = 900,
    focal: float = 900.0,
    noise_px: float = 0.5,
) -> dict:
    rng = np.random.default_rng(seed)
    K = make_intrinsics(width, height, focal)

    points_3d = rng.uniform([-1.4, -0.8, 3.5], [1.4, 0.9, 6.0], size=(n_points, 3))
    positions = [
        np.array([-0.8, 0.08, 0.0]),
        np.array([0.8, 0.02, 0.1]),
    ]
    target = np.array([0.0, 0.0, 4.6])

    cameras = []
    observations = []
    for position in positions:
        R, t = look_at(position, target)
        camera = Camera(K, R, t)
        projected = project_points(camera.P, points_3d)
        projected += rng.normal(0.0, noise_px, size=projected.shape)
        cameras.append(camera)
        observations.append(projected)

    return {
        "K": K,
        "points_3d": points_3d,
        "cameras": cameras,
        "observations": observations,
        "width": width,
        "height": height,
    }
