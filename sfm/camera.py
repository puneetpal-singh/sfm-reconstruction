from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R_cw = np.vstack([right, true_up, forward])
    t = -R_cw @ position
    return R_cw, t


def make_intrinsics(width: int, height: int, focal: float) -> np.ndarray:
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
