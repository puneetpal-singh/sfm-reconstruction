import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = np.full((points.shape[0], 3), 235, dtype=np.uint8)

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_ppm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(image.astype(np.uint8).tobytes())


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for x, y in np.rint(points).astype(int):
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            canvas[y - 1 : y + 2, x - 1 : x + 2] = color
    return canvas


def camera_pose_payload(camera) -> dict:
    return {
        "R": camera.R.tolist(),
        "t": camera.t.tolist(),
        "rotationVector": Rotation.from_matrix(camera.R).as_rotvec().tolist(),
    }
