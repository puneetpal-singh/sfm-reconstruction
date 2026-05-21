from pathlib import Path
import tarfile
from urllib.request import urlretrieve

import numpy as np

from .camera import make_intrinsics

VGG_DINO_ROOT = "https://www.robots.ox.ac.uk/~vgg/data/dino"
VGG_DINO_TRACKS_URL = f"{VGG_DINO_ROOT}/viff.xy"
VGG_DINO_README_URL = f"{VGG_DINO_ROOT}/README.txt"
VGG_DINO_IMAGES_URL = f"{VGG_DINO_ROOT}/images.tar.gz"


def download_vgg_dinosaur(data_dir: Path) -> dict:
    target = data_dir / "vgg_dinosaur"
    target.mkdir(parents=True, exist_ok=True)
    tracks_path = target / "viff.xy"
    readme_path = target / "README.txt"
    archive_path = target / "images.tar.gz"

    if not tracks_path.exists():
        urlretrieve(VGG_DINO_TRACKS_URL, tracks_path)
    if not readme_path.exists():
        urlretrieve(VGG_DINO_README_URL, readme_path)
    if not archive_path.exists():
        urlretrieve(VGG_DINO_IMAGES_URL, archive_path)
    if not (target / "viff.000.ppm").exists():
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(target)

    return {
        "tracks": tracks_path,
        "readme": readme_path,
        "archive": archive_path,
        "imageDir": target,
        "source": "Oxford VGG Multi-view Dinosaur",
        "sourceUrl": "https://www.robots.ox.ac.uk/~vgg/data/mview/",
    }


def load_vgg_dinosaur(data_dir: Path, focal: float = 850.0) -> dict:
    paths = download_vgg_dinosaur(data_dir)
    tracks = load_vgg_dinosaur_tracks(paths["tracks"])
    width = int(np.ceil(np.max(tracks[:, :, 0][tracks[:, :, 0] >= 0]) + 20))
    height = int(np.ceil(np.max(tracks[:, :, 1][tracks[:, :, 1] >= 0]) + 20))
    images = [read_ppm(paths["imageDir"] / f"viff.{index:03d}.ppm") for index in range(36)]

    return {
        "K": make_intrinsics(width, height, focal),
        "tracks": tracks,
        "images": images,
        "width": width,
        "height": height,
        "focal": focal,
        "source": paths["source"],
        "sourceUrl": paths["sourceUrl"],
    }


def read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        magic = file.readline().strip()
        if magic != b"P6":
            raise ValueError("Only binary P6 PPM images are supported.")
        line = file.readline()
        while line.startswith(bytes([35])):
            line = file.readline()
        width, height = [int(value) for value in line.split()]
        max_value = int(file.readline())
        if max_value != 255:
            raise ValueError("Only 8-bit PPM images are supported.")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(height, width, 3)


def sample_colors(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    xy = np.rint(points).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    return image[xy[:, 1], xy[:, 0]]


def load_vgg_dinosaur_tracks(tracks_path: Path) -> np.ndarray:
    raw = np.loadtxt(tracks_path)
    if raw.shape[1] != 72:
        raise ValueError("Expected 36 views with x/y coordinates.")
    return raw.reshape(raw.shape[0], 36, 2)


def best_track_pair(tracks: np.ndarray, min_shared: int = 8) -> tuple[int, int, np.ndarray]:
    valid = np.all(tracks >= 0, axis=2)
    best_count = -1
    best_pair = (0, 1)
    best_mask = valid[:, 0] & valid[:, 1]

    for left in range(valid.shape[1]):
        for right in range(left + 1, valid.shape[1]):
            mask = valid[:, left] & valid[:, right]
            count = int(np.sum(mask))
            if count > best_count:
                best_count = count
                best_pair = (left, right)
                best_mask = mask

    if best_count < min_shared:
        raise RuntimeError("No view pair has enough shared tracks.")

    return best_pair[0], best_pair[1], best_mask


def build_vgg_dinosaur_pair(
    data_dir: Path,
    view_a: int | None = None,
    view_b: int | None = None,
    max_tracks: int = 600,
    focal: float = 850.0,
) -> dict:
    paths = download_vgg_dinosaur(data_dir)
    tracks = load_vgg_dinosaur_tracks(paths["tracks"])

    if view_a is None or view_b is None:
        view_a, view_b, mask = best_track_pair(tracks)
    else:
        mask = np.all(tracks[:, view_a] >= 0, axis=1) & np.all(tracks[:, view_b] >= 0, axis=1)

    points_a = tracks[mask, view_a]
    points_b = tracks[mask, view_b]

    if points_a.shape[0] > max_tracks:
        points_a = points_a[:max_tracks]
        points_b = points_b[:max_tracks]

    width = int(np.ceil(np.max(tracks[:, :, 0][tracks[:, :, 0] >= 0]) + 20))
    height = int(np.ceil(np.max(tracks[:, :, 1][tracks[:, :, 1] >= 0]) + 20))

    return {
        "K": make_intrinsics(width, height, focal),
        "points_a": points_a,
        "points_b": points_b,
        "colors_a": sample_colors(read_ppm(paths["imageDir"] / f"viff.{view_a:03d}.ppm"), points_a),
        "colors_b": sample_colors(read_ppm(paths["imageDir"] / f"viff.{view_b:03d}.ppm"), points_b),
        "image_a": read_ppm(paths["imageDir"] / f"viff.{view_a:03d}.ppm"),
        "image_b": read_ppm(paths["imageDir"] / f"viff.{view_b:03d}.ppm"),
        "view_a": int(view_a),
        "view_b": int(view_b),
        "shared_tracks": int(np.sum(mask)),
        "used_tracks": int(points_a.shape[0]),
        "width": width,
        "height": height,
        "focal": focal,
        "source": paths["source"],
        "sourceUrl": paths["sourceUrl"],
    }
