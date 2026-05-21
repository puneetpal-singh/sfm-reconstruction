import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((points.shape[0], 1))])


def from_homogeneous(points: np.ndarray) -> np.ndarray:
    scale = points[:, -1:]
    return points[:, :-1] / scale


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean
    distances = np.linalg.norm(centered, axis=1)
    scale = np.sqrt(2.0) / np.mean(distances)
    T = np.array(
        [
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    normalized = (T @ to_homogeneous(points).T).T
    return normalized[:, :2], T


def estimate_fundamental(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if points_a.shape[0] < 8:
        raise ValueError("At least 8 correspondences are required.")

    norm_a, T_a = normalize_points(points_a)
    norm_b, T_b = normalize_points(points_b)

    x1 = norm_a[:, 0]
    y1 = norm_a[:, 1]
    x2 = norm_b[:, 0]
    y2 = norm_b[:, 1]
    A = np.column_stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones_like(x1)])

    _, _, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)
    u, s, vh = np.linalg.svd(F)
    s[-1] = 0.0
    F = u @ np.diag(s) @ vh
    F = T_b.T @ F @ T_a
    return F / np.linalg.norm(F)


def sampson_errors(F: np.ndarray, points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    homogeneous_a = to_homogeneous(points_a)
    homogeneous_b = to_homogeneous(points_b)
    F_a = (F @ homogeneous_a.T).T
    Ft_b = (F.T @ homogeneous_b.T).T
    numerator = np.sum(homogeneous_b * F_a, axis=1) ** 2
    denominator = F_a[:, 0] ** 2 + F_a[:, 1] ** 2 + Ft_b[:, 0] ** 2 + Ft_b[:, 1] ** 2
    return numerator / np.maximum(denominator, 1e-12)


def estimate_fundamental_ransac(
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold: float = 1.5,
    iterations: int = 1000,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    if points_a.shape[0] < 8:
        raise ValueError("At least 8 correspondences are required.")

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(points_a.shape[0], dtype=bool)
    best_F = estimate_fundamental(points_a, points_b)

    for _ in range(iterations):
        sample = rng.choice(points_a.shape[0], size=8, replace=False)
        try:
            F = estimate_fundamental(points_a[sample], points_b[sample])
        except np.linalg.LinAlgError:
            continue
        inliers = sampson_errors(F, points_a, points_b) < threshold**2
        if int(np.sum(inliers)) > int(np.sum(best_inliers)):
            best_inliers = inliers
            best_F = F

    if np.sum(best_inliers) >= 8:
        best_F = estimate_fundamental(points_a[best_inliers], points_b[best_inliers])
    else:
        best_inliers[:] = True

    return best_F, best_inliers


def essential_from_fundamental(F: np.ndarray, K_a: np.ndarray, K_b: np.ndarray) -> np.ndarray:
    E = K_b.T @ F @ K_a
    u, _, vh = np.linalg.svd(E)
    return u @ np.diag([1.0, 1.0, 0.0]) @ vh


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    u, _, vh = np.linalg.svd(E)
    if np.linalg.det(u @ vh) < 0:
        vh = -vh

    W = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R1 = u @ W @ vh
    R2 = u @ W.T @ vh
    t = u[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate_points(P_a: np.ndarray, P_b: np.ndarray, points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    points_3d = []
    for point_a, point_b in zip(points_a, points_b, strict=True):
        x1, y1 = point_a
        x2, y2 = point_b
        A = np.vstack(
            [
                x1 * P_a[2] - P_a[0],
                y1 * P_a[2] - P_a[1],
                x2 * P_b[2] - P_b[0],
                y2 * P_b[2] - P_b[1],
            ]
        )
        _, _, vh = np.linalg.svd(A)
        X = vh[-1]
        points_3d.append(X[:3] / X[3])
    return np.asarray(points_3d)


def project_points(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    projected = (P @ to_homogeneous(points_3d).T).T
    return projected[:, :2] / projected[:, 2:3]


def reprojection_errors(P: np.ndarray, points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    return np.linalg.norm(project_points(P, points_3d) - points_2d, axis=1)


def points_in_front(R: np.ndarray, t: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    depths = (R @ points_3d.T + t.reshape(3, 1))[2]
    return depths > 0
