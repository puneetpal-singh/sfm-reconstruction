import argparse
from pathlib import Path

import numpy as np

from .datasets import build_vgg_dinosaur_pair
from .dense import dense_match, depth_percentiles
from .image_pipeline import reconstruct_pair
from .io import camera_pose_payload, draw_points, save_json, save_ply, save_ppm
from .multiview import reconstruct_vgg_multiview
from .reconstruction import TwoViewReconstructor
from .synthetic import create_scene


def run_demo(args: argparse.Namespace) -> None:
    scene = create_scene(seed=args.seed, n_points=args.points, noise_px=args.noise)
    reconstruction = TwoViewReconstructor(scene["K"]).reconstruct(scene["observations"][0], scene["observations"][1])
    valid_points = reconstruction.points_3d[reconstruction.inliers]
    out_dir = Path(args.out)
    save_ply(out_dir / "reconstruction.ply", valid_points)
    save_json(
        out_dir / "report.json",
        {
            "inputPoints": int(scene["points_3d"].shape[0]),
            "triangulatedPoints": int(valid_points.shape[0]),
            "inlierRatio": float(np.mean(reconstruction.inliers)),
            "ransacInlierRatio": float(np.mean(reconstruction.ransac_inliers)),
            "bundleAdjusted": reconstruction.bundle_adjusted,
            "meanReprojectionError": reconstruction.mean_reprojection_error,
            "fundamental": reconstruction.fundamental.tolist(),
            "essential": reconstruction.essential.tolist(),
        },
    )
    print(f"Wrote {out_dir / 'reconstruction.ply'}")
    print(f"Wrote {out_dir / 'report.json'}")


def run_pair(args: argparse.Namespace) -> None:
    report = reconstruct_pair(Path(args.image_a), Path(args.image_b), Path(args.out), args.focal)
    print(f"Matches: {report['matches']}")
    print(f"Triangulated points: {report['triangulatedPoints']}")
    print(f"Mean reprojection error: {report['meanReprojectionError']:.3f}px")
    print(f"Wrote {Path(args.out) / 'pair_reconstruction.ply'}")


def run_vgg_dino(args: argparse.Namespace) -> None:
    data = build_vgg_dinosaur_pair(
        Path(args.data_dir),
        args.view_a,
        args.view_b,
        args.max_tracks,
        args.focal,
    )
    reconstruction = TwoViewReconstructor(data["K"]).reconstruct(data["points_a"], data["points_b"])
    valid_points = reconstruction.points_3d[reconstruction.inliers]
    colors = np.rint((data["colors_a"][reconstruction.inliers].astype(float) + data["colors_b"][reconstruction.inliers].astype(float)) / 2.0).astype(np.uint8)
    out_dir = Path(args.out)
    save_ply(out_dir / "vgg_dinosaur_reconstruction.ply", valid_points, colors)
    save_ppm(out_dir / "view_a_tracks.ppm", draw_points(data["image_a"], data["points_a"][reconstruction.inliers], (255, 30, 30)))
    save_ppm(out_dir / "view_b_tracks.ppm", draw_points(data["image_b"], data["points_b"][reconstruction.inliers], (30, 200, 255)))
    save_json(
        out_dir / "vgg_dinosaur_report.json",
        {
            "dataset": data["source"],
            "sourceUrl": data["sourceUrl"],
            "viewA": data["view_a"] + 1,
            "viewB": data["view_b"] + 1,
            "imageSize": [data["width"], data["height"]],
            "focal": data["focal"],
            "sharedTracks": data["shared_tracks"],
            "usedTracks": data["used_tracks"],
            "triangulatedPoints": int(valid_points.shape[0]),
            "coloredPoints": int(colors.shape[0]),
            "inlierRatio": float(np.mean(reconstruction.inliers)),
            "ransacInlierRatio": float(np.mean(reconstruction.ransac_inliers)),
            "bundleAdjusted": reconstruction.bundle_adjusted,
            "meanReprojectionError": reconstruction.mean_reprojection_error,
            "cameraA": camera_pose_payload(reconstruction.camera_a),
            "cameraB": camera_pose_payload(reconstruction.camera_b),
        },
    )
    print(f"Dataset: {data['source']}")
    print(f"Views: {data['view_a'] + 1} and {data['view_b'] + 1}")
    print(f"Shared tracks: {data['shared_tracks']}")
    print(f"Triangulated points: {valid_points.shape[0]}")
    print(f"Mean reprojection error: {reconstruction.mean_reprojection_error:.3f}px")
    print(f"Wrote {out_dir / 'vgg_dinosaur_reconstruction.ply'}")
    print(f"Wrote {out_dir / 'view_a_tracks.ppm'}")
    print(f"Wrote {out_dir / 'view_b_tracks.ppm'}")


def run_dense_vgg_dino(args: argparse.Namespace) -> None:
    data = build_vgg_dinosaur_pair(
        Path(args.data_dir),
        args.view_a,
        args.view_b,
        args.max_tracks,
        args.focal,
    )
    reconstruction = TwoViewReconstructor(data["K"]).reconstruct(data["points_a"], data["points_b"])
    dense = dense_match(
        data["image_a"],
        data["image_b"],
        reconstruction.fundamental,
        reconstruction.camera_a.P,
        reconstruction.camera_b.P,
        data["points_a"],
        data["points_b"],
        stride=args.stride,
        radius=args.radius,
        search=args.search,
        texture_threshold=args.texture_threshold,
        score_threshold=args.score_threshold,
        max_points=args.max_points,
    )
    out_dir = Path(args.out)
    save_ply(out_dir / "vgg_dinosaur_dense.ply", dense.points_3d, dense.colors)
    save_ppm(out_dir / "dense_view_a_matches.ppm", draw_points(data["image_a"], dense.points_a, (255, 30, 30)))
    save_ppm(out_dir / "dense_view_b_matches.ppm", draw_points(data["image_b"], dense.points_b, (30, 200, 255)))
    save_json(
        out_dir / "vgg_dinosaur_dense_report.json",
        {
            "dataset": data["source"],
            "sourceUrl": data["sourceUrl"],
            "viewA": data["view_a"] + 1,
            "viewB": data["view_b"] + 1,
            "imageSize": [data["width"], data["height"]],
            "focal": data["focal"],
            "sparseTracks": data["used_tracks"],
            "denseCandidates": dense.candidates,
            "densePoints": int(dense.points_3d.shape[0]),
            "meanNccScore": float(np.mean(dense.scores)) if dense.scores.size else 0.0,
            "depthPercentiles": depth_percentiles(dense.points_3d),
            "stride": args.stride,
            "patchRadius": args.radius,
            "searchRadius": args.search,
            "scoreThreshold": args.score_threshold,
            "sparseMeanReprojectionError": reconstruction.mean_reprojection_error,
            "cameraA": camera_pose_payload(reconstruction.camera_a),
            "cameraB": camera_pose_payload(reconstruction.camera_b),
        },
    )
    print(f"Dataset: {data['source']}")
    print(f"Views: {data['view_a'] + 1} and {data['view_b'] + 1}")
    print(f"Dense candidates: {dense.candidates}")
    print(f"Dense points: {dense.points_3d.shape[0]}")
    print(f"Mean NCC score: {(float(np.mean(dense.scores)) if dense.scores.size else 0.0):.3f}")
    print(f"Wrote {out_dir / 'vgg_dinosaur_dense.ply'}")


def run_multiview_vgg_dino(args: argparse.Namespace) -> None:
    result = reconstruct_vgg_multiview(
        Path(args.data_dir),
        args.start_view,
        args.end_view,
        args.focal,
        args.reprojection_threshold,
        args.depth_low,
        args.depth_high,
        args.neighbor_k,
        args.std_ratio,
    )
    out_dir = Path(args.out)
    save_ply(out_dir / "vgg_dinosaur_multiview.ply", result.points_3d, result.colors)
    save_json(
        out_dir / "vgg_dinosaur_multiview_report.json",
        {
            "dataset": "Oxford VGG Multi-view Dinosaur",
            "sourceUrl": "https://www.robots.ox.ac.uk/~vgg/data/mview/",
            "viewRange": [args.start_view, args.end_view],
            "pairCount": len(result.pair_reports),
            "pairReports": result.pair_reports,
            "fusedTracksBeforeFiltering": result.fused_tracks,
            "pointsAfterFiltering": result.filtered_tracks,
            "outlierFilters": {
                "reprojectionThreshold": args.reprojection_threshold,
                "depthPercentiles": [args.depth_low, args.depth_high],
                "neighborK": args.neighbor_k,
                "stdRatio": args.std_ratio,
            },
        },
    )
    print("Dataset: Oxford VGG Multi-view Dinosaur")
    print(f"Views: {args.start_view} to {args.end_view}")
    print(f"Pairs reconstructed: {len(result.pair_reports)}")
    print(f"Fused tracks before filtering: {result.fused_tracks}")
    print(f"Points after filtering: {result.filtered_tracks}")
    print(f"Wrote {out_dir / 'vgg_dinosaur_multiview.ply'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sfm-reconstruction", description="Structure-from-Motion reconstruction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo")
    demo.add_argument("--out", default="outputs/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--points", type=int, default=180)
    demo.add_argument("--noise", type=float, default=0.5)
    demo.set_defaults(func=run_demo)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--image-a", required=True)
    pair.add_argument("--image-b", required=True)
    pair.add_argument("--out", default="outputs/pair")
    pair.add_argument("--focal", type=float)
    pair.set_defaults(func=run_pair)

    vgg = subparsers.add_parser("vgg-dino")
    vgg.add_argument("--data-dir", default="data")
    vgg.add_argument("--out", default="outputs/vgg_dinosaur")
    vgg.add_argument("--view-a", type=int)
    vgg.add_argument("--view-b", type=int)
    vgg.add_argument("--max-tracks", type=int, default=600)
    vgg.add_argument("--focal", type=float, default=850.0)
    vgg.set_defaults(func=run_vgg_dino)

    dense = subparsers.add_parser("dense-vgg-dino")
    dense.add_argument("--data-dir", default="data")
    dense.add_argument("--out", default="outputs/vgg_dinosaur_dense")
    dense.add_argument("--view-a", type=int)
    dense.add_argument("--view-b", type=int)
    dense.add_argument("--max-tracks", type=int, default=600)
    dense.add_argument("--focal", type=float, default=850.0)
    dense.add_argument("--stride", type=int, default=4)
    dense.add_argument("--radius", type=int, default=3)
    dense.add_argument("--search", type=int, default=36)
    dense.add_argument("--texture-threshold", type=float, default=8.0)
    dense.add_argument("--score-threshold", type=float, default=0.8)
    dense.add_argument("--max-points", type=int, default=12000)
    dense.set_defaults(func=run_dense_vgg_dino)

    multiview = subparsers.add_parser("multiview-vgg-dino")
    multiview.add_argument("--data-dir", default="data")
    multiview.add_argument("--out", default="outputs/vgg_dinosaur_multiview")
    multiview.add_argument("--start-view", type=int, default=20)
    multiview.add_argument("--end-view", type=int, default=28)
    multiview.add_argument("--focal", type=float, default=850.0)
    multiview.add_argument("--reprojection-threshold", type=float, default=1.5)
    multiview.add_argument("--depth-low", type=float, default=1.0)
    multiview.add_argument("--depth-high", type=float, default=99.0)
    multiview.add_argument("--neighbor-k", type=int, default=12)
    multiview.add_argument("--std-ratio", type=float, default=2.0)
    multiview.set_defaults(func=run_multiview_vgg_dino)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
