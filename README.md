# SFM Reconstruction

Computer vision project implementing Structure-from-Motion on public multi-view data.

It reconstructs sparse 3D structure and relative camera motion from public multi-view data. The core math is implemented in Python with NumPy/SciPy, while OpenCV is optional for feature extraction on arbitrary image pairs.

## What It Includes

- Oxford VGG Dinosaur public dataset integration
- Multi-view track fusion across several frames
- Normalized eight-point fundamental matrix estimation
- RANSAC epipolar outlier rejection
- Essential matrix decomposition
- Cheirality-based pose selection
- Linear triangulation
- Two-view bundle adjustment
- Reprojection, depth, and statistical outlier filtering
- Reprojection-error evaluation
- Colored sparse point-cloud export to PLY
- Dense two-view stereo point-cloud export to PLY
- Track overlay images for visual inspection
- Optional image-pair reconstruction with OpenCV SIFT/ORB

## Setup

```powershell
cd C:\Users\Puneet\.gemini\antigravity\scratch\SFM
python -m pip install -r requirements.txt
```

Optional real-image support:

```powershell
python -m pip install opencv-python
```

## Public Dataset

The project supports the public Oxford VGG Multi-view Dinosaur dataset:

- Source: https://www.robots.ox.ac.uk/~vgg/data/mview/
- Dataset: Dinosaur
- Data used: 2D tracked points across 36 frames and original PPM images for point colors

Download and reconstruct from the public dataset:

```powershell
python -m sfm.cli vgg-dino --out outputs\vgg_dinosaur
```

Outputs:

```text
outputs/vgg_dinosaur/vgg_dinosaur_reconstruction.ply
outputs/vgg_dinosaur/vgg_dinosaur_report.json
outputs/vgg_dinosaur/view_a_tracks.ppm
outputs/vgg_dinosaur/view_b_tracks.ppm
```

The PLY contains RGB colors sampled from the real VGG Dinosaur images.

Latest verified sparse result:

```text
Views: 24 and 25
Shared tracks: 445
Triangulated colored points: 445
Mean reprojection error: 0.120 px
```

## Dense Reconstruction

Run dense stereo on the VGG Dinosaur pair:

```powershell
python -m sfm.cli dense-vgg-dino --out outputs\vgg_dinosaur_dense
```

The default dense settings produce a few thousand colored points on the Dinosaur pair. Increase `--score-threshold` for cleaner but fewer points, or decrease `--stride` for a denser but slower run.

Outputs:

```text
outputs/vgg_dinosaur_dense/vgg_dinosaur_dense.ply
outputs/vgg_dinosaur_dense/vgg_dinosaur_dense_report.json
outputs/vgg_dinosaur_dense/dense_view_a_matches.ppm
outputs/vgg_dinosaur_dense/dense_view_b_matches.ppm
```

The dense stage uses sparse SfM geometry to constrain correspondence search, then triangulates many photometrically matched pixels into a colored point cloud.

Latest verified dense result:

```text
Dense candidates: 5170
Dense colored points: 3586
Mean NCC score: 0.903
```

## Multi-View Reconstruction

Fuse tracks across more than two VGG Dinosaur views:

```powershell
python -m sfm.cli multiview-vgg-dino --out outputs\vgg_dinosaur_multiview
```

Outputs:

```text
outputs/vgg_dinosaur_multiview/vgg_dinosaur_multiview.ply
outputs/vgg_dinosaur_multiview/vgg_dinosaur_multiview_report.json
```

The multi-view stage reconstructs adjacent view pairs, aligns pair clouds through shared track IDs, fuses repeated tracks, and removes outliers using reprojection, depth percentile, and statistical neighbor filters.

Latest verified multi-view result:

```text
Views: 20 to 28
Pairs reconstructed: 8
Fused tracks before filtering: 1480
Points after filtering: 1417
```

## Run Synthetic Reconstruction

```powershell
python -m sfm.cli demo --out outputs\demo
```

Outputs:

```text
outputs/demo/reconstruction.ply
outputs/demo/report.json
```

Open the `.ply` file in MeshLab, CloudCompare, or Open3D viewer.

## Validation

```powershell
python -m sfm.cli vgg-dino --out outputs\vgg_dinosaur
python -m sfm.cli dense-vgg-dino --out outputs\vgg_dinosaur_dense
python -m sfm.cli multiview-vgg-dino --out outputs\vgg_dinosaur_multiview
```

Validation is based on reprojection error, RANSAC inlier ratio, dense NCC score, fused multi-view track count, outlier-filtered point count, and visual inspection of the exported PLY point clouds.

## Real Image Pair

With OpenCV installed:

```powershell
python -m sfm.cli pair --image-a path\to\view1.jpg --image-b path\to\view2.jpg --out outputs\pair
```

For best results, use two overlapping images of a mostly static scene with enough texture.

## Pipeline

This project implements the educational SfM core:

```text
features/correspondences
-> epipolar geometry
-> relative camera pose
-> triangulated 3D points
-> reprojection metrics
```

The next steps would be full incremental camera registration, global bundle adjustment, and multi-view depth fusion.
