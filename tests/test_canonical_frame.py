import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from src.perception.losses.centernet_loss import WaymoDetectionLoss
from src.perception.models.student_bev import LiftSplatViewTransformer, canonical_pose_sample_grid, se3_inverse
from src.perception.utils.target_encoder import BEVGridEncoder


def _eye_pose(batch, device='cpu'):
    eye = torch.eye(4, device=device).view(1, 4, 4).expand(batch, -1, -1).contiguous()
    return eye


def test_identity_pose_samples_same_cell():
    current = _eye_pose(2)
    past = _eye_pose(2)
    grid = canonical_pose_sample_grid(2, 160, 160, current, past)
    rows = torch.arange(160)
    cols = torch.arange(160)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')
    expected_x = 2.0 * (col_grid.float() + 0.5) / 160.0 - 1.0
    expected_y = 2.0 * (row_grid.float() + 0.5) / 160.0 - 1.0
    assert torch.allclose(grid[0, ..., 0], expected_x, atol=1e-5)
    assert torch.allclose(grid[0, ..., 1], expected_y, atol=1e-5)
    assert torch.allclose(grid[0], grid[1])


def test_known_forward_shift_pulls_features_from_closer_cell():
    # Current ego is 2.0 m ahead of the past ego. A current cell at x was at x+2
    # in the past frame, so the sample moves toward row 0 (+70 m), a smaller row index.
    current = _eye_pose(1)
    current = current.clone()
    current[0, 0, 3] = 2.0
    past = _eye_pose(1)
    grid = canonical_pose_sample_grid(1, 160, 160, current, past)
    identity = canonical_pose_sample_grid(1, 160, 160, past, past)
    cells = 2.0 / (70.0 / 160.0)
    delta = grid[0, 80, 80, 1] - identity[0, 80, 80, 1]
    assert torch.allclose(delta, torch.tensor(-2.0 * cells / 160.0), atol=1e-4)
    # Lateral translation of +1 m in the past frame is one-half cell toward +Y (col 0).
    past_lat = _eye_pose(1).clone()
    past_lat[0, 1, 3] = -1.0
    current_lat = _eye_pose(1)
    lateral = canonical_pose_sample_grid(1, 160, 160, current_lat, past_lat)
    # T_rel = inv(past) @ current puts the current point at y=+1 in the past frame,
    # which is a smaller column index.
    lat_delta = lateral[0, 80, 80, 0] - identity[0, 80, 80, 0]
    assert lat_delta < 0
    assert torch.allclose(lat_delta.abs(), torch.tensor(2.0 * (1.0 / 0.5) / 160.0), atol=1e-4)


def test_pullback_uses_past_then_current_not_the_opposite_multiply():
    current = _eye_pose(1).clone()
    current[0, 0, 3] = 3.0
    past = _eye_pose(1).clone()
    past[0, 1, 3] = 1.0
    expected = se3_inverse(past) @ current
    wrong = se3_inverse(current) @ past
    assert not torch.allclose(expected, wrong)
    point = torch.tensor([10.0, 0.0, 0.0, 1.0])
    pulled = (expected @ point)[0]
    assert torch.allclose(pulled[:2], torch.tensor([13.0, -1.0]), atol=1e-5)


def test_horizontal_flip_reflects_camera_to_vehicle():
    # Camera origin sits 1 m to the left of the rear axle. A horizontal flip
    # must put that origin 1 m to the right, and swap the image-left ray.
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[1, 3] = 1.0
    flip_y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
    reflected = flip_y @ extrinsics @ flip_y
    origin = reflected @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert np.allclose(origin[:3], np.array([0.0, -1.0, 0.0], dtype=np.float32))
    # Image-left in the already-converted Waymo camera frame is -Y. Reflecting the
    # camera point before the extrinsic, then the vehicle point after it, flips that ray.
    cam_point = np.array([10.0, -2.0, 1.0, 1.0], dtype=np.float32)
    flipped_point = reflected @ (flip_y @ cam_point)
    unflipped_point = extrinsics @ cam_point
    assert np.allclose(flipped_point[:3], flip_y[:3, :3] @ unflipped_point[:3])


def test_target_encoder_drops_border_and_supervises_center_only():
    encoder = BEVGridEncoder()
    # x just inside the far edge floors outside [0, 159] under the inverted index.
    boxes = torch.zeros(1, 2, 10)
    boxes[0, 0, 3] = 70.0
    boxes[0, 0, 4] = 0.0
    boxes[0, 0, 6:9] = torch.tensor([4.0, 1.8, 1.5])
    boxes[0, 1, 3] = 35.0
    boxes[0, 1, 4] = 0.0
    boxes[0, 1, 6:9] = torch.tensor([4.0, 1.8, 1.5])
    boxes[0, 1, 9] = 0.3
    encoded = encoder.encode(boxes, torch.tensor([2]))
    assert encoded['mask'].sum().item() == 1.0
    occupied = encoded['mask'][0, 0].nonzero(as_tuple=False)
    cy, cx = occupied[0].tolist()
    assert encoded['dimensions'][0, 0, cy, cx].item() == 4.0
    assert encoded['dimensions'][0, 0].sum().item() == 4.0
    assert encoded['bev_occupancy'][0, 0].sum().item() > 1.0
    # Fractional remainder is stored, and only at the center.
    assert 0.0 <= encoded['offset'][0, 0, cy, cx].item() < 1.0
    assert encoded['offset'][0].abs().sum().item() == encoded['offset'][0, :, cy, cx].abs().sum().item()


def test_canonical_splat_index_matches_floor_formula_not_legacy_truncation():
    # 10.1 m / 0.4375 m = 23.0857. Legacy truncates then subtracts from 159.
    # Canonical floors the already-inverted continuous index, which is one cell off.
    res = 70.0 / 160.0
    continuous = 10.1 / res
    legacy = (160 - 1) - int(continuous)
    canonical = math.floor((160 - 1) - continuous)
    assert legacy != canonical
    splat = LiftSplatViewTransformer(legacy_frame=False)
    geom = torch.zeros(1, 1, 1, 1, 3)
    geom[0, 0, 0, 0, 0] = 10.1
    geom[0, 0, 0, 0, 1] = 0.0
    geom[0, 0, 0, 0, 2] = 0.0
    volume = torch.ones(1, 1, 1, 1, 1)
    pooled = splat.voxel_pooling(geom, volume)
    row = int(pooled[0, 0].argmax().item() // 160)
    assert row == canonical


def test_teacher_focal_normalizes_by_positive_count():
    loss_fn = WaymoDetectionLoss()
    preds = {
        'bev_occupancy': torch.zeros(1, 1, 4, 4, requires_grad=True),
        'dimensions': torch.zeros(1, 3, 4, 4),
        'orientation': torch.zeros(1, 2, 4, 4),
        'offset': torch.zeros(1, 2, 4, 4),
    }
    targets = {
        'bev_occupancy': torch.zeros(1, 1, 4, 4),
        'dimensions': torch.zeros(1, 3, 4, 4),
        'orientation': torch.zeros(1, 2, 4, 4),
        'offset': torch.zeros(1, 2, 4, 4),
        'mask': torch.zeros(1, 1, 4, 4),
    }
    targets['bev_occupancy'][0, 0, 1, 1] = 1.0
    targets['mask'][0, 0, 1, 1] = 1.0
    out = loss_fn(preds, targets)
    assert set(out) == {'loss_total', 'loss_focal', 'loss_dice', 'loss_dim', 'loss_ori', 'loss_offset'}
    out['loss_total'].backward()
    assert preds['bev_occupancy'].grad is not None
    # A second positive peak must cut the per-peak focal term roughly in half, not divide by 16 cells.
    one = out['loss_focal'].detach()
    preds2 = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in preds.items()}
    targets['bev_occupancy'][0, 0, 2, 2] = 1.0
    two = loss_fn(preds2, targets)['loss_focal']
    assert two < one
    assert two > one * 0.4


if __name__ == '__main__':
    test_identity_pose_samples_same_cell()
    test_known_forward_shift_pulls_features_from_closer_cell()
    test_pullback_uses_past_then_current_not_the_opposite_multiply()
    test_horizontal_flip_reflects_camera_to_vehicle()
    test_target_encoder_drops_border_and_supervises_center_only()
    test_canonical_splat_index_matches_floor_formula_not_legacy_truncation()
    test_teacher_focal_normalizes_by_positive_count()
    print('canonical frame tests passed')
