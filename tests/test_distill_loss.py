import torch
from src.perception.losses.distill_loss import CrossModalDistillationLoss

def test_distill_loss_backward_compatibility():
    loss_fn = CrossModalDistillationLoss()
    B, H, W = 2, 160, 160
    
    student_outputs = {
        'bev_features': torch.randn(B, 64, H, W, requires_grad=True),
        'bev_occupancy': torch.randn(B, 1, H, W, requires_grad=True),
        'depth_logits': torch.randn(B, 48, 80, 120, requires_grad=True),
    }
    teacher_features = torch.randn(B, 64, H, W)
    ground_truth = torch.zeros(B, 1, H, W)
    depth_labels = torch.randint(0, 48, (B, 80, 120))
    
    loss, loss_dict = loss_fn(
        student_outputs, teacher_features, ground_truth,
        teacher_logits=None, depth_labels=depth_labels
    )
    
    assert loss > 0.0
    assert 'loss_dim' in loss_dict and loss_dict['loss_dim'] == 0.0
    loss.backward()
    assert student_outputs['bev_occupancy'].grad is not None
    assert student_outputs['bev_features'].grad is not None

def test_distill_loss_with_regression_heads():
    loss_fn = CrossModalDistillationLoss(alpha_dim=2.0, alpha_ori=1.0, alpha_offset=1.0)
    B, H, W = 2, 160, 160
    
    student_outputs = {
        'bev_features': torch.randn(B, 64, H, W, requires_grad=True),
        'bev_occupancy': torch.randn(B, 1, H, W, requires_grad=True),
        'dimensions': torch.randn(B, 3, H, W, requires_grad=True),
        'orientation': torch.randn(B, 2, H, W, requires_grad=True),
        'offset': torch.randn(B, 2, H, W, requires_grad=True),
        'depth_logits': torch.randn(B, 48, 80, 120, requires_grad=True),
    }
    
    teacher_features = torch.randn(B, 64, H, W)
    teacher_logits = torch.randn(B, 1, H, W)
    depth_labels = torch.randint(0, 48, (B, 80, 120))
    
    # Ground truth targets
    targets = {
        'bev_occupancy': torch.zeros(B, 1, H, W),
        'dimensions': torch.ones(B, 3, H, W) * 4.5,
        'orientation': torch.zeros(B, 2, H, W),
        'offset': torch.zeros(B, 2, H, W),
        'mask': torch.zeros(B, 1, H, W),
    }
    
    # Put a positive box target in camera FOV
    targets['bev_occupancy'][:, :, 80, 80] = 1.0
    targets['mask'][:, :, 80, 80] = 1.0
    
    loss, loss_dict = loss_fn(
        student_outputs,
        teacher_features,
        ground_truth=targets['bev_occupancy'],
        teacher_logits=teacher_logits,
        depth_labels=depth_labels,
        targets=targets
    )
    
    assert loss > 0.0
    assert loss_dict['loss_dim'] > 0.0
    assert loss_dict['loss_ori'] > 0.0
    assert loss_dict['loss_offset'] > 0.0
    assert loss_dict['loss_reg'] > 0.0
    
    loss.backward()
    
    # Verify gradients flow to all regression heads
    assert student_outputs['dimensions'].grad is not None
    assert student_outputs['orientation'].grad is not None
    assert student_outputs['offset'].grad is not None
    assert torch.any(student_outputs['dimensions'].grad != 0)
    assert torch.any(student_outputs['orientation'].grad != 0)
    assert torch.any(student_outputs['offset'].grad != 0)
    print("All regression loss tests passed successfully!")

if __name__ == '__main__':
    test_distill_loss_backward_compatibility()
    test_distill_loss_with_regression_heads()
    print("SUCCESS: CrossModalDistillationLoss regression tests passed!")
