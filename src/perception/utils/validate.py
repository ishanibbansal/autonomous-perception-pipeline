import os
import glob
import torch
import time
from torch.utils.data import DataLoader, ConcatDataset

def validate_model(model, dataloader, criterion, encoder, device):
    """
    Optimized validation loop for Multi-Modal LiDAR + Camera BEV model (Teacher).
    """
    model.eval()
    
    total_loss = 0.0
    total_intersection = 0.0
    total_union = 0.0
    
    total_batches = len(dataloader)
    print(f"\n--- Starting Validation ({total_batches} batches) ---")
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            lidar_points = batch['lidar_points'].to(device, non_blocking=True)
            batch_indices = batch['batch_indices'].to(device, non_blocking=True)
            
            camera_images = batch['camera_images'].to(device, non_blocking=True)
            lidar_uvs = batch['lidar_uvs'].to(device, non_blocking=True)
            
            encoded_targets = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
            targets_gpu = {k: v.to(device, non_blocking=True) for k, v in encoded_targets.items()}
            
            with torch.amp.autocast('cuda'):
                predictions = model(lidar_points, batch_indices, camera_images, lidar_uvs)
                loss = criterion(predictions, targets_gpu)
            
            total_loss += loss.item()
            
            preds_prob = torch.sigmoid(predictions['bev_occupancy'])
            
            # The Teacher's predictions are physically mirrored. Flip them to match Ground Truth alignment.
            preds_prob = torch.flip(preds_prob, dims=[-1])
            
            preds_binary = (preds_prob > 0.4).float()
            targets_binary = (targets_gpu['bev_occupancy'] > 0.3).float()
            
            intersection = (preds_binary * targets_binary).sum()
            union = preds_binary.sum() + targets_binary.sum() - intersection
            
            total_intersection += intersection.item()
            total_union += union.item()
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - val_start_time
                print(f"  -> Validated Batch {batch_idx + 1:03d}/{total_batches} | Elapsed: {elapsed:.1f}s")
            
    avg_loss = total_loss / total_batches
    global_iou = total_intersection / (total_union + 1e-6)
    
    print(f"--- Validation Complete in {time.time() - val_start_time:.1f}s ---\n")
    
    return avg_loss, global_iou

def validate_student(student, teacher, dataloader, criterion, encoder, device):
    student.eval()
    
    val_loss_total = 0.0
    total_intersection = 0.0
    total_union = 0.0
    total_batches = len(dataloader)
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            val_cam = batch['camera_images'].to(device, non_blocking=True)
            val_lidar = batch['lidar_points'].to(device, non_blocking=True)
            val_indices = batch['batch_indices'].to(device, non_blocking=True)
            val_uvs = batch['lidar_uvs'].to(device, non_blocking=True)
            
            val_int = batch['intrinsics'].to(device, non_blocking=True)
            val_ext = batch['extrinsics'].to(device, non_blocking=True)
            val_depths = batch['depth_labels'].to(device, non_blocking=True)
            
            encoded_val = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
            val_gt = encoded_val['bev_occupancy'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                t_out = teacher(val_lidar, val_indices, val_cam, val_uvs)
                
                t_feat = t_out.get('bev_features', None)
                if t_feat is not None:
                    t_feat = torch.flip(t_feat, dims=[-1])

                s_out = student(val_cam, val_int, val_ext)
                
                # Clean explicit keyword arguments
                v_loss, v_dict = criterion(
                    s_out, 
                    t_feat, 
                    ground_truth=val_gt, 
                    teacher_logits=None, 
                    depth_labels=val_depths
                )
                
            val_loss_total += v_dict['loss_total']

            preds_prob = torch.sigmoid(s_out['bev_occupancy'])
            
            preds_binary = (preds_prob > 0.15).float()
            targets_binary = (val_gt > 0.3).float()
            
            intersection = (preds_binary * targets_binary).sum()
            union = preds_binary.sum() + targets_binary.sum() - intersection
            
            total_intersection += intersection.item()
            total_union += union.item()
                
    avg_val_loss = val_loss_total / total_batches
    val_score = total_intersection / (total_union + 1e-6)
    
    return avg_val_loss, val_score


# ==========================================
# Standalone Testing Block
# ==========================================
if __name__ == '__main__':
    import sys
    
    # Fix absolute imports by appending the project root (3 levels up from this file) to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    from src.perception.models.teacher_bev import WaymoBEVDetector
    from src.perception.utils.dataset import WaymoDataset, waymo_collate_fn
    from src.perception.utils.target_encoder import BEVGridEncoder
    from src.perception.models.teacher_bev import WaymoBEVDetector
    from src.perception.utils.dataset import WaymoDataset, waymo_collate_fn
    from src.perception.utils.target_encoder import BEVGridEncoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing standalone Teacher validation on: {device}")
    
    # 1. Load the frozen Teacher Model
    teacher_model = WaymoBEVDetector().to(device)
    checkpoint_path = 'best_waymo_bev_checkpoint.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: '{checkpoint_path}' not found! Testing with randomized weights.")
        
    teacher_model.eval()

    # 2. Setup Validation Dataloader
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        print("Error: No validation tfrecords found in data/raw/val/")
        exit()
        
    print(f"Found {len(val_files)} validation records. Building dataset...")
    val_datasets = [WaymoDataset(tfrecord_path=f, is_train=False, num_sweeps=3) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True, 
        collate_fn=waymo_collate_fn
    )
    
    # 3. Setup Target Encoder & Dummy Loss
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    
    # We bypass the complex loss calculation here since we only care about the physical IoU score
    def dummy_criterion(preds, targets):
        return torch.tensor(0.0, device=device)
        
    # 4. Execute the validation loop
    avg_loss, final_iou = validate_model(teacher_model, val_dataloader, dummy_criterion, encoder, device)
    
    print(f"\n========================================")
    print(f"  Final Standalone Teacher IoU: {final_iou:.4f}")
    print(f"========================================")