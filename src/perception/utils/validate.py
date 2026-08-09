import torch
import time

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
                t_feat = torch.flip(t_out['bev_features'], dims=[-1])

                s_out = student(val_cam, val_int, val_ext)
                v_loss, v_dict = criterion(s_out, t_feat, val_gt, depth_labels=val_depths)
                
            val_loss_total += v_dict['loss_total']

            preds_prob = torch.sigmoid(s_out['bev_occupancy'])
            preds_binary = (preds_prob > 0.4).float()
            targets_binary = (val_gt > 0.3).float()
            
            intersection = (preds_binary * targets_binary).sum()
            union = preds_binary.sum() + targets_binary.sum() - intersection
            
            total_intersection += intersection.item()
            total_union += union.item()
                
    avg_val_loss = val_loss_total / total_batches
    val_score = total_intersection / (total_union + 1e-6)
    
    return avg_val_loss, val_score