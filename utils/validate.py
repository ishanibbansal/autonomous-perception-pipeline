import torch
import time

def validate_model(model, dataloader, criterion, encoder, device):
    """
    Optimized validation loop for Multi-Modal LiDAR + Camera BEV model.
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
            
            # --- FIX: Use .item() to prevent graph accumulation memory leak ---
            total_loss += loss.item()
            
            preds_prob = torch.sigmoid(predictions['bev_occupancy'])
            preds_binary = (preds_prob > 0.4).float()
            targets_binary = (targets_gpu['bev_occupancy'] > 0.3).float()
            
            intersection = (preds_binary * targets_binary).sum()
            union = preds_binary.sum() + targets_binary.sum() - intersection
            
            total_intersection += intersection.item()
            total_union += union.item()
            
            # --- Progress Tracking ---
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - val_start_time
                print(f"  -> Validated Batch {batch_idx + 1:03d}/{total_batches} | Elapsed: {elapsed:.1f}s")
            
    avg_loss = total_loss / total_batches
    global_iou = total_intersection / (total_union + 1e-6)
    
    print(f"--- Validation Complete in {time.time() - val_start_time:.1f}s ---\n")
    
    return avg_loss, global_iou