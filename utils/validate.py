import torch

def validate_model(model, dataloader, criterion, encoder, device):
    """
    Optimized validation loop for BEV Occupancy model using 
    GPU-side tensor accumulation to prevent CPU-GPU sync stalls.
    """
    model.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    total_intersection = torch.tensor(0.0, device=device)
    total_union = torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['front_image'].to(device, dtype=torch.float32, non_blocking=True)
            targets_dict = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
            targets = targets_dict['bev_occupancy'].to(device, non_blocking=True)
            
            predictions = model(images)
            
            loss = criterion(predictions['bev_occupancy'], targets)
            total_loss += loss
            
            preds_prob = torch.sigmoid(predictions['bev_occupancy'])
            preds_binary = (preds_prob > 0.5).float()
            
            intersection = (preds_binary * targets).sum()
            union = preds_binary.sum() + targets.sum() - intersection
            
            total_intersection += intersection
            total_union += union
            
    avg_loss = (total_loss / len(dataloader)).item()
    global_iou = (total_intersection / (total_union + 1e-6)).item()
    
    return avg_loss, global_iou