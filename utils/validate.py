import torch
from utils.metrics import calculate_map

def decode_predictions(preds, conf_thresh=0.25):
    """
    Scans the dense output grid and extracts boxes with high confidence.
    Returns a list of tensors of shape [N, 7] -> [Conf, X, Y, Z, L, W, H]
    """
    # Dynamically find the correct keys in the predictions dictionary
    keys = list(preds.keys())
    cls_key = next(k for k in keys if 'class' in k.lower() or 'cls' in k.lower())
    loc_key = next(k for k in keys if 'loc' in k.lower())
    dim_key = next(k for k in keys if 'dim' in k.lower())
    
    batch_size = preds[cls_key].shape[0]
    cls_probs = torch.sigmoid(preds[cls_key])  # Convert logits to probabilities
    
    batch_boxes = []
    
    for b in range(batch_size):
        # Get the highest class probability for each grid cell [40, 60]
        max_probs, _ = torch.max(cls_probs[b], dim=0)
        
        # Create a boolean mask of cells that pass the confidence threshold
        mask = max_probs > conf_thresh
        
        # Extract the confidence scores for the activated cells [N]
        conf = max_probs[mask]
        
        if len(conf) == 0:
            batch_boxes.append(torch.zeros((0, 7), device=conf.device))
            continue
            
        # Extract location and dimension predictions for the activated cells [3, N]
        loc = preds[loc_key][b][:, mask]
        dim = preds[dim_key][b][:, mask]
        
        # Transpose to [N, 3] and combine into [N, 7]
        boxes = torch.cat([conf.unsqueeze(1), loc.t(), dim.t()], dim=1)
        batch_boxes.append(boxes)
        
    return batch_boxes

def validate_model(model, dataloader, criterion, encoder, device):
    """
    Evaluates the model on a holdout dataset, calculating both Loss and mAP.
    """
    model.eval()
    val_loss = 0.0
    total_map = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['front_image'].to(device, dtype=torch.float32)
            raw_bboxes = batch['bboxes'].to(device)
            valid_boxes = batch['num_valid_boxes']
            
            # 1. Loss Targets
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            targets = {k: v.to(device) for k, v in encoded_targets.items()}
            
            # 2. Forward Pass & Loss
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            val_loss += loss.item()
            
            # 3. Decode Predictions for mAP
            decoded_preds = decode_predictions(predictions, conf_thresh=0.25)
            
            # 4. Calculate mAP per frame in the batch
            for b in range(len(images)):
                num_valid = valid_boxes[b].item()
                target_boxes = raw_bboxes[b, :num_valid]
                
                frame_map = calculate_map(decoded_preds[b], target_boxes, iou_threshold=0.5)
                total_map += frame_map
                
    avg_val_loss = val_loss / len(dataloader)
    avg_map = total_map / len(dataloader.dataset)
    
    return avg_val_loss, avg_map