import torch
from utils.metrics import calculate_map

def apply_nms(boxes, distance_threshold=1.5):
    """
    Applies Non-Maximum Suppression based on confidence and BEV center distance.
    boxes shape: [N, 8] -> [Conf, X, Y, Z, L, W, H, Heading]
    """
    if len(boxes) == 0:
        return boxes
        
    # Sort boxes by confidence score descending
    scores = boxes[:, 0]
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    
    keep = []
    while len(boxes) > 0:
        keep.append(boxes[0])
        if len(boxes) == 1:
            break
            
        current = boxes[0]
        rest = boxes[1:]
        
        # Calculate Euclidean distance in BEV (X, Y) to suppress redundant overlapping cells
        dists = torch.sqrt((rest[:, 1] - current[1])**2 + (rest[:, 2] - current[2])**2)
        
        # Keep boxes that are further apart than the distance threshold
        overlap_mask = dists > distance_threshold
        boxes = rest[overlap_mask]
        
    return torch.stack(keep) if keep else torch.zeros((0, 8), device=boxes.device)


def decode_predictions(preds, conf_thresh=0.1):
    """
    Scans the dense output grid and extracts boxes with high confidence.
    Converts relative grid offsets and depth back into absolute 3D world coordinates.
    Returns a list of tensors of shape [N, 8] -> [Conf, X, Y, Z, L, W, H, Heading]
    """
    IMAGE_WIDTH = 960.0
    IMAGE_HEIGHT = 640.0
    FOCAL_LENGTH = 1000.0
    CAMERA_HEIGHT_OFFSET = 1.5
    STRIDE = 32.0 

    keys = list(preds.keys())
    cls_key = next(k for k in keys if 'class' in k.lower() or 'cls' in k.lower())
    loc_key = next(k for k in keys if 'loc' in k.lower())
    dim_key = next(k for k in keys if 'dim' in k.lower())
    ori_key = next(k for k in keys if 'ori' in k.lower() or 'orientation' in k.lower())
    
    batch_size = preds[cls_key].shape[0]
    cls_probs = torch.sigmoid(preds[cls_key])  
    
    batch_boxes = []
    
    for b in range(batch_size):
        vehicle_probs = cls_probs[b][0] 
        mask = vehicle_probs > conf_thresh
        conf = vehicle_probs[mask]
        
        if len(conf) == 0:
            batch_boxes.append(torch.zeros((0, 8), device=conf.device))
            continue
            
        # Get the grid cell indices [Y, X] for valid detections
        y_indices, x_indices = torch.where(mask)
        
        loc = preds[loc_key][b][:, y_indices, x_indices] # Shape: [3, N]
        dim = preds[dim_key][b][:, y_indices, x_indices] # Shape: [3, N]
        ori = preds[ori_key][b][:, y_indices, x_indices] # Shape: [2, N]
        
        # 1. Extract Offsets and Depth
        offset_x = loc[0, :]
        offset_y = loc[1, :]
        cx = loc[2, :] # Forward depth
        
        # 2. Reconstruct 2D Pixels
        pixel_x = (x_indices.float() + offset_x) * STRIDE
        pixel_y = (y_indices.float() + offset_y) * STRIDE
        
        # 3. Project back to absolute 3D World Coordinates
        cy = ((IMAGE_WIDTH / 2.0) - pixel_x) * cx / FOCAL_LENGTH
        cz = ((IMAGE_HEIGHT / 2.0) - pixel_y) * cx / FOCAL_LENGTH + CAMERA_HEIGHT_OFFSET
        
        # Stack absolute coordinates back into standard format
        abs_loc = torch.stack([cx, cy, cz], dim=0)
        
        # Convert Heading
        sin_val = ori[0, :]
        cos_val = ori[1, :]
        heading = torch.atan2(sin_val, cos_val).unsqueeze(0) 
        
        # Concatenate into [N, 8]
        boxes = torch.cat([conf.unsqueeze(1), abs_loc.t(), dim.t(), heading.t()], dim=1)
        
        # Apply NMS to clean up overlapping duplicate clusters
        boxes = apply_nms(boxes, distance_threshold=1.5)
        
        batch_boxes.append(boxes)
        
    return batch_boxes


def validate_model(model, dataloader, criterion, encoder, device, conf_thresh=0.1):
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
            
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            targets = {k: v.to(device) for k, v in encoded_targets.items()}
            
            original_mask = targets['mask'].bool()
            vehicle_class_mask = (targets['class'][:, 0, :, :] == 1).unsqueeze(1)
            targets['mask'] = original_mask & vehicle_class_mask
            
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            val_loss += loss.item()
            
            decoded_preds = decode_predictions(predictions, conf_thresh=conf_thresh)
            
            for b in range(len(images)):
                num_valid = valid_boxes[b].item()
                
                if num_valid > 0:
                    valid_raw_boxes = raw_bboxes[b, :num_valid]
                    is_vehicle_mask = valid_raw_boxes[:, 0] == 1 
                    target_boxes = valid_raw_boxes[is_vehicle_mask][:, 3:10].to(device)
                else:
                    target_boxes = torch.zeros((0, 7), device=device)
                
                frame_map = calculate_map(decoded_preds[b], target_boxes, iou_threshold=0.25)
                total_map += frame_map
                
    avg_val_loss = val_loss / len(dataloader)
    avg_map = total_map / len(dataloader.dataset)
    
    return avg_val_loss, avg_map