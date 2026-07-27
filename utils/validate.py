import torch
import torch.nn.functional as F
from utils.metrics import calculate_map

def apply_nms(boxes, distance_threshold=1.5):
    """
    Applies Non-Maximum Suppression based on confidence and BEV center distance.
    boxes shape: [N, 8] -> [Conf, X, Y, Z, L, W, H, Heading]
    """
    if len(boxes) == 0:
        return boxes
        
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
        
        dists = torch.sqrt((rest[:, 1] - current[1])**2 + (rest[:, 2] - current[2])**2)
        overlap_mask = dists > distance_threshold
        boxes = rest[overlap_mask]
        
    return torch.stack(keep) if keep else torch.zeros((0, 8), device=boxes.device)


def _extract_peaks(heatmap, kernel=3):
    """
    Applies a 3x3 max pooling operation to a CenterNet heatmap to extract local maxima.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(preds, conf_thresh=0.1, max_depth=80.0, num_depth_bins=40):
    """
    Extracts object peaks from the CenterNet heatmap and decodes bin-based depth
    back into absolute 3D world coordinates.
    Returns a list of tensors of shape [N, 8] -> [Conf, X, Y, Z, L, W, H, Heading]
    """
    IMAGE_WIDTH = 960.0
    IMAGE_HEIGHT = 640.0
    FOCAL_LENGTH = 1000.0
    CAMERA_HEIGHT_OFFSET = 1.5
    STRIDE = 32.0 
    BIN_SIZE = max_depth / num_depth_bins

    # Extract all prediction heads
    cls_preds = preds['class']
    loc_preds = preds['location']
    depth_bin_preds = preds['depth_bin']
    depth_res_preds = preds['depth_res']
    dim_preds = preds['dimensions']
    ori_preds = preds['orientation']
    
    batch_size = cls_preds.shape[0]
    
    # Apply Sigmoid and extract CenterNet spatial peaks
    cls_probs = torch.sigmoid(cls_preds)  
    cls_probs = _extract_peaks(cls_probs)
    
    # Get highest probability depth bin for each cell
    depth_bin_indices = torch.argmax(depth_bin_preds, dim=1) # Shape: [B, H, W]
    
    batch_boxes = []
    
    for b in range(batch_size):
        vehicle_probs = cls_probs[b][0] # Index 0 for vehicle class
        mask = vehicle_probs > conf_thresh
        conf = vehicle_probs[mask]
        
        if len(conf) == 0:
            batch_boxes.append(torch.zeros((0, 8), device=conf.device))
            continue
            
        y_indices, x_indices = torch.where(mask)
        
        # 1. Extract 2D Offsets
        offset_x = loc_preds[b, 0, y_indices, x_indices]
        offset_y = loc_preds[b, 1, y_indices, x_indices]
        
        # 2. Decode Bin-Based Depth
        bin_idx = depth_bin_indices[b, y_indices, x_indices].float()
        residual = depth_res_preds[b, 0, y_indices, x_indices]
        cx = (bin_idx + residual) * BIN_SIZE
        
        # 3. Reconstruct 2D Pixels
        pixel_x = (x_indices.float() + offset_x) * STRIDE
        pixel_y = (y_indices.float() + offset_y) * STRIDE
        
        # 4. Project back to absolute 3D World Coordinates
        cy = ((IMAGE_WIDTH / 2.0) - pixel_x) * cx / FOCAL_LENGTH
        cz = ((IMAGE_HEIGHT / 2.0) - pixel_y) * cx / FOCAL_LENGTH + CAMERA_HEIGHT_OFFSET
        
        abs_loc = torch.stack([cx, cy, cz], dim=0)
        
        # 5. Extract Dimensions and Orientation
        dim = dim_preds[b, :, y_indices, x_indices]
        ori = ori_preds[b, :, y_indices, x_indices]
        
        sin_val = ori[0, :]
        cos_val = ori[1, :]
        heading = torch.atan2(sin_val, cos_val).unsqueeze(0) 
        
        # Concatenate into [N, 8]
        boxes = torch.cat([conf.unsqueeze(1), abs_loc.t(), dim.t(), heading.t()], dim=1)
        
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