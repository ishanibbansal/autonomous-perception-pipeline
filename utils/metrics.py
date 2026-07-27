import torch

def compute_3d_iou(boxes1, boxes2):
    """
    Calculates Axis-Aligned 3D Intersection over Union (IoU).
    Expects tensors of shape [N, 6] containing [X, Y, Z, Length, Width, Height].
    Assumes X, Y, Z represent the center of the bounding box.
    """
    # 1. Extract dimensions (Length, Width, Height)
    l1, w1, h1 = boxes1[:, 3], boxes1[:, 4], boxes1[:, 5]
    l2, w2, h2 = boxes2[:, 3], boxes2[:, 4], boxes2[:, 5]
    
    # 2. Calculate independent volumes
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    
    # 3. Convert center coordinates (X, Y, Z) to min/max coordinates
    # Boxes 1
    b1_xmin = boxes1[:, 0] - (l1 / 2)
    b1_xmax = boxes1[:, 0] + (l1 / 2)
    b1_ymin = boxes1[:, 1] - (w1 / 2)
    b1_ymax = boxes1[:, 1] + (w1 / 2)
    b1_zmin = boxes1[:, 2] - (h1 / 2)
    b1_zmax = boxes1[:, 2] + (h1 / 2)
    
    # Boxes 2
    b2_xmin = boxes2[:, 0] - (l2 / 2)
    b2_xmax = boxes2[:, 0] + (l2 / 2)
    b2_ymin = boxes2[:, 1] - (w2 / 2)
    b2_ymax = boxes2[:, 1] + (w2 / 2)
    b2_zmin = boxes2[:, 2] - (h2 / 2)
    b2_zmax = boxes2[:, 2] + (h2 / 2)
    
    # 4. Calculate overlap boundaries
    overlap_xmin = torch.max(b1_xmin, b2_xmin)
    overlap_xmax = torch.min(b1_xmax, b2_xmax)
    
    overlap_ymin = torch.max(b1_ymin, b2_ymin)
    overlap_ymax = torch.min(b1_ymax, b2_ymax)
    
    overlap_zmin = torch.max(b1_zmin, b2_zmin)
    overlap_zmax = torch.min(b1_zmax, b2_zmax)
    
    # 5. Calculate intersection dimensions (clamp to 0 to prevent negative overlap)
    overlap_l = torch.clamp(overlap_xmax - overlap_xmin, min=0)
    overlap_w = torch.clamp(overlap_ymax - overlap_ymin, min=0)
    overlap_h = torch.clamp(overlap_zmax - overlap_zmin, min=0)
    
    # 6. Calculate Intersection and Union Volume
    intersection = overlap_l * overlap_w * overlap_h
    union = vol1 + vol2 - intersection
    
    # 7. Compute IoU (add epsilon to prevent division by zero)
    iou = intersection / (union + 1e-6)
    
    return iou

def compute_ap(recall, precision):
    """
    Computes the Average Precision (AP) given recall and precision arrays.
    Uses the 11-point interpolation method standard in early VOC/KITTI evaluations.
    """
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1):
        if torch.sum(recall >= t) == 0:
            p = 0
        else:
            p = torch.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculates Mean Average Precision (mAP) for a batch.
    
    predictions: Tensor of shape [N, 7] -> [Confidence, X, Y, Z, Length, Width, Height]
    targets: Tensor of shape [M, 6] -> [X, Y, Z, Length, Width, Height]
    """
    if len(predictions) == 0:
        return 0.0
        
    # Sort predictions by confidence (highest first)
    sorted_indices = torch.argsort(predictions[:, 0], descending=True)
    predictions = predictions[sorted_indices]
    
    # Extract bounding box coordinates
    pred_boxes = predictions[:, 1:]
    
    num_preds = len(pred_boxes)
    num_targets = len(targets)
    
    true_positives = torch.zeros(num_preds)
    false_positives = torch.zeros(num_preds)
    
    # Keep track of which ground truth boxes have already been matched
    target_matched = torch.zeros(num_targets)
    
    for i, pred_box in enumerate(pred_boxes):
        if num_targets == 0:
            false_positives[i] = 1
            continue
            
        # Compare this single prediction against ALL ground truth boxes
        # (unsqueeze pred_box to match shapes for the IoU function)
        ious = compute_3d_iou(pred_box.unsqueeze(0).expand(num_targets, -1), targets)
        
        best_iou, best_target_idx = torch.max(ious, dim=0)
        
        if best_iou >= iou_threshold and target_matched[best_target_idx] == 0:
            true_positives[i] = 1
            target_matched[best_target_idx] = 1 # Mark as matched so it isn't double-counted
        else:
            false_positives[i] = 1
            
    # Calculate cumulative precision and recall
    cum_true_positives = torch.cumsum(true_positives, dim=0)
    cum_false_positives = torch.cumsum(false_positives, dim=0)
    
    precision = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-6)
    recall = cum_true_positives / (num_targets + 1e-6)
    
    # Calculate AP
    ap = compute_ap(recall, precision)
    
    return ap.item()

# --- Testing the 3D IoU and mAP Math ---
if __name__ == '__main__':
    print("Validating 3D IoU Calculations...\n")
    
    # Format: [X, Y, Z, Length, Width, Height]
    base_box = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
    identical_box = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
    disjoint_box = torch.tensor([[10.0, 10.0, 10.0, 2.0, 2.0, 2.0]])
    partial_box = torch.tensor([[1.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
    
    predictions = torch.cat([base_box, base_box, base_box], dim=0)
    targets = torch.cat([identical_box, disjoint_box, partial_box], dim=0)
    
    ious = compute_3d_iou(predictions, targets)
    
    print("--- 3D IoU Results ---")
    print(f"Perfect Overlap : {ious[0].item():.4f} (Expected: 1.0000)")
    print(f"No Overlap      : {ious[1].item():.4f} (Expected: 0.0000)")
    print(f"Partial Overlap : {ious[2].item():.4f} (Expected: 0.3333)")
    
    print("\n-----------------------------------\n")
    print("Validating mAP Calculations...\n")
    
    # Targets: 2 ground-truth boxes [X, Y, Z, L, W, H]
    target_boxes = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        [5.0, 5.0, 0.0, 2.0, 2.0, 2.0]
    ])
    
    # Predictions: [Confidence, X, Y, Z, L, W, H]
    pred_boxes = torch.tensor([
        [0.95, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],  # TP: Perfect match
        [0.80, 5.2, 5.0, 0.0, 2.0, 2.0, 2.0],  # TP: Slight X offset, IoU > 0.5
        [0.40, 10.0, 10.0, 0.0, 2.0, 2.0, 2.0] # FP: Completely wrong location
    ])
    
    map_score = calculate_map(pred_boxes, target_boxes, iou_threshold=0.5)
    
    print("--- mAP Results ---")
    print(f"Calculated mAP (IoU > 0.5): {map_score:.4f}")