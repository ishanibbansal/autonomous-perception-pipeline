import torch

def validate_model(model, dataloader, criterion, encoder, device):
    """
    Evaluates the model on a holdout dataset without calculating gradients.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad():  # Disable gradient tracking to save memory and compute
        for batch in dataloader:
            # Extract inputs and raw targets
            images = batch['front_image'].to(device, dtype=torch.float32)
            raw_bboxes = batch['bboxes']
            valid_boxes = batch['num_valid_boxes']
            
            # Encode raw bounding boxes into dense spatial grids
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            
            # Move encoded targets to the GPU
            targets = {k: v.to(device) for k, v in encoded_targets.items()}
            
            # Forward Pass (No backward pass!)
            predictions = model(images)
            
            # Calculate Loss
            loss, _ = criterion(predictions, targets)
            val_loss += loss.item()
            
    # Return the average validation loss for this dataset
    return val_loss / len(dataloader)