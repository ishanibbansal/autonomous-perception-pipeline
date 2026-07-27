import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import math
import os

from model import Waymo3DDetector 
from utils.validate import decode_predictions

def draw_3d_wireframe(img, x, y, z, l, w, h, heading, color=(0, 255, 0), thickness=2):
    IMAGE_WIDTH = 960.0
    IMAGE_HEIGHT = 640.0
    FOCAL_LENGTH = 1000.0
    CAMERA_HEIGHT_OFFSET = 1.5
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    
    corners_3d = []
    for i in range(8):
        local_x = x_corners[i]
        local_y = y_corners[i]
        local_z = z_corners[i]
        
        rotated_x = (local_x * cos_h) - (local_y * sin_h)
        rotated_y = (local_x * sin_h) + (local_y * cos_h)
        
        c_x = x + rotated_x
        c_y = y + rotated_y
        c_z = z + local_z
        
        corners_3d.append((c_x, c_y, c_z))
        
    corners_2d = []
    for c_x, c_y, c_z in corners_3d:
        if c_x <= 0:
            return img 
            
        pixel_x = int((IMAGE_WIDTH / 2) - (FOCAL_LENGTH * c_y / c_x))
        pixel_y = int((IMAGE_HEIGHT / 2) - (FOCAL_LENGTH * (c_z - CAMERA_HEIGHT_OFFSET) / c_x))
        corners_2d.append((pixel_x, pixel_y))
        
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), 
        (4, 5), (5, 6), (6, 7), (7, 4), 
        (0, 4), (1, 5), (2, 6), (3, 7)  
    ]
    
    for start_idx, end_idx in edges:
        pt1 = corners_2d[start_idx]
        pt2 = corners_2d[end_idx]
        
        if -500 < pt1[0] < IMAGE_WIDTH + 500 and -500 < pt1[1] < IMAGE_HEIGHT + 500:
            cv2.line(img, pt1, pt2, color, thickness)
            
    return img

def predict_single_frame(image_path, checkpoint_path, output_path='prediction_output.jpg'):
    print(f"Loading model from {checkpoint_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Waymo3DDetector() 
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    print(f"Processing image {image_path}...")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() 
    img_tensor = TF.resize(img_tensor, [640, 960], antialias=True)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        predictions = model(img_tensor)
        
    class_probs = torch.sigmoid(predictions['class'][0])
    print(f"DEBUG -> The absolute peak CenterNet score is: {class_probs.max().item():.3f}")
    
    decoded_boxes = decode_predictions(predictions, conf_thresh=0.08)[0] 
    
    print(f"\n--- Found {len(decoded_boxes)} vehicles ---")
    
    draw_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    
    for i, box in enumerate(decoded_boxes):
        conf = box[0]
        x, y, z = box[1], box[2], box[3]
        length, width, height = box[4], box[5], box[6]
        heading = box[7] 
        
        print(f"Vehicle {i+1} [Conf: {conf:.2f}]:")
        print(f"  Loc: X: {x:.1f}m, Y: {y:.1f}m, Z: {z:.1f}m | Heading: {heading:.2f} rad")
        
        intensity = int(255 * conf)
        color = (0, intensity, 255 - intensity) 
        
        draw_img = draw_3d_wireframe(draw_img, x, y, z, length, width, height, heading, color=color)
        
    cv2.imwrite(output_path, draw_img)
    print(f"\nSaved 3D wireframe visualization to {output_path}")

if __name__ == '__main__':
    test_image_path = 'test_frame.jpg' 
    checkpoint_file = 'waymo_3d_checkpoint.pt'
    
    predict_single_frame(test_image_path, checkpoint_file)