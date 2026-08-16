import torch
import os
import sys

# Inject root path so imports from 'src' work from inside the 'scripts' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perception.models.student_bev import StudentBEVDetector

def export_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing ONNX Export...")

    checkpoint_path = 'best_student_bev_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = StudentBEVDetector().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    dummy_image = torch.randn(1, 3, 1280, 1920, device=device)
    dummy_intrinsics = torch.eye(3, device=device).unsqueeze(0)
    dummy_extrinsics_inv = torch.eye(4, device=device).unsqueeze(0)
    
    dummy_inputs = (dummy_image, dummy_intrinsics, dummy_extrinsics_inv)

    onnx_file_path = "student_bev.onnx"

    print("Tracing and exporting model to ONNX. This may take a moment...")
    torch.onnx.export(
        model,                                
        dummy_inputs,                         
        onnx_file_path,                       
        export_params=True,                   
        opset_version=14,                     
        do_constant_folding=True,             
        input_names=['camera_image', 'intrinsics', 'extrinsics_inv'],   
        output_names=['bev_occupancy'],       
        dynamic_axes={                        
            'camera_image': {0: 'batch_size'},
            'intrinsics': {0: 'batch_size'},
            'extrinsics_inv': {0: 'batch_size'},
            'bev_occupancy': {0: 'batch_size'}
        }
    )

    print(f"Success! ONNX model exported to: {onnx_file_path}")

if __name__ == '__main__':
    export_to_onnx()