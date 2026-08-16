import torch
import os
import sys

# Inject root path so imports from 'src' work from inside the 'scripts' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perception.models.student_bev import StudentBEVDetector

# 1. Wrapper to strip out everything except the 1-channel BEV heatmap
class BEVExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, intrinsics, extrinsics):
        out = self.model(img, intrinsics, extrinsics)
        # Ensure we only return the actual [1, 1, 160, 160] heatmap
        return out['bev_occupancy'] if isinstance(out, dict) else out

def export_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing ONNX Export...")

    checkpoint_path = 'best_student_bev_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    base_model = StudentBEVDetector().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Wrap the model
    model = BEVExportWrapper(base_model).to(device)
    model.eval()

    # 2. Use REAL Waymo matrices for dummy inputs so static shape tracing is geometrically accurate
    dummy_image = torch.randn(1, 3, 1280, 1920, device=device)
    
    dummy_intrinsics = torch.tensor([[
        [2083.091212, 0.0, 957.293829],
        [0.0, 2083.091212, 650.569793],
        [0.0, 0.0, 1.0]
    ]], dtype=torch.float32, device=device)

    dummy_extrinsics = torch.tensor([[
        [ 0.0,  0.0,  1.0,  2.0],
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  1.5],
        [ 0.0,  0.0,  0.0,  1.0]
    ]], dtype=torch.float32, device=device)

    # CRITICAL FIX: Define the dummy_inputs tuple!
    dummy_inputs = (dummy_image, dummy_intrinsics, dummy_extrinsics)

    onnx_file_path = "student_bev.onnx"
    print("Tracing and exporting model to ONNX. This may take a moment...")
    
    torch.onnx.export(
        model,                                
        dummy_inputs,                         
        onnx_file_path,                       
        export_params=True,                   
        opset_version=17,                     
        do_constant_folding=True,             
        input_names=['camera_image', 'intrinsics', 'extrinsics_inv'],   
        output_names=['bev_occupancy']
    )

    print(f"Success! ONNX model exported to: {onnx_file_path}")

if __name__ == '__main__':
    export_to_onnx()