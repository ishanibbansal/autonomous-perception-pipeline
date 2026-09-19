import tensorflow as tf
from waymo_open_dataset import dataset_pb2
import numpy as np
import sys

def extract_calibration(tfrecord_path):
    print(f"Reading {tfrecord_path}...")
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    for data in dataset:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        for calibration in frame.context.camera_calibrations:
            if calibration.name == dataset_pb2.CameraName.FRONT:
                # 1. Extract Intrinsics
                # Waymo provides 1D array: [f_u, f_v, c_u, c_v, k_1, k_2, p_1, p_2, k_3]
                f_u, f_v, c_u, c_v = calibration.intrinsic[0:4]
                
                print("\n// --- COPY THIS INTO YOUR C++ NODE ---")
                print("// 1. Front Camera Intrinsics")
                print("float intrinsics[9] = {")
                print(f"    {f_u:.6f}f, 0.0f, {c_u:.6f}f,")
                print(f"    0.0f, {f_v:.6f}f, {c_v:.6f}f,")
                print("    0.0f, 0.0f, 1.0f")
                print("};")
                
                # 2. Extract Extrinsics (Sensor to Vehicle)
                extrinsics = np.array(calibration.extrinsic.transform).reshape(4, 4)
                
                # RAW extrinsics (Sensor to Vehicle) - this is what the model was trained on
                print("\n// 2. Front Camera Extrinsics RAW (Sensor to Vehicle)")
                print("// NOTE: The StudentBEVDetector was trained with raw extrinsics passed")
                print("//       into the 'extrinsics_inv' parameter. Use these values.")
                print("float extrinsics_inv[16] = {")
                for row in extrinsics:
                    print(f"    {row[0]:.6f}f, {row[1]:.6f}f, {row[2]:.6f}f, {row[3]:.6f}f,")
                print("};")
                
                # Also print the actual inverse for reference
                extrinsics_inv = np.linalg.inv(extrinsics)
                print("\n// 3. Front Camera Extrinsics Inverse (Vehicle to Sensor) [REFERENCE ONLY]")
                print("// float extrinsics_inv_actual[16] = {")
                for row in extrinsics_inv:
                    print(f"//     {row[0]:.6f}f, {row[1]:.6f}f, {row[2]:.6f}f, {row[3]:.6f}f,")
                print("// };")
                print("// ------------------------------------\n")
                return
                
        print("Error: Could not find FRONT camera calibration in this frame.")
        return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/extract_calib.py <path_to_tfrecord>")
    else:
        extract_calibration(sys.argv[1])