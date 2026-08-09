import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

# Point this to one of your downloaded tfrecord files
DATA_PATH = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'

def inspect_first_frame(file_path):
    print(f"Opening {file_path}...")
    
    # Load the TFRecord file
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    
    # Iterate through the dataset (grabbing just the first frame)
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        print("\n--- Frame Extracted ---")
        print(f"Timestamp (micros): {frame.timestamp_micros}")
        print(f"Time of day: {frame.context.stats.time_of_day}")
        print(f"Location: {frame.context.stats.location}")
        
        # Sensor data counts
        print(f"\nNumber of images in frame: {len(frame.images)}")
        print(f"Number of LiDAR scans: {len(frame.lasers)}")
        print(f"Number of 3D bounding box labels: {len(frame.laser_labels)}")
        
        break  # We only want to inspect the first frame for now

if __name__ == '__main__':
    # Ensure the script is run from the root of the repository
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}. Make sure you are running this from the repo root.")
    else:
        inspect_first_frame(DATA_PATH)