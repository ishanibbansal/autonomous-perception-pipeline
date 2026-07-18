import torch
from torch.utils.data import Dataset, DataLoader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Keep TensorFlow quiet!

class WaymoDataset(Dataset):
    def __init__(self, tfrecord_path):
        """
        Initialize the dataset by loading the TFRecord file and 
        counting the total number of frames.
        """
        self.tfrecord_path = tfrecord_path
        
        # We load the dataset using TensorFlow
        self.raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path, compression_type='')
        
        # PyTorch needs to know exactly how many items are in the dataset
        # We have to iterate through once to count the frames
        print("Initializing dataset and counting frames... this may take a second.")
        self.frame_list = [data.numpy() for data in self.raw_dataset]
        self.num_frames = len(self.frame_list)

    def __len__(self):
        """Returns the total number of frames."""
        return self.num_frames

    def __getitem__(self, idx):
        """
        Grabs a single frame by its index, parses the protobuf, 
        and will eventually return PyTorch tensors.
        """
        raw_data = self.frame_list[idx]
        
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw_data))
        
        # For now, let's just extract the timestamp to prove it works
        timestamp = frame.timestamp_micros
        
        # We will return this as a PyTorch tensor
        return torch.tensor(timestamp, dtype=torch.int64)

# --- Testing the Class ---
if __name__ == '__main__':
    data_path = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    
    # Instantiate the object
    waymo_data = WaymoDataset(data_path)
    print(f"Total frames in this segment: {len(waymo_data)}")
    
    # Wrap it in a DataLoader to handle batching
    dataloader = DataLoader(waymo_data, batch_size=4, shuffle=False)
    
    # Grab the first batch
    for batch in dataloader:
        print(f"\nFirst batch of timestamps as PyTorch Tensors:\n{batch}")
        break