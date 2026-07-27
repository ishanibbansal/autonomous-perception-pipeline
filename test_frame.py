import cv2
import glob
from utils.dataset import WaymoDataset

def extract_first_frame():
    tf_files = glob.glob('data/raw/train/*.tfrecord') + glob.glob('data/raw/*.tfrecord')
    
    if not tf_files:
        print("Error: Could not find any .tfrecord files.")
        return
        
    print(f"Loading dataset from: {tf_files[0]}")
    dataset = WaymoDataset(tf_files[0])
    
    sample = dataset[0]
    img_tensor = sample['front_image'] 
    
    # The tensor is already 0-255, so we just convert it to uint8 directly
    img_np = img_tensor.permute(1, 2, 0).numpy().astype('uint8')
    
    # Convert RGB to BGR for OpenCV saving
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('test_frame.jpg', img_bgr)
    print("Successfully extracted test_frame.jpg!")

if __name__ == '__main__':
    extract_first_frame()