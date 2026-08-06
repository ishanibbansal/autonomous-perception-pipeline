import os
import glob
import subprocess
import argparse

# Dynamically find the base directory (one level up from this script)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def sync_waymo_dataset(bucket_uri, local_dir, target_count):
    os.makedirs(local_dir, exist_ok=True)
    
    # Count local files
    local_paths = glob.glob(os.path.join(local_dir, '*.tfrecord'))
    local_files = {os.path.basename(f) for f in local_paths}
    current_count = len(local_files)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"[INFO] {local_dir} already has {current_count} files (Target: {target_count}). No download needed.")
        return

    print(f"[INFO] Found {current_count} files in {local_dir}. Need {needed} more.")
    
    # Get list of files from GCS
    print(f"[INFO] Fetching file list from {bucket_uri}...")
    try:
        result = subprocess.run(['gsutil', 'ls', bucket_uri], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to list bucket. Ensure you are authenticated with 'gcloud auth login'.")
        return
        
    remote_urls = [url for url in result.stdout.strip().split('\n') if url.endswith('.tfrecord')]
    
    # Filter for files you don't already have
    to_download = []
    for url in remote_urls:
        filename = url.split('/')[-1]
        if filename not in local_files:
            to_download.append(url)
        if len(to_download) == needed:
            break
            
    if not to_download:
        print("[INFO] No new files found in the bucket.")
        return
        
    # Download files in parallel
    print(f"[INFO] Downloading {len(to_download)} new files to {local_dir}...")
    download_cmd = ['gsutil', '-m', 'cp'] + to_download + [local_dir]
    
    try:
        subprocess.run(download_cmd, check=True)
        print(f"[SUCCESS] Download complete for {local_dir}.")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Download failed. Check your network connection or gsutil permissions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sync Waymo Dataset TFRecords")
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both', 
                        help="Which dataset split to download (train, val, or both)")
    args = parser.parse_args()

    # Configuration 
    TARGET_TRAIN_FILES = 25
    TARGET_VAL_FILES = 5
    
    # Updated to use the correct individual_files structure
    TRAIN_BUCKET = "gs://waymo_open_dataset_v_1_4_3/individual_files/training/"
    VAL_BUCKET = "gs://waymo_open_dataset_v_1_4_3/individual_files/validation/"
    
    # Resolve absolute paths from the base directory
    TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
    VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "val")
    
    print("--- Waymo Dataset Sync ---")
    
    if args.split in ['train', 'both']:
        sync_waymo_dataset(TRAIN_BUCKET, TRAIN_DIR, TARGET_TRAIN_FILES)
        
    if args.split in ['val', 'both']:
        if args.split == 'both':
            print("-" * 25)
        sync_waymo_dataset(VAL_BUCKET, VAL_DIR, TARGET_VAL_FILES)