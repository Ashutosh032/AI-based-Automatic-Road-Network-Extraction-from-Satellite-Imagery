import kagglehub
import shutil
import os

def download_and_setup_data():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("balraj98/deepglobe-road-extraction-dataset")
    print("Path to dataset files:", path)
    
    # Target directory
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Moving files to {target_dir}...")
    
    # The dataset download might be a folder containing the files
    # We want to copy/move the contents to data/raw
    
    # Check if 'train' exists inside the downloaded path (DeepGlobe structure usually has train/test/valid)
    # Based on the dataset description, it might just be images or have a specific structure.
    # Let's verify what's inside first by listing it, but for automation we will copy everything.
    
    # robust move
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
            
    print("Data setup complete.")

if __name__ == "__main__":
    download_and_setup_data()
