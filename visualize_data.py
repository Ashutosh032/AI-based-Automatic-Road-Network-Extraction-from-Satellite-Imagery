import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
import argparse
import numpy as np

def visualize_samples(data_dir, output_path='samples.png', n=5):
    # Find images
    search_path = os.path.join(data_dir, '**', '*_sat.jpg')
    image_paths = sorted(glob(search_path, recursive=True))
    
    if not image_paths:
        print(f"No images found in {data_dir}")
        return

    # Randomly select n samples
    indices = np.random.choice(len(image_paths), min(n, len(image_paths)), replace=False)
    
    plt.figure(figsize=(10, 2*n))
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2])
            
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(image)
        plt.title("Satellite Image")
        plt.axis('off')
        
        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Road Mask")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/raw')
    args = parser.parse_args()
    
    visualize_samples(args.data_dir)
