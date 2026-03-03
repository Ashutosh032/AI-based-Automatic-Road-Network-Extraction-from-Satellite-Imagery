import torch
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import numpy as np

def predict(model, image_path, device):
    model.eval()
    
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=original_image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        
    mask = (output > 0.5).astype(np.uint8)
    
    # Resize mask back to original size if needed for overlay
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return original_image, mask_resized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='../models/best_model.pth')
    parser.add_argument('--output', type=str, default='output.png')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Warning: Weights file {args.weights} not found. Using random weights.")
    
    image, mask = predict(model, args.input_image, device)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Road Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.savefig(args.output)
    print(f"Result saved to {args.output}")

if __name__ == '__main__':
    main()
