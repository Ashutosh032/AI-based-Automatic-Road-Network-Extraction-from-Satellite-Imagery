import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

class DeepGlobeRoadDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        \"\"\"
        Args:
            root_dir (str): Path to the dataset root directory (containing train directory).
            split (str): 'train' or 'valid'.
            transform (albumentations.Compose): transform to be applied on a sample.
        \"\"\"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Assuming the standard DeepGlobe structure where images are *_sat.jpg and masks are *_mask.png
        # We will split the 'train' folder into train/val manually or expect a split
        # For this implementation, let's look for all images in the directory
        
        # specific check for 'train' folder
        train_dir = os.path.join(root_dir, 'train')
        valid_dir = os.path.join(root_dir, 'valid')
        
        if os.path.exists(train_dir):
            if split == 'train':
                search_root = train_dir
            elif split == 'valid' and os.path.exists(valid_dir):
                search_root = valid_dir
            elif split == 'valid':
                # Splitting train dir if no valid dir exists
                search_root = train_dir
            else:
                search_root = root_dir
        else:
            search_root = root_dir

        search_path = os.path.join(search_root, '**', '*_sat.jpg')
        all_image_paths = sorted(glob(search_path, recursive=True))
        
        if len(all_image_paths) == 0:
             # Fallback to just .jpg
            search_path = os.path.join(search_root, '**', '*.jpg')
            all_image_paths = sorted(glob(search_path, recursive=True))

        print(f"Found {len(all_image_paths)} images in {search_root}")

        # If we are using distinct folders, don't split by index unless we are splitting 'train' to make a validation set
        if os.path.exists(valid_dir) and split in ['train', 'valid']:
            # We already selected the correct folder above
            self.image_paths = all_image_paths
        else:
            # We need to split the found images (either from root or from train dir if valid doesn't exist)
            split_idx = int(0.8 * len(all_image_paths))
            if self.split == 'train':
                self.image_paths = all_image_paths[:split_idx]
            else:
                self.image_paths = all_image_paths[split_idx:]
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            # Fallback if mask missing (e.g. test set), strictly for training this shouldn't happen usually
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Binarize mask (road = 255, background = 0) -> (1, 0)
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default transform if none provided
            t = A.Compose([A.Resize(512, 512), ToTensorV2()])
            augmented = t(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.unsqueeze(0) # (C, H, W)

def get_transforms(split='train'):
    if split == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
