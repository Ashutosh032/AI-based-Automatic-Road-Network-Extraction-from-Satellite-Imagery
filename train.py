import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DeepGlobeRoadDataset, get_transforms
from model import UNet
from utils import dice_loss, iou_score, AverageMeter
import argparse
import os
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = AverageMeter()
    ious = AverageMeter()
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Loss = BCE + Dice
        bce = criterion(outputs, masks)
        dice = dice_loss(outputs, masks)
        loss = bce + dice
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        ious.update(iou_score(outputs, masks).item(), images.size(0))
        
        pbar.set_postfix({'loss': losses.avg, 'iou': ious.avg})
        
    return losses.avg, ious.avg

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    ious = AverageMeter()
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            bce = criterion(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce + dice
            
            losses.update(loss.item(), images.size(0))
            ious.update(iou_score(outputs, masks).item(), images.size(0))
            
    return losses.avg, ious.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/raw', help='Path to data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weights', type=str, default='../models/best_model.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_dataset = DeepGlobeRoadDataset(args.data_dir, split='train', transform=get_transforms('train'))
    val_dataset = DeepGlobeRoadDataset(args.data_dir, split='valid', transform=get_transforms('valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.weights)
            print(f"Saved best model with IoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
