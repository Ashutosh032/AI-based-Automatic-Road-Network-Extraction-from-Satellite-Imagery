import torch

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def iou_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    
    pred_bin = pred_bin.view(-1)
    target = target.view(-1)
    
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)

class AverageMeter:
    \"\"\"Computes and stores the average and current value\"\"\"
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
