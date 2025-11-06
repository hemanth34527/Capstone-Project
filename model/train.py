"""
Training Script for Weighted Feature Fusion Network
Includes training, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import json

from weighted_fusion_model import get_model


class RemoteSensingDataset(Dataset):
    """
    Dataset for Remote Sensing Image Segmentation
    Supports multi-modal inputs (RGB, Infrared, etc.)
    """
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=7):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png').replace('.tif', '.png'))
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
            mask = (mask * 255).long().squeeze(0)
        
        return image, mask


def get_transforms(img_size=256, augment=True):
    """
    Data augmentation and preprocessing
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Cross Entropy + Dice Loss
    """
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.weight_ce * ce + self.weight_dice * dice


def calculate_metrics(pred, target, num_classes=7):
    """
    Calculate IoU, Pixel Accuracy, and F1 Score
    """
    pred = torch.argmax(pred, dim=1)
    
    # Pixel Accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total
    
    # IoU per class
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)
    
    mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return accuracy, mean_iou


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        acc, iou = calculate_metrics(outputs, masks)
        
        running_loss += loss.item()
        running_acc += acc
        running_iou += iou
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': acc,
            'mIoU': iou
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_iou


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            acc, iou = calculate_metrics(outputs, masks)
            
            running_loss += loss.item()
            running_acc += acc
            running_iou += iou
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': acc,
                'mIoU': iou
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_iou


def train_model(train_dir, val_dir, mask_train_dir, mask_val_dir,
                num_epochs=50, batch_size=8, lr=1e-4, num_classes=7,
                checkpoint_dir='checkpoints', device='cuda'):
    """
    Main training function
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_transform = get_transforms(img_size=256, augment=True)
    val_transform = get_transforms(img_size=256, augment=False)
    
    train_dataset = RemoteSensingDataset(train_dir, mask_train_dir, transform=train_transform)
    val_dataset = RemoteSensingDataset(val_dir, mask_val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = get_model(num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_iou': [],
        'val_loss': [], 'val_acc': [], 'val_iou': []
    }
    
    best_val_iou = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_iou = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"✓ Saved best model with mIoU: {val_iou:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save training history
    with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\nTraining completed! Best validation mIoU: {best_val_iou:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()
    
    # Example usage - updated to match actual folder structure
    train_dir = "data/train/images"
    val_dir = "data/valid/images"  # Fixed from val to valid
    mask_train_dir = "data/train/masks"
    mask_val_dir = "data/valid/masks"  # Fixed from val to valid
    
    model, history = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        mask_train_dir=mask_train_dir,
        mask_val_dir=mask_val_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=7,
        checkpoint_dir='checkpoints',
        device='cuda'
    )
