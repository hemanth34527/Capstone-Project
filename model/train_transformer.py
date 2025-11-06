"""
Training script for Pure Transformer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse

from pure_transformer_model import get_pure_transformer


class RemoteSensingDataset(Dataset):
    """Dataset for remote sensing images and masks"""
    def __init__(self, image_dir, mask_dir, transform=None, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.png', '.jpg', '.tif'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.tif', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, mask


def get_transforms(train=True):
    """Get image transforms"""
    if train:
        return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate Dice
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def calculate_metrics(pred, target, num_classes=7):
    """Calculate accuracy and IoU"""
    pred = torch.argmax(pred, dim=1)
    
    # Accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total
    
    # IoU per class
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union > 0:
            iou_per_class.append(intersection / union)
        else:
            iou_per_class.append(0.0)
    
    mean_iou = np.mean(iou_per_class)
    
    return accuracy, mean_iou


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc, iou = calculate_metrics(outputs, masks)
        
        running_loss += loss.item()
        running_acc += acc
        running_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.2f}',
            'acc': f'{acc:.3f}',
            'mIoU': f'{iou:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_iou


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
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
                'loss': f'{loss.item():.2f}',
                'acc': f'{acc:.3f}',
                'mIoU': f'{iou:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_acc, epoch_iou


def train_model(
    model_size='base',
    data_dir='data',
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    image_size=256,
    num_classes=7,
    checkpoint_dir='checkpoints_transformer',
    device='cuda'
):
    """Main training function"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = get_pure_transformer(
        num_classes=num_classes,
        image_size=image_size,
        model_size=model_size
    )
    model = model.to(device)
    print(f"Model created: {model.get_num_params() / 1e6:.2f}M parameters")
    
    # Create datasets
    train_dataset = RemoteSensingDataset(
        os.path.join(data_dir, 'train/images'),
        os.path.join(data_dir, 'train/masks'),
        transform=get_transforms(train=True),
        image_size=image_size
    )
    
    val_dataset = RemoteSensingDataset(
        os.path.join(data_dir, 'valid/images'),
        os.path.join(data_dir, 'valid/masks'),
        transform=get_transforms(train=False),
        image_size=image_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_iou = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_iou = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"✓ Saved best model with mIoU: {best_iou:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_acc': val_acc,
            }, checkpoint_path)
    
    print(f"\nTraining completed! Best validation mIoU: {best_iou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pure Transformer model')
    parser.add_argument('--model_size', type=str, default='base', 
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Train with command-line arguments (config file not needed)
    train_model(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        device=args.device
    )
