"""
Testing script for Pure Transformer model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pure_transformer_model import get_pure_transformer


class RemoteSensingDataset(Dataset):
    """Dataset for remote sensing images and masks"""
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.png', '.jpg', '.tif'))])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = self.transform(image)
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.tif', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask, img_name


class ModelEvaluator:
    """Evaluator for Pure Transformer model"""
    def __init__(self, model_path, num_classes=7, model_size='base', image_size=256, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = get_pure_transformer(
            num_classes=num_classes,
            image_size=image_size,
            model_size=model_size
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ['Water', 'Vegetation', 'Urban', 'Agriculture', 
                           'Forest', 'Bare Land', 'Road']
    
    def evaluate(self, dataloader):
        """Evaluate model on dataset"""
        all_preds = []
        all_targets = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for images, masks, _ in tqdm(dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        
        return all_preds, all_targets
    
    def calculate_metrics(self, preds, targets):
        """Calculate evaluation metrics"""
        # Overall accuracy
        accuracy = (preds == targets).mean()
        
        # IoU per class
        iou_per_class = []
        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)
            
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            iou_per_class.append(iou)
        
        mean_iou = np.mean(iou_per_class)
        
        return accuracy, mean_iou, iou_per_class
    
    def plot_confusion_matrix(self, preds, targets, save_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Pure Transformer')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_class_iou(self, iou_per_class, save_path):
        """Plot per-class IoU"""
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, iou_per_class, color='steelblue')
        
        # Color bars
        for bar, iou in zip(bars, iou_per_class):
            if iou > 0.7:
                bar.set_color('green')
            elif iou > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xlabel('Class')
        plt.ylabel('IoU Score')
        plt.title('Per-Class IoU - Pure Transformer')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, iou) in enumerate(zip(self.class_names, iou_per_class)):
            plt.text(i, iou + 0.02, f'{iou:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class IoU plot saved to {save_path}")
    
    def generate_report(self, accuracy, mean_iou, iou_per_class, save_path):
        """Generate JSON evaluation report"""
        report = {
            'model': 'Pure Transformer (SETR-style)',
            'overall_accuracy': float(accuracy),
            'mean_iou': float(mean_iou),
            'per_class_iou': {
                name: float(iou) 
                for name, iou in zip(self.class_names, iou_per_class)
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\nEvaluation Report saved to {save_path}")
        return report


def test_model(
    model_path='checkpoints_transformer/best_model.pth',
    test_dir='data/test',
    model_size='base',
    image_size=256,
    num_classes=7,
    output_dir='evaluation_results_transformer',
    device='cuda'
):
    """Test the Pure Transformer model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    test_dataset = RemoteSensingDataset(
        os.path.join(test_dir, 'images'),
        os.path.join(test_dir, 'masks'),
        image_size=image_size
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        num_classes=num_classes,
        model_size=model_size,
        image_size=image_size,
        device=device
    )
    
    # Evaluate
    preds, targets = evaluator.evaluate(test_loader)
    
    # Calculate metrics
    accuracy, mean_iou, iou_per_class = evaluator.calculate_metrics(preds, targets)
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(
        preds, targets, 
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    evaluator.plot_class_iou(
        iou_per_class,
        os.path.join(output_dir, 'class_iou.png')
    )
    
    # Generate report
    report = evaluator.generate_report(
        accuracy, mean_iou, iou_per_class,
        os.path.join(output_dir, 'evaluation_report.json')
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS - Pure Transformer")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("\nPer-Class IoU:")
    for name, iou in zip(evaluator.class_names, iou_per_class):
        print(f"  {name:15s}: {iou:.4f}")
    print("=" * 60)
    
    print("\nEvaluation completed!")
    
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Pure Transformer model')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_transformer/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Test data directory')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--output_dir', type=str, 
                       default='evaluation_results_transformer',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.checkpoint,
        test_dir=args.test_dir,
        model_size=args.model_size,
        image_size=args.image_size,
        output_dir=args.output_dir,
        device=args.device
    )
