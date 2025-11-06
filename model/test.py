"""
Testing Script for the Weighted Feature Fusion Network
Evaluate model performance on test dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from weighted_fusion_model import get_model
from train import RemoteSensingDataset, calculate_metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    def __init__(self, model_path, num_classes=7, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = get_model(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ['Water', 'Vegetation', 'Urban', 'Agriculture', 
                           'Forest', 'Bare Land', 'Road']
        
    def evaluate_dataset(self, test_loader):
        """
        Evaluate model on entire test dataset
        """
        all_preds = []
        all_targets = []
        iou_per_class = [[] for _ in range(self.num_classes)]
        
        total_acc = 0.0
        total_iou = 0.0
        
        print("Evaluating model...")
        with torch.no_grad():
            for images, masks in tqdm(test_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate metrics
                acc, mean_iou = calculate_metrics(outputs, masks, self.num_classes)
                total_acc += acc
                total_iou += mean_iou
                
                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
                
                # Calculate IoU per class
                for cls in range(self.num_classes):
                    pred_mask = (preds == cls)
                    target_mask = (masks == cls)
                    
                    intersection = (pred_mask & target_mask).sum().item()
                    union = (pred_mask | target_mask).sum().item()
                    
                    if union > 0:
                        iou = intersection / union
                        iou_per_class[cls].append(iou)
        
        # Calculate overall metrics
        avg_acc = total_acc / len(test_loader)
        avg_iou = total_iou / len(test_loader)
        
        # Calculate per-class IoU
        class_iou = {}
        for cls in range(self.num_classes):
            if iou_per_class[cls]:
                class_iou[self.class_names[cls]] = np.mean(iou_per_class[cls])
            else:
                class_iou[self.class_names[cls]] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds, labels=range(self.num_classes))
        
        results = {
            'overall_accuracy': float(avg_acc),
            'mean_iou': float(avg_iou),
            'class_iou': class_iou,
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_class_iou(self, class_iou, save_path='class_iou.png'):
        """
        Plot per-class IoU
        """
        classes = list(class_iou.keys())
        iou_values = list(class_iou.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, iou_values, color='steelblue')
        plt.xlabel('Class')
        plt.ylabel('IoU Score')
        plt.title('Per-Class IoU Performance')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class IoU plot saved to {save_path}")
    
    def generate_report(self, results, save_path='evaluation_report.json'):
        """
        Generate detailed evaluation report
        """
        report = {
            'model': 'Weighted Feature Fusion Network',
            'overall_metrics': {
                'accuracy': f"{results['overall_accuracy']:.4f}",
                'mean_iou': f"{results['mean_iou']:.4f}"
            },
            'per_class_iou': {k: f"{v:.4f}" for k, v in results['class_iou'].items()}
        }
        
        # Calculate additional metrics
        cm = np.array(results['confusion_matrix'])
        
        # Precision, Recall, F1 per class
        precision = {}
        recall = {}
        f1_score = {}
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            precision[class_name] = f"{prec:.4f}"
            recall[class_name] = f"{rec:.4f}"
            f1_score[class_name] = f"{f1:.4f}"
        
        report['precision'] = precision
        report['recall'] = recall
        report['f1_score'] = f1_score
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\nEvaluation Report saved to {save_path}")
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        print("\nPer-Class IoU:")
        for class_name, iou in results['class_iou'].items():
            print(f"  {class_name:15s}: {iou:.4f}")
        print("="*60)
        
        return report


def test_model(model_path, test_dir, mask_test_dir, batch_size=8, 
               num_classes=7, output_dir='evaluation_results', device='cuda'):
    """
    Main testing function
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = RemoteSensingDataset(test_dir, mask_test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, num_classes=num_classes, device=device)
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_loader)
    
    # Generate visualizations
    cm = np.array(results['confusion_matrix'])
    evaluator.plot_confusion_matrix(cm, save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    evaluator.plot_class_iou(results['class_iou'], save_path=os.path.join(output_dir, 'class_iou.png'))
    
    # Generate report
    report = evaluator.generate_report(results, save_path=os.path.join(output_dir, 'evaluation_report.json'))
    
    print("\nEvaluation completed!")
    
    return results, report


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best_model.pth"
    test_dir = "data/test/images"
    mask_test_dir = "data/test/masks"
    
    results, report = test_model(
        model_path=model_path,
        test_dir=test_dir,
        mask_test_dir=mask_test_dir,
        batch_size=8,
        num_classes=7,
        output_dir='evaluation_results',
        device='cuda'
    )
