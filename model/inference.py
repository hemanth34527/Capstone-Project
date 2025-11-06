"""
Inference Script for Weighted Feature Fusion Network
Integrates with the website for real-time segmentation
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

from weighted_fusion_model import get_model


class SegmentationInference:
    """
    Inference class for the segmentation model
    """
    def __init__(self, model_path, num_classes=7, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = get_model(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class colors (matching website)
        self.class_colors = [
            [59, 130, 246],   # Water - Blue
            [16, 185, 129],   # Vegetation - Green
            [239, 68, 68],    # Urban - Red
            [245, 158, 11],   # Agriculture - Orange
            [5, 150, 105],    # Forest - Dark Green
            [139, 92, 246],   # Bare Land - Purple
            [107, 114, 128]   # Road - Gray
        ]
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        """
        if isinstance(image, str):
            # If image is a path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If image is numpy array
            image = Image.fromarray(image)
        
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor, original_size
    
    def postprocess_output(self, output, original_size):
        """
        Postprocess model output to segmentation mask
        """
        # Get predicted class for each pixel
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original size
        pred = cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        
        return pred
    
    def create_color_mask(self, pred_mask):
        """
        Create colored segmentation mask
        """
        h, w = pred_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx in range(self.num_classes):
            mask = pred_mask == class_idx
            color_mask[mask] = self.class_colors[class_idx]
        
        return color_mask
    
    def blend_with_original(self, original_image, color_mask, alpha=0.5):
        """
        Blend segmentation mask with original image
        """
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Ensure same size
        if original_image.shape[:2] != color_mask.shape[:2]:
            color_mask = cv2.resize(color_mask, (original_image.shape[1], original_image.shape[0]))
        
        blended = cv2.addWeighted(original_image, 1-alpha, color_mask, alpha, 0)
        
        return blended
    
    @torch.no_grad()
    def predict(self, image, return_probs=False):
        """
        Perform segmentation on input image
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        output = self.model(image_tensor)
        
        # Postprocess
        pred_mask = self.postprocess_output(output, original_size)
        
        if return_probs:
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            return pred_mask, probs
        
        return pred_mask
    
    def predict_and_visualize(self, image_path, output_path=None, blend_alpha=0.5):
        """
        Predict and create visualization
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Predict
        pred_mask = self.predict(original_image)
        
        # Create color mask
        color_mask = self.create_color_mask(pred_mask)
        
        # Blend with original
        blended = self.blend_with_original(original_image, color_mask, blend_alpha)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        return pred_mask, color_mask, blended
    
    def predict_from_base64(self, base64_str):
        """
        Predict from base64 encoded image (for web API)
        """
        # Decode base64
        image_data = base64.b64decode(base64_str.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Predict
        pred_mask = self.predict(image)
        
        # Create color mask
        color_mask = self.create_color_mask(pred_mask)
        
        # Blend
        blended = self.blend_with_original(image, color_mask, alpha=0.5)
        
        # Convert back to base64
        buffered = BytesIO()
        Image.fromarray(blended).save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{result_base64}"
    
    def get_statistics(self, pred_mask):
        """
        Calculate segmentation statistics
        """
        total_pixels = pred_mask.size
        class_stats = {}
        
        for class_idx in range(self.num_classes):
            class_name = ['Water', 'Vegetation', 'Urban', 'Agriculture', 
                         'Forest', 'Bare Land', 'Road'][class_idx]
            count = np.sum(pred_mask == class_idx)
            percentage = (count / total_pixels) * 100
            
            class_stats[class_name] = {
                'pixels': int(count),
                'percentage': float(percentage)
            }
        
        return class_stats


def batch_inference(model_path, image_dir, output_dir, device='cuda'):
    """
    Perform batch inference on a directory of images
    """
    import os
    from glob import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference
    inferencer = SegmentationInference(model_path, device=device)
    
    # Get all images
    image_paths = glob(os.path.join(image_dir, '*.jpg')) + \
                  glob(os.path.join(image_dir, '*.png')) + \
                  glob(os.path.join(image_dir, '*.tif'))
    
    print(f"Found {len(image_paths)} images")
    
    for img_path in image_paths:
        print(f"Processing: {os.path.basename(img_path)}")
        
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        pred_mask, color_mask, blended = inferencer.predict_and_visualize(
            img_path, output_path
        )
        
        # Get statistics
        stats = inferencer.get_statistics(pred_mask)
        print(f"  Statistics: {stats}")
    
    print("Batch inference completed!")


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best_model.pth"
    image_path = "test_image.jpg"
    
    # Initialize inference
    inferencer = SegmentationInference(model_path, device='cuda')
    
    # Predict and visualize
    pred_mask, color_mask, blended = inferencer.predict_and_visualize(
        image_path, 
        output_path="result.png"
    )
    
    # Get statistics
    stats = inferencer.get_statistics(pred_mask)
    print("Segmentation Statistics:")
    for class_name, stat in stats.items():
        print(f"  {class_name}: {stat['percentage']:.2f}%")
