"""
Flask API Server for Model Inference
Connects your trained model with the website
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time

from inference import SegmentationInference

app = Flask(__name__)
CORS(app)  # Enable CORS for website integration

# Initialize models
MODELS = {
    'proposed': None,
    'unet': None,
    'deeplab': None,
    'transformer': None
}

# Model paths
MODEL_PATHS = {
    'proposed': 'checkpoints/weighted_fusion_best.pth',
    'unet': 'checkpoints/unet_best.pth',
    'deeplab': 'checkpoints/deeplab_best.pth',
    'transformer': 'checkpoints/transformer_best.pth'
}


def initialize_models():
    """Initialize all models on server start"""
    print("Loading models...")
    for model_name, model_path in MODEL_PATHS.items():
        try:
            MODELS[model_name] = SegmentationInference(model_path, device='cuda')
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
    print("Models ready!")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {k: v is not None for k, v in MODELS.items()}
    })


@app.route('/api/segment', methods=['POST'])
def segment_image():
    """
    Main segmentation endpoint
    Expects JSON: {
        "image": "base64_encoded_image",
        "model": "proposed|unet|deeplab|transformer"
    }
    """
    try:
        data = request.get_json()
        
        # Get parameters
        image_base64 = data.get('image')
        model_name = data.get('model', 'proposed')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        if model_name not in MODELS:
            return jsonify({'error': f'Invalid model: {model_name}'}), 400
        
        if MODELS[model_name] is None:
            return jsonify({'error': f'Model {model_name} not loaded'}), 500
        
        # Start timing
        start_time = time.time()
        
        # Decode image
        image_data = base64.b64decode(image_base64.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Perform segmentation
        inferencer = MODELS[model_name]
        pred_mask = inferencer.predict(image)
        
        # Create visualization
        color_mask = inferencer.create_color_mask(pred_mask)
        blended = inferencer.blend_with_original(image, color_mask, alpha=0.5)
        
        # Get statistics
        stats = inferencer.get_statistics(pred_mask)
        
        # Convert result to base64
        buffered = BytesIO()
        Image.fromarray(blended).save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response
        return jsonify({
            'success': True,
            'segmentation': f"data:image/png;base64,{result_base64}",
            'statistics': stats,
            'processing_time': round(processing_time, 3),
            'model': model_name,
            'image_size': f"{image.width}x{image.height}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch_segment', methods=['POST'])
def batch_segment():
    """
    Batch segmentation endpoint
    Expects JSON: {
        "images": ["base64_1", "base64_2", ...],
        "model": "proposed"
    }
    """
    try:
        data = request.get_json()
        
        images_base64 = data.get('images', [])
        model_name = data.get('model', 'proposed')
        
        if not images_base64:
            return jsonify({'error': 'No images provided'}), 400
        
        if model_name not in MODELS or MODELS[model_name] is None:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        inferencer = MODELS[model_name]
        results = []
        
        for idx, img_base64 in enumerate(images_base64):
            try:
                start_time = time.time()
                
                # Decode and process
                image_data = base64.b64decode(img_base64.split(',')[1])
                image = Image.open(BytesIO(image_data)).convert('RGB')
                
                # Segment
                pred_mask = inferencer.predict(image)
                color_mask = inferencer.create_color_mask(pred_mask)
                blended = inferencer.blend_with_original(image, color_mask, alpha=0.5)
                
                # Get stats
                stats = inferencer.get_statistics(pred_mask)
                
                # Convert to base64
                buffered = BytesIO()
                Image.fromarray(blended).save(buffered, format="PNG")
                result_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                processing_time = time.time() - start_time
                
                results.append({
                    'index': idx,
                    'success': True,
                    'segmentation': f"data:image/png;base64,{result_base64}",
                    'statistics': stats,
                    'processing_time': round(processing_time, 3)
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_images': len(images_base64)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/compare_models', methods=['POST'])
def compare_models():
    """
    Compare all models on the same image
    """
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image_data = base64.b64decode(image_base64.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        results = {}
        
        for model_name, inferencer in MODELS.items():
            if inferencer is None:
                continue
            
            try:
                start_time = time.time()
                
                # Segment
                pred_mask = inferencer.predict(image)
                color_mask = inferencer.create_color_mask(pred_mask)
                blended = inferencer.blend_with_original(image, color_mask, alpha=0.5)
                
                # Get stats
                stats = inferencer.get_statistics(pred_mask)
                
                # Convert to base64
                buffered = BytesIO()
                Image.fromarray(blended).save(buffered, format="PNG")
                result_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                processing_time = time.time() - start_time
                
                results[model_name] = {
                    'success': True,
                    'segmentation': f"data:image/png;base64,{result_base64}",
                    'statistics': stats,
                    'processing_time': round(processing_time, 3)
                }
                
            except Exception as e:
                results[model_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True)
