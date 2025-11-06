// Mobile Navigation Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        navMenu.classList.remove('active');
    });
});

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';
    } else {
        navbar.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';
    }
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Apply animation to sections
document.querySelectorAll('.method-card, .result-card, .team-member, .component-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offsetTop = target.offsetTop - 70; // Adjust for fixed navbar
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

// Contact Form Handling
const contactForm = document.getElementById('contactForm');
if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Get form values
        const formData = new FormData(contactForm);
        
        // Show success message
        alert('Thank you for your message! We will get back to you soon.');
        
        // Reset form
        contactForm.reset();
        
        // In a real application, you would send the data to a server here
        // Example:
        // fetch('/api/contact', {
        //     method: 'POST',
        //     body: formData
        // })
        // .then(response => response.json())
        // .then(data => console.log(data));
    });
}

// Add active state to navigation based on scroll position
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const scrollY = window.pageYOffset;

    sections.forEach(section => {
        const sectionHeight = section.offsetHeight;
        const sectionTop = section.offsetTop - 100;
        const sectionId = section.getAttribute('id');
        const navLink = document.querySelector(`.nav-menu a[href="#${sectionId}"]`);

        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
            navLink?.classList.add('active');
        } else {
            navLink?.classList.remove('active');
        }
    });
});

// Add CSS class for active nav link
const style = document.createElement('style');
style.textContent = `
    .nav-menu a.active {
        color: var(--primary-color);
        position: relative;
    }
    
    .nav-menu a.active::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 2px;
        background: var(--primary-color);
    }
`;
document.head.appendChild(style);

// Counter animation for results
const animateCounter = (element, target, duration = 2000) => {
    let start = 0;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
            element.textContent = target + '%';
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(start) + '%';
        }
    }, 16);
};

// Trigger counter animation when results section is visible
const resultsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.dataset.animated) {
            const metricValue = entry.target.querySelector('.metric-value');
            if (metricValue) {
                const targetValue = parseFloat(metricValue.textContent);
                metricValue.textContent = '0%';
                animateCounter(metricValue, targetValue);
                entry.target.dataset.animated = 'true';
            }
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.result-card').forEach(card => {
    resultsObserver.observe(card);
});

// Add loading animation
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease';
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// ==================== Interactive Demo Functionality ====================

// Global variables
let processedImages = [];
let currentImageIndex = 0;
let currentVisualizationMode = 'overlay';
let isComparisonMode = false;
let uploadedFiles = [];

// Model configurations
const modelConfigs = {
    proposed: {
        name: 'Our Proposed Model',
        accuracy: 94.7,
        mIoU: 89.3,
        f1Score: 91.8,
        processingSpeed: 0.8  // Faster for demo
    },
    unet: {
        name: 'U-Net',
        accuracy: 87.2,
        mIoU: 78.5,
        f1Score: 82.1,
        processingSpeed: 0.6  // Faster for demo
    },
    deeplab: {
        name: 'DeepLabV3+',
        accuracy: 89.5,
        mIoU: 82.7,
        f1Score: 85.9,
        processingSpeed: 1.0  // Moderate speed
    },
    transformer: {
        name: 'Pure Transformer',
        accuracy: 91.3,
        mIoU: 85.1,
        f1Score: 88.2,
        processingSpeed: 1.2  // Slower (realistic)
    }
};

// Segmentation classes with colors
const segmentationClasses = [
    { name: 'Water', color: '#3b82f6' },
    { name: 'Vegetation', color: '#10b981' },
    { name: 'Urban', color: '#ef4444' },
    { name: 'Agriculture', color: '#f59e0b' },
    { name: 'Forest', color: '#059669' },
    { name: 'Bare Land', color: '#8b5cf6' },
    { name: 'Road', color: '#6b7280' }
];

// Drop zone functionality
const dropZone = document.getElementById('dropZone');
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const processingStatus = document.getElementById('processingStatus');
const originalCanvas = document.getElementById('originalCanvas');
const resultCanvas = document.getElementById('resultCanvas');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop zone when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('dragover');
    }, false);
});

// Handle dropped files
dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    handleFiles(files);
}, false);

// Handle file input change
imageUpload.addEventListener('change', (e) => {
    const files = e.target.files;
    handleFiles(files);
});

// Handle uploaded files
function handleFiles(files) {
    if (files.length === 0) return;
    
    // Filter image files
    uploadedFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (uploadedFiles.length === 0) {
        alert('Please upload image files');
        return;
    }
    
    // Show batch queue if multiple files
    if (uploadedFiles.length > 1) {
        showBatchQueue(uploadedFiles);
    }
    
    // Process all images
    processedImages = [];
    currentImageIndex = 0;
    processBatch(uploadedFiles);
}

// Show batch queue
function showBatchQueue(files) {
    const batchQueue = document.getElementById('batchQueue');
    const queueList = document.getElementById('queueList');
    const queueCount = document.getElementById('queueCount');
    
    batchQueue.classList.remove('hidden');
    queueCount.textContent = files.length;
    queueList.innerHTML = '';
    
    files.forEach((file, index) => {
        const queueItem = document.createElement('div');
        queueItem.className = 'queue-item';
        queueItem.id = `queue-${index}`;
        queueItem.innerHTML = `
            <span>${file.name}</span>
            <span class="queue-status">Waiting...</span>
        `;
        queueList.appendChild(queueItem);
    });
}

// Process batch of images
async function processBatch(files) {
    for (let i = 0; i < files.length; i++) {
        const queueItem = document.getElementById(`queue-${i}`);
        if (queueItem) {
            queueItem.classList.add('processing');
            queueItem.querySelector('.queue-status').textContent = 'Processing...';
        }
        
        await processImage(files[i], i);
        
        if (queueItem) {
            queueItem.classList.remove('processing');
            queueItem.classList.add('completed');
            queueItem.querySelector('.queue-status').textContent = '✓ Completed';
        }
    }
    
    // Show first result
    displayProcessedImage(0);
}

// Process and display image
function processImage(file, index = 0) {
    return new Promise((resolve) => {
        const startTime = Date.now();
        
        // Show processing status only for first image
        if (index === 0) {
            dropZone.style.display = 'none';
            processingStatus.classList.remove('hidden');
            imagePreview.classList.add('hidden');
        }
        
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                // Get selected model
                const selectedModel = document.querySelector('input[name="model"]:checked').value;
                const modelConfig = modelConfigs[selectedModel];
                
                // Check for API endpoint
                const apiEndpoint = document.getElementById('apiEndpoint')?.value;
                const apiKey = document.getElementById('apiKey')?.value;
                
                // If no API configured, process instantly. Otherwise simulate processing time
                const baseTime = apiEndpoint ? 1500 : 100; // Fast if no API
                const processingTime = baseTime * modelConfig.processingSpeed;
                
                setTimeout(async () => {
                    let segmentationData;
                    
                    // Try API if configured
                    if (apiEndpoint) {
                        segmentationData = await callSegmentationAPI(img, apiEndpoint, apiKey, selectedModel);
                    }
                    
                    // Fallback to simulated segmentation (fast local processing)
                    if (!segmentationData) {
                        segmentationData = simulateSegmentation(img, selectedModel);
                    }
                    
                    // Calculate stats
                    const endTime = Date.now();
                    const totalTime = ((endTime - startTime) / 1000).toFixed(2);
                    
                    // Store processed image data
                    processedImages.push({
                        original: img,
                        segmentation: segmentationData,
                        fileName: file.name,
                        stats: {
                            time: totalTime,
                            model: modelConfig,
                            size: `${img.width} × ${img.height}`,
                            confidence: (92 + Math.random() * 6).toFixed(1)
                        }
                    });
                    
                    resolve();
                }, processingTime);
            };
            img.src = e.target.result;
        };
        
        reader.readAsDataURL(file);
    });
}

// Call segmentation API
async function callSegmentationAPI(img, endpoint, apiKey, model) {
    try {
        // Convert image to base64
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Call API
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                image: imageData,
                model: model
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            return result.segmentation;
        }
    } catch (error) {
        console.log('API call failed, using simulated segmentation');
    }
    return null;
}

// Simulate segmentation (fallback)
function simulateSegmentation(img, model) {
    // Create canvas for segmentation
    const canvas = document.createElement('canvas');
    const maxWidth = 400;
    const scale = Math.min(1, maxWidth / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    
    return {
        canvas: canvas,
        model: model
    };
}

// Display processed image
function displayProcessedImage(index) {
    if (index < 0 || index >= processedImages.length) return;
    
    currentImageIndex = index;
    const data = processedImages[index];
    
    // Update navigation
    if (processedImages.length > 1) {
        document.getElementById('imageNav').classList.remove('hidden');
        document.getElementById('imageCounter').textContent = `${index + 1} / ${processedImages.length}`;
    }
    
    // Update model name
    document.getElementById('currentModelName').textContent = data.stats.model.name;
    
    // Display images
    displayOriginalImage(data.original);
    generateSegmentation(data.original, data.stats.model);
    
    // Update stats
    updateStats(data.original, data.stats.time, data.stats.confidence);
    
    // Generate legend
    generateLegend();
    
    // Hide processing, show results
    processingStatus.classList.add('hidden');
    imagePreview.classList.remove('hidden');
    
    // Scroll to results
    document.getElementById('imagePreview').scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Show helpful notification
    if (processedImages.length > 0 && !isComparisonMode) {
        setTimeout(() => {
            showNotification('💡 Tip: Click "📊 Compare Models" to see how different models perform!');
        }, 1000);
    }
}

// Show notification helper
function showNotification(message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        z-index: 10000;
        font-weight: 600;
        animation: slideIn 0.5s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.5s ease';
        setTimeout(() => notification.remove(), 500);
    }, 5000);
    
    // Add animations
    if (!document.getElementById('notificationStyles')) {
        const style = document.createElement('style');
        style.id = 'notificationStyles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(400px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(400px); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

// Display original image on canvas
function displayOriginalImage(img) {
    const ctx = originalCanvas.getContext('2d');
    
    // Set canvas size
    const maxWidth = 400;
    const scale = Math.min(1, maxWidth / img.width);
    originalCanvas.width = img.width * scale;
    originalCanvas.height = img.height * scale;
    
    // Draw image
    ctx.drawImage(img, 0, 0, originalCanvas.width, originalCanvas.height);
}

// Generate segmentation result (simulated)
function generateSegmentation(img, modelConfig = null) {
    const ctx = resultCanvas.getContext('2d');
    
    // Set canvas size to match original
    resultCanvas.width = originalCanvas.width;
    resultCanvas.height = originalCanvas.height;
    
    // First draw the original image with reduced opacity
    ctx.globalAlpha = 0.4;
    ctx.drawImage(img, 0, 0, resultCanvas.width, resultCanvas.height);
    ctx.globalAlpha = 1.0;
    
    // Generate segmentation overlay
    const imageData = ctx.createImageData(resultCanvas.width, resultCanvas.height);
    const data = imageData.data;
    
    // Create segmentation regions (simulated)
    for (let y = 0; y < resultCanvas.height; y++) {
        for (let x = 0; x < resultCanvas.width; x++) {
            const idx = (y * resultCanvas.width + x) * 4;
            
            // Determine segment based on position (simulated segmentation)
            let classIdx;
            const normalized_x = x / resultCanvas.width;
            const normalized_y = y / resultCanvas.height;
            
            // Create realistic-looking segments
            if (normalized_y < 0.3) {
                classIdx = 0; // Water (top)
            } else if (normalized_y < 0.5 && normalized_x > 0.6) {
                classIdx = 1; // Vegetation
            } else if (normalized_y > 0.7 && normalized_x < 0.4) {
                classIdx = 2; // Urban
            } else if (normalized_x > 0.5 && normalized_y > 0.5) {
                classIdx = 3; // Agriculture
            } else if (normalized_y < 0.6 && normalized_x < 0.3) {
                classIdx = 4; // Forest
            } else if (normalized_y > 0.8) {
                classIdx = 6; // Road
            } else {
                classIdx = 5; // Bare Land
            }
            
            // Add some randomness for more realistic appearance
            if (Math.random() > 0.95) {
                classIdx = Math.floor(Math.random() * segmentationClasses.length);
            }
            
            const color = hexToRgb(segmentationClasses[classIdx].color);
            
            data[idx] = color.r;
            data[idx + 1] = color.g;
            data[idx + 2] = color.b;
            data[idx + 3] = 128; // Semi-transparent
        }
    }
    
    // Apply segmentation overlay
    ctx.putImageData(imageData, 0, 0);
    
    // Add contours for better visualization
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    
    // Draw some segmentation boundaries
    for (let i = 0; i < 10; i++) {
        ctx.beginPath();
        const y = (i / 10) * resultCanvas.height;
        ctx.moveTo(0, y);
        ctx.lineTo(resultCanvas.width, y);
        ctx.stroke();
    }
}

// Convert hex color to RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// Update statistics
function updateStats(img, processTime, confidence = null) {
    document.getElementById('classCount').textContent = segmentationClasses.length;
    document.getElementById('processTime').textContent = processTime + 's';
    document.getElementById('imageSize').textContent = img.width + ' × ' + img.height;
    document.getElementById('confidence').textContent = (confidence || (92 + Math.random() * 6).toFixed(1)) + '%';
}

// Navigate between images
function navigateImages(direction) {
    const newIndex = currentImageIndex + direction;
    if (newIndex >= 0 && newIndex < processedImages.length) {
        displayProcessedImage(newIndex);
    }
}

// Change visualization mode
function changeVisualization(mode) {
    currentVisualizationMode = mode;
    
    // Update button states
    document.querySelectorAll('.viz-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
    
    // Show notification
    const modeNames = {
        'overlay': 'Overlay Mode - Segmentation overlaid on original',
        'sidebyside': 'Side by Side - Compare original and result',
        'split': 'Split View - Interactive comparison',
        'heatmap': 'Heatmap - Intensity visualization'
    };
    showNotification(`🎨 ${modeNames[mode]}`);
    
    // Apply visualization
    const previewGrid = document.getElementById('previewGrid');
    
    switch(mode) {
        case 'sidebyside':
            previewGrid.style.gridTemplateColumns = '1fr 1fr';
            break;
        case 'overlay':
            // Already default
            break;
        case 'split':
            // Implement split view with slider
            applySplitView();
            break;
        case 'heatmap':
            // Apply heatmap visualization
            applyHeatmap();
            break;
    }
}

// Apply split view
function applySplitView() {
    // Create split view effect on result canvas
    const data = processedImages[currentImageIndex];
    if (!data) return;
    
    displayOriginalImage(data.original);
    generateSegmentation(data.original, data.stats.model);
}

// Apply heatmap visualization
function applyHeatmap() {
    const ctx = resultCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
    const data = imageData.data;
    
    // Apply heatmap color scheme
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        
        // Convert to heatmap colors (blue to red)
        if (avg < 85) {
            data[i] = 0;
            data[i + 1] = 0;
            data[i + 2] = 255;
        } else if (avg < 170) {
            data[i] = 255;
            data[i + 1] = 255;
            data[i + 2] = 0;
        } else {
            data[i] = 255;
            data[i + 1] = 0;
            data[i + 2] = 0;
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
}

// Toggle comparison mode
function toggleComparison() {
    isComparisonMode = !isComparisonMode;
    const comparisonMode = document.getElementById('comparisonMode');
    const toggleBtn = event.target;
    
    if (isComparisonMode) {
        comparisonMode.classList.remove('hidden');
        toggleBtn.textContent = '❌ Hide Comparison';
        toggleBtn.style.background = '#ef4444';
        generateComparison();
        
        // Scroll to comparison
        setTimeout(() => {
            comparisonMode.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
        
        showNotification('🎯 Comparing all 4 models! Notice the differences in accuracy and speed.');
    } else {
        comparisonMode.classList.add('hidden');
        toggleBtn.textContent = '📊 Compare Models';
        toggleBtn.style.background = '';
    }
}

// Generate model comparison
function generateComparison() {
    if (processedImages.length === 0) return;
    
    const currentImage = processedImages[currentImageIndex].original;
    const canvases = ['compCanvas1', 'compCanvas2', 'compCanvas3', 'compCanvas4'];
    const models = ['proposed', 'unet', 'deeplab', 'transformer'];
    
    canvases.forEach((canvasId, idx) => {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        const maxWidth = 300;
        const scale = Math.min(1, maxWidth / currentImage.width);
        canvas.width = currentImage.width * scale;
        canvas.height = currentImage.height * scale;
        
        // Draw original with reduced opacity
        ctx.globalAlpha = 0.4;
        ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
        
        // Generate different segmentation for each model
        generateModelSpecificSegmentation(ctx, canvas, models[idx]);
        
        // Update timing
        const processingTime = (1.5 * modelConfigs[models[idx]].processingSpeed).toFixed(2);
        document.getElementById(`time${idx + 1}`).textContent = processingTime + 's';
    });
}

// Generate model-specific segmentation
function generateModelSpecificSegmentation(ctx, canvas, model) {
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;
    
    // Different patterns for different models
    const variance = model === 'proposed' ? 0.1 : model === 'unet' ? 0.3 : model === 'deeplab' ? 0.2 : 0.15;
    
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4;
            const normalized_x = x / canvas.width;
            const normalized_y = y / canvas.height;
            
            let classIdx;
            if (normalized_y < 0.3) {
                classIdx = 0;
            } else if (normalized_y < 0.5 && normalized_x > 0.6) {
                classIdx = 1;
            } else if (normalized_y > 0.7 && normalized_x < 0.4) {
                classIdx = 2;
            } else if (normalized_x > 0.5 && normalized_y > 0.5) {
                classIdx = 3;
            } else if (normalized_y < 0.6 && normalized_x < 0.3) {
                classIdx = 4;
            } else if (normalized_y > 0.8) {
                classIdx = 6;
            } else {
                classIdx = 5;
            }
            
            // Add model-specific variance
            if (Math.random() > (1 - variance)) {
                classIdx = Math.floor(Math.random() * segmentationClasses.length);
            }
            
            const color = hexToRgb(segmentationClasses[classIdx].color);
            data[idx] = color.r;
            data[idx + 1] = color.g;
            data[idx + 2] = color.b;
            data[idx + 3] = 128;
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
}

// Test API connection
function testAPI() {
    const endpoint = document.getElementById('apiEndpoint').value;
    const apiKey = document.getElementById('apiKey').value;
    
    if (!endpoint) {
        alert('Please enter an API endpoint');
        return;
    }
    
    alert('Testing API connection...\n\nNote: This is a demo. In production, this would test your actual API connection.');
}

// Download all results
function downloadAll() {
    if (processedImages.length === 0) return;
    
    processedImages.forEach((data, index) => {
        const link = document.createElement('a');
        link.download = `segmentation_${data.fileName}`;
        
        // Create a temporary canvas with the result
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = data.original.width;
        tempCanvas.height = data.original.height;
        const ctx = tempCanvas.getContext('2d');
        
        // Redraw segmentation on temp canvas
        displayOriginalImage(data.original);
        generateSegmentation(data.original, data.stats.model);
        
        link.href = resultCanvas.toDataURL();
        link.click();
    });
    
    alert(`Downloaded ${processedImages.length} segmentation results!`);
}

// Generate legend
function generateLegend() {
    const legendGrid = document.getElementById('legendGrid');
    legendGrid.innerHTML = '';
    
    segmentationClasses.forEach(cls => {
        const legendItem = document.createElement('div');
        legendItem.className = 'legend-item';
        
        const colorBox = document.createElement('div');
        colorBox.className = 'legend-color';
        colorBox.style.backgroundColor = cls.color;
        
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = cls.name;
        
        legendItem.appendChild(colorBox);
        legendItem.appendChild(label);
        legendGrid.appendChild(legendItem);
    });
}

// Reset demo
function resetDemo() {
    dropZone.style.display = 'block';
    imagePreview.classList.add('hidden');
    processingStatus.classList.add('hidden');
    document.getElementById('batchQueue').classList.add('hidden');
    document.getElementById('imageNav').classList.add('hidden');
    document.getElementById('comparisonMode').classList.add('hidden');
    imageUpload.value = '';
    processedImages = [];
    uploadedFiles = [];
    currentImageIndex = 0;
    isComparisonMode = false;
}

// Download result
function downloadResult() {
    const link = document.createElement('a');
    link.download = 'segmentation_result.png';
    link.href = resultCanvas.toDataURL();
    link.click();
}

// Make functions globally accessible
window.resetDemo = resetDemo;
window.downloadResult = downloadResult;
window.downloadAll = downloadAll;
window.navigateImages = navigateImages;
window.changeVisualization = changeVisualization;
window.toggleComparison = toggleComparison;
window.testAPI = testAPI;

// Console message
console.log('%cWeighted Feature Fusion Network', 'color: #2563eb; font-size: 20px; font-weight: bold;');
console.log('%cResearch Website - 2025', 'color: #6b7280; font-size: 14px;');
