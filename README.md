# Pulmonary Nodule Detection in Chest X-rays Using YOLOv8

An end-to-end deep learning solution for automated detection and localization of pulmonary nodules in chest X-ray images, designed to assist healthcare professionals in early lung cancer screening.

## 🎯 Project Overview

Early detection of pulmonary nodules is critical for diagnosing lung diseases, particularly lung cancer. This project implements a YOLOv8-based object detection system that can automatically identify and localize nodules in chest X-rays with high accuracy, potentially improving diagnostic efficiency in clinical settings.

## ✨ Key Features

- **High Performance**: Achieved F1-score of 0.72 with 95% detection rate on test images
- **Real-time Detection**: Fast inference suitable for clinical environments  
- **User-friendly Interface**: Streamlit web application for easy interaction
- **Customizable Parameters**: Adjustable confidence and IoU thresholds
- **Medical Focus**: Optimized specifically for small nodule detection in chest X-rays
- **Export Functionality**: Save annotated results for medical documentation

## 🏗️ Architecture

The system uses **YOLOv8s** (small variant) as the optimal balance between accuracy and speed:

- **Backbone**: CSPDarknet53 with Cross-Stage Partial connections
- **Neck**: PANet for multi-scale feature fusion  
- **Head**: Decoupled detection heads for classification and bounding box regression
- **Input Resolution**: 1024×1024 pixels to preserve small nodule details
- **Single Class**: Focused exclusively on "nodule" detection

## 📊 Performance Metrics

- **F1-Score**: 0.72
- **Detection Rate**: 95% of test images had nodules correctly detected
- **Average Nodules per Image**: 3.79
- **Training Time**: 47 minutes on Kaggle T4×2 GPUs
- **Dataset Size**: ~2,000 annotated chest X-ray images

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install pandas
pip install matplotlib
pip install Pillow
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/saadiahmed-hash/Pulmonary_nodule_detection.git
cd Pulmonary_nodule_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Model

```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=1024,
    batch=8,
    patience=20,
    optimizer='AdamW',
    lr0=0.0005,
    weight_decay=0.0005,
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.3,
    mosaic=1.0,
    mixup=0.1
)
```

#### Running the Web Application

```bash
streamlit run app.py
```

#### Making Predictions

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Run inference
results = model('path/to/chest_xray.jpg', conf=0.34, iou=0.45)

# Display results
results[0].show()
```

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── jpg/          # Training X-ray images (~1,500 images)
│   └── anno/         # XML annotation files
├── test/
│   └── jpg/          # Test X-ray images (~500 images)
└── dataset.yaml     # YOLO dataset configuration
```

## 🔧 Data Preprocessing

The preprocessing pipeline converts XML annotations to YOLO format:

1. **XML Parsing**: Extract bounding box coordinates from XML files
2. **Normalization**: Convert coordinates to relative values (0-1)
3. **Format Conversion**: Transform to YOLO format: `class_id center_x center_y width height`
4. **Dataset Split**: 80% training, 20% validation

## 🎛️ Web Application Features

The Streamlit application provides:

- **Image Upload**: Support for various image formats
- **Parameter Tuning**: 
  - Confidence Threshold (default: 0.34)
  - IoU Threshold (default: 0.45)
- **Real-time Visualization**: Bounding boxes with confidence scores
- **Results Export**: Download annotated images
- **Batch Processing**: Handle multiple images

## 📈 Training Configuration

### Hyperparameters
- **Epochs**: 50
- **Batch Size**: 8  
- **Image Size**: 1024×1024
- **Optimizer**: AdamW
- **Learning Rate**: 0.0005
- **Weight Decay**: 0.0005

### Data Augmentation
- Hue variation (±1.5%)
- Saturation adjustment (70%)  
- Brightness variation (40%)
- Horizontal flip (50%)
- Vertical flip (30%)
- Mosaic augmentation (100%)
- Mixup augmentation (10%)

## 🏥 Clinical Applications

This system is designed to assist healthcare professionals in:

- **Screening Programs**: Mass lung cancer screening
- **Diagnostic Support**: Second opinion for radiologists  
- **Education**: Training medical students and residents
- **Rural Healthcare**: AI assistance in areas with limited specialist availability

**Note**: This tool is intended as a supportive technology and should not replace professional medical diagnosis.

## 🧪 Technical Challenges Addressed

1. **Class Imbalance**: Sparse nodule distribution in X-rays
2. **Small Object Detection**: Nodules often <5mm in large images
3. **Anatomical Interference**: False positives from blood vessels
4. **Variable Image Quality**: Different imaging conditions and equipment

## 📋 Results Analysis

### Strengths
- High sensitivity for nodules >5mm
- Robust performance across different image qualities
- Effective multi-nodule detection in single images

### Limitations  
- Reduced accuracy for very small nodules (<3mm)
- Occasional false positives in dense vascular areas
- Limited to 2D chest X-rays only

## 🔮 Future Improvements

- **3D Imaging**: Extend to CT scan analysis
- **Malignancy Classification**: Distinguish benign vs malignant nodules
- **Ensemble Methods**: Combine multiple detection architectures
- **Larger Datasets**: Train on more diverse clinical data
- **Integration**: PACS/DICOM system compatibility

## 📚 References

1. [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
2. Wang, J., et al. (2022). "Deep Learning for Pulmonary Nodule Detection: A Review." IEEE Reviews in Biomedical Engineering
3. [Kaggle Dataset](https://www.kaggle.com/t/ce834067d3e94f73be271a458d01fe1b)

## 👥 Contributors

- **SAADI Ahmed** - ESI-SBA, Algeria
- **Supervisor**: Dr. Dif Nassima

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or collaboration opportunities, please reach out:
- Email: saadi.ahmed.eng@gmail.com
- LinkedIn: [SAADI Ahmed](https://www.linkedin.com/in/saadi-ahmed-898281232)
- GitHub: [@saadiahmed-hash](https://github.com/saadiahmed-hash)

---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. It should not be used as the sole basis for medical decisions. Always consult qualified healthcare professionals for medical diagnosis and treatment.
