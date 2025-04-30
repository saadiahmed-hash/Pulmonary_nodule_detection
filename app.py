import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import time
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Pulmonary Nodule Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #FF4B8B;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #CCCCCC;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF4B8B;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    
    .stProgress .st-bo {
        background-color: #FF4B8B;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #1E1E1E;
    }
    
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #CCCCCC;
        font-size: 0.8rem;
    }
    
    .analysis-results {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .about-tool-card {
        background-color: #00BCD4;
        background-opacity: 0.2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    
    .stDataFrame {
        background-color: #2D2D2D;
    }
    
    .stDataFrame [data-testid="stTable"] {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    
    .stDataFrame [data-testid="stTable"] th {
        background-color: #3D3D3D;
        color: #FFFFFF;
    }
    
    .stDataFrame [data-testid="stTable"] td {
        color: #FFFFFF;
    }
    
    /* Style for sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        border-right: 1px solid #333333;
    }
    
    /* Style for buttons */
    .stButton button {
        background-color: #FF4B8B;
        color: white;
    }
    
    /* Style for sliders */
    .stSlider [data-testid="stThumbValue"] {
        color: #FF4B8B;
    }
    
    /* Style for file uploader */
    .stFileUploader {
        background-color: #2D2D2D;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Style for success/info/error messages */
    .stSuccess, .stInfo, .stError {
        background-color: #2D2D2D;
        color: white;
        border-radius: 0.5rem;
    }
    
    /* Hide deprecation warnings */
    .stWarning {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to get downloadable image
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" style="color: #FF4B8B;">{text}</a>'
    return href

# Function to create heatmap of detection areas
def create_heatmap(img_shape, boxes):
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        heatmap[y1:y2, x1:x2] += 1
    return heatmap

# Sidebar content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
    st.markdown("<h2>About This Tool</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-tool-card">
    This application uses a YOLOv8 model trained specifically for detecting pulmonary nodules in chest X-ray images. 
    Pulmonary nodules are small, round or oval-shaped growths in the lungs that may indicate various conditions.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Information")
    st.markdown("- **Model Type**: YOLOv8 Custom")
    st.markdown("- **Training Dataset**: Chest X-ray Images")
    st.markdown("- **Primary Use**: Early detection screening")
    
    st.markdown("### Settings")
    confidence_threshold = st.slider("Detection Confidence Threshold", 
                                    min_value=0.1, 
                                    max_value=0.9, 
                                    value=0.3, 
                                    step=0.05,
                                    help="Adjust the confidence threshold for detection")
    
    iou_threshold = st.slider("IOU Threshold", 
                             min_value=0.1, 
                             max_value=0.9, 
                             value=0.45, 
                             step=0.05,
                             help="Intersection over Union threshold")
    
    show_advanced = st.checkbox("Show Advanced Analysis", value=False)

# Main content
st.markdown('<h1 class="main-header">🫁 Pulmonary Nodule Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered detection of lung nodules in chest X-rays</p>', unsafe_allow_html=True)

# Load model
with st.spinner("Initializing AI model..."):
    model = load_model()
    if model:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load model. Please check if the model file exists.")
        st.stop()

# Upload and Detect section
st.markdown("### Upload a Chest X-ray Image")
st.write("Upload a chest X-ray image and the AI will analyze it for potential pulmonary nodules.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to OpenCV image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        # Changed use_column_width to use_container_width
        st.image(img_array, use_container_width=True)
    
    # Run YOLOv8 prediction
    with st.spinner("Analyzing image for pulmonary nodules..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        results = model.predict(source=img_array, conf=confidence_threshold, iou=iou_threshold, imgsz=1280)[0]
        
        # Create a copy for drawing
        img_result = img_array.copy()
        
        # Draw boxes
        boxes = []
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add a label
            cv2.putText(img_result, "Nodule", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add coordinates on the image
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            coord_text = f"({center_x}, {center_y})"
            cv2.putText(img_result, coord_text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    with col2:
        st.markdown("#### Detection Results")
        # Changed use_column_width to use_container_width
        st.image(img_result, use_container_width=True)
        
        # Create download link for the result image
        result_pil = Image.fromarray(img_result)
        st.markdown(get_image_download_link(result_pil, "nodule_detection_result.png", "Download Result Image"), unsafe_allow_html=True)
    
    # Display detection details with improved background
    st.markdown("### Analysis Results 🔗")
    
    if len(results.boxes) > 0:
        # Create metrics in a row
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
            <h3 style="color: #1E1E1E;">Nodules Detected</h3>
            <h2 style="color: #1E1E1E; font-size: 2rem;">{len(results.boxes)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            avg_conf = np.mean([box[4] for box in results.boxes.data.cpu().numpy()])
            st.markdown(f"""
            <div class="metric-card">
            <h3 style="color: #1E1E1E;">Average Confidence</h3>
            <h2 style="color: #1E1E1E; font-size: 2rem;">{avg_conf:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            avg_size = np.mean([(box[2]-box[0])*(box[3]-box[1]) for box in results.boxes.data.cpu().numpy()])
            st.markdown(f"""
            <div class="metric-card">
            <h3 style="color: #1E1E1E;">Average Size (px²)</h3>
            <h2 style="color: #1E1E1E; font-size: 2rem;">{int(avg_size)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a table with detailed information
        st.markdown("### Detailed Findings")
        
        data = []
        for i, box in enumerate(results.boxes.data.cpu().numpy()):
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            location = "Upper" if y1 < img_array.shape[0]/2 else "Lower"
            location += " " + ("Right" if x1 < img_array.shape[1]/2 else "Left")
            
            # Calculate center coordinates
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            data.append({
                "ID": i+1,
                "Confidence": f"{conf:.2f}",
                "Size (px²)": f"{int(area)}",
                "Location": location,
                "Dimensions": f"{int(width)}×{int(height)}",
                "Top-Left": f"({x1}, {y1})",
                "Bottom-Right": f"({x2}, {y2})",
                "Center": f"({center_x}, {center_y})"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Advanced analysis if enabled
        if show_advanced:
            st.markdown("#### Advanced Analysis")
            
            # Set matplotlib style for dark theme
            plt.style.use('dark_background')
            
            adv_cols = st.columns(2)
            
            with adv_cols[0]:
                # Create heatmap
                heatmap = create_heatmap(img_array.shape, results.boxes.data.cpu().numpy())
                plt.figure(figsize=(8, 6))
                plt.imshow(img_array)
                plt.imshow(heatmap, alpha=0.5, cmap='hot')
                plt.colorbar(label='Detection Intensity')
                plt.title('Nodule Detection Heatmap')
                plt.axis('off')
                st.pyplot(plt)
            
            with adv_cols[1]:
                # Create confidence distribution
                confidences = [box[4] for box in results.boxes.data.cpu().numpy()]
                plt.figure(figsize=(8, 6))
                plt.hist(confidences, bins=10, alpha=0.7, color='#FF4B8B')
                plt.xlabel('Confidence Score')
                plt.ylabel('Number of Detections')
                plt.title('Confidence Score Distribution')
                plt.grid(alpha=0.3)
                st.pyplot(plt)
            
            # Size distribution
            sizes = [(box[2]-box[0])*(box[3]-box[1]) for box in results.boxes.data.cpu().numpy()]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sizes)), sizes, alpha=0.7, color='#00BCD4')
            plt.xlabel('Nodule ID')
            plt.ylabel('Size (px²)')
            plt.title('Nodule Size Distribution')
            plt.grid(alpha=0.3)
            st.pyplot(plt)
            
            # Coordinates visualization
            st.markdown("#### Nodule Coordinates Map")
            plt.figure(figsize=(10, 8))
            plt.imshow(img_array, alpha=0.7)
            
            # Plot the centers of the detected nodules
            for i, box in enumerate(results.boxes.data.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                plt.scatter(center_x, center_y, s=100, c='#FF4B8B', marker='x', linewidths=2)
                plt.text(center_x + 10, center_y, f"Nodule {i+1}\n({center_x}, {center_y})", 
                         color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
            
            plt.title('Nodule Coordinate Map')
            plt.axis('off')
            st.pyplot(plt)
            
    else:
        st.success("No pulmonary nodules detected in this image.")
        st.markdown("""
        <div class="info-box">
        This does not guarantee the absence of nodules. If you have concerns, please consult with a healthcare professional.
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Please upload a chest X-ray image to begin analysis.")

# Footer
st.markdown("""
<div class="footer">
© 2023 Pulmonary Nodule Detection System
</div>
""", unsafe_allow_html=True)