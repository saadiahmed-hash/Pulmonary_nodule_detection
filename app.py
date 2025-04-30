import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper: download link for PIL image
def get_image_download_link(img: Image.Image, filename: str, text: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{img_str}" download="{filename}" style="color: #FF4B8B;">{text}</a>'

# Create heatmap from boxes
def create_heatmap(img_shape, boxes):
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        heatmap[y1:y2, x1:x2] += 1
    return heatmap

# Sidebar settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
    st.header("About This Tool")
    st.write("YOLOv8-based pulmonary nodule detection in chest X-rays.")
    st.subheader("Settings")
    confidence_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.1, 0.9, 0.45, 0.05)
    show_advanced = st.checkbox("Show Advanced Analysis")

# Main title and model loading
st.title("🫁 Pulmonary Nodule Detection System")
model = load_model()
if not model:
    st.stop()

# File uploader
uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(arr, caption="Original Image", use_container_width=True)

    # Inference with progress
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress.progress(i+1)
    results = model.predict(source=arr, conf=confidence_threshold, iou=iou_threshold, imgsz=1280)[0]

    # Draw detections using Pillow
    pil_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    boxes = results.boxes.xyxy.cpu().numpy()

    rows = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2, conf = *map(int, box[:4]), float(box[4])
        # draw box
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
        draw.text((x1, y1-10), "Nodule", fill=(0,255,0), font=font)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        draw.text((cx, cy), f"({cx},{cy})", fill=(255,255,0), font=font)
        # metrics
        width, height = x2 - x1, y2 - y1
        area = width * height
        location = ("Upper" if y1 < arr.shape[0]/2 else "Lower") + (
                   " Left" if x2 < arr.shape[1]/2 else " Right")
        rows.append({
            "ID": idx+1,
            "Confidence": f"{conf:.2f}",
            "Size(px²)": area,
            "Location": location,
            "Top-Left": f"({x1},{y1})",
            "Bottom-Right": f"({x2},{y2})",
            "Center": f"({cx},{cy})"
        })

    result_arr = np.array(pil_img)
    with col2:
        st.image(result_arr, caption="Detection Results", use_container_width=True)
        st.markdown(get_image_download_link(pil_img, "nodule_result.png", "Download Image"), unsafe_allow_html=True)

    # Display table
    st.subheader("Detection Details")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Advanced analysis
    if show_advanced and len(boxes) > 0:
        st.subheader("Advanced Analysis")
        # Heatmap
        heatmap = create_heatmap(arr.shape, boxes)
        fig1, ax1 = plt.subplots()
        ax1.imshow(arr)
        hm = ax1.imshow(heatmap, cmap='hot', alpha=0.5)
        fig1.colorbar(hm, ax=ax1, label='Intensity')
        ax1.axis('off')
        st.pyplot(fig1)

        # Confidence distribution
        confs = [float(box[4]) for box in boxes]
        fig2, ax2 = plt.subplots()
        ax2.hist(confs, bins=10, alpha=0.7)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        st.pyplot(fig2)

        # Coordinates map
        fig3, ax3 = plt.subplots()
        ax3.imshow(arr)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            ax3.scatter(cx, cy, marker='x', s=100)
            ax3.text(cx+5, cy, f"{idx+1} ({cx},{cy})", color='white', fontsize=9,
                     bbox=dict(facecolor='black', alpha=0.5))
        ax3.axis('off')
        st.pyplot(fig3)
else:
    st.info("Please upload an image to start analysis.")

# Footer
st.markdown("<div style='text-align:center;color:#CCCCCC;font-size:0.8rem;'>© 2023 Pulmonary Nodule Detection System</div>", unsafe_allow_html=True)