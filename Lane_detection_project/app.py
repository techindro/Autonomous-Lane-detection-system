"""
Streamlit Web Application for Autonomous Lane Detection System
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image

# Add src to system path
sys.path.append(str(Path(__file__).parent / "src"))

from src.traditional.hough_detector import HoughLaneDetector
from src.traditional.sliding_window import SlidingWindowDetector
from src.utils.visualization import LaneVisualizer
from src.config import config

# Set Page Configuration
st.set_page_config(
    page_title="Autonomous Lane Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4A90E2 0%, #50E3C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1A202C;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #CBD5E0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #63B3ED;
        margin-top: 0.3rem;
    }
    .status-ok {
        color: #48BB78;
        font-weight: bold;
    }
    .status-warning {
        color: #F6AD55;
        font-weight: bold;
    }
    .status-alert {
        color: #F56565;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚗 Autonomous Lane Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time computer vision & deep learning lane perception pipeline</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Controls & Settings")

mode = st.sidebar.radio(
    "Select Mode",
    ["📷 Image Processing", "🎥 Video Processing", "📊 Method Comparison", "⚙️ Algorithm Config"]
)

method_choice = st.sidebar.selectbox(
    "Detection Method",
    ["traditional", "sliding_window", "deep_learning", "hybrid"],
    format_func=lambda x: {
        "traditional": "Hough Line Transform",
        "sliding_window": "Sliding Window Polynomial",
        "deep_learning": "Deep Learning (U-Net)",
        "hybrid": "Hybrid Ensemble"
    }[x]
)

@st.cache_resource
def get_detector(method: str):
    if method == "traditional":
        return HoughLaneDetector()
    elif method == "sliding_window":
        return SlidingWindowDetector()
    elif method == "deep_learning":
        from src.deep_learning.predictor import DeepLearningPredictor
        return DeepLearningPredictor()
    return None

visualizer = LaneVisualizer()

if mode == "📷 Image Processing":
    st.subheader("Image Lane Detection")
    
    input_source = st.radio("Image Source", ["Sample Images", "Upload Custom Image"], horizontal=True)
    
    image = None
    image_name = ""
    
    if input_source == "Sample Images":
        sample_dir = Path(__file__).parent / "test_image"
        sample_files = sorted(list(sample_dir.glob("*.jpg")))
        if sample_files:
            selected_file = st.selectbox("Choose sample image", sample_files, format_func=lambda x: x.name)
            image = cv2.imread(str(selected_file))
            image_name = selected_file.name
    else:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_name = uploaded_file.name

    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Original: {image_name}", use_container_width=True)
            
        with st.spinner("Processing image..."):
            start_time = time.time()
            
            if method_choice == "hybrid":
                detector_hough = HoughLaneDetector()
                detector_sw = SlidingWindowDetector()
                res_hough = detector_hough.detect(image.copy())
                res_sw = detector_sw.detect(image.copy())
                processed_img = res_sw.get("visualization", image.copy())
                curvature = res_sw.get("curvature", 0)
                offset = res_sw.get("offset", 0)
            else:
                detector = get_detector(method_choice)
                results = detector.detect(image)
                
                if "visualization" in results:
                    processed_img = results["visualization"]
                elif "left_lane" in results:
                    processed_img = visualizer.draw_lanes(image, results.get("left_lane"), results.get("right_lane"))
                    processed_img = visualizer.draw_lane_area(processed_img, results.get("left_lane"), results.get("right_lane"))
                    processed_img = visualizer.draw_curvature_text(processed_img, results.get("curvature", 0), results.get("offset", 0))
                else:
                    processed_img = image.copy()
                    
                curvature = results.get("curvature", 0)
                offset = results.get("offset", 0)
                
            proc_time = (time.time() - start_time) * 1000

        with col2:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"Detected Lanes ({method_choice})", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Detection Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Latency</div><div class="metric-value">{proc_time:.1f} ms</div></div>', unsafe_allow_html=True)
        with m2:
            curv_str = f"{curvature:.1f} m" if curvature < 1000 and curvature > 0 else "Straight"
            st.markdown(f'<div class="metric-card"><div class="metric-title">Road Curvature</div><div class="metric-value">{curv_str}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Vehicle Offset</div><div class="metric-value">{abs(offset):.2f} m</div></div>', unsafe_allow_html=True)
        with m4:
            status_cls = "status-ok" if abs(offset) <= 0.3 else ("status-warning" if abs(offset) <= 0.5 else "status-alert")
            status_txt = "CENTERED" if abs(offset) <= 0.3 else ("DRIFTING" if abs(offset) <= 0.5 else "DEPARTURE!")
            st.markdown(f'<div class="metric-card"><div class="metric-title">Lane Status</div><div class="metric-value {status_cls}">{status_txt}</div></div>', unsafe_allow_html=True)

elif mode == "🎥 Video Processing":
    st.subheader("Video Processing Pipeline")
    
    video_source = st.radio("Video Source", ["Sample Video (test_video.mp4)", "Upload Custom Video"], horizontal=True)
    video_path = None
    
    if video_source == "Sample Video (test_video.mp4)":
        sample_vid = Path(__file__).parent / "test_video.mp4"
        if sample_vid.exists():
            video_path = str(sample_vid)
    else:
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
    if video_path and os.path.exists(video_path):
        st.video(video_path)
        
        if st.button("🚀 Process Video Pipeline", type="primary"):
            st.info("Processing video frames... Output will be rendered below.")
            
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"web_processed_{method_choice}.mp4")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            detector = HoughLaneDetector() if method_choice == "traditional" else SlidingWindowDetector()
            
            frame_idx = 0
            start_proc = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                results = detector.detect(frame)
                if "visualization" in results:
                    vis_frame = results["visualization"]
                elif "left_lane" in results:
                    vis_frame = visualizer.draw_lanes(frame, results.get("left_lane"), results.get("right_lane"))
                    vis_frame = visualizer.draw_lane_area(vis_frame, results.get("left_lane"), results.get("right_lane"))
                    vis_frame = visualizer.draw_curvature_text(vis_frame, results.get("curvature", 0), results.get("offset", 0))
                else:
                    vis_frame = frame
                    
                out.write(vis_frame)
                
                if total_frames > 0 and frame_idx % 5 == 0:
                    pct = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(pct)
                    status_text.text(f"Processing frame {frame_idx}/{total_frames} ({pct*100:.1f}%)")
                    
            cap.release()
            out.release()
            
            total_time = time.time() - start_proc
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            
            st.success(f"Video processing finished! Saved to {output_path}")
            st.write(f"⏱️ Total time: {total_time:.2f} s | ⚡ Avg FPS: {avg_fps:.1f}")

elif mode == "📊 Method Comparison":
    st.subheader("Algorithm Comparison Matrix")
    
    st.write("Compare Classical Computer Vision (Hough Lines) vs. Curved Lane Fitting (Sliding Window):")
    
    sample_dir = Path(__file__).parent / "test_image"
    sample_files = sorted(list(sample_dir.glob("*.jpg")))
    
    if sample_files:
        selected_file = st.selectbox("Select image for benchmark", sample_files, format_func=lambda x: x.name)
        img = cv2.imread(str(selected_file))
        
        col1, col2 = st.columns(2)
        
        det_hough = HoughLaneDetector()
        det_sw = SlidingWindowDetector()
        
        t0 = time.time()
        res_hough = det_hough.detect(img.copy())
        t_hough = (time.time() - t0) * 1000
        
        t0 = time.time()
        res_sw = det_sw.detect(img.copy())
        t_sw = (time.time() - t0) * 1000
        
        with col1:
            st.markdown("#### 1. Hough Transform Method")
            vis_hough = visualizer.draw_lanes(img.copy(), res_hough.get("left_lane"), res_hough.get("right_lane"))
            vis_hough = visualizer.draw_lane_area(vis_hough, res_hough.get("left_lane"), res_hough.get("right_lane"))
            st.image(cv2.cvtColor(vis_hough, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.json({
                "Latency": f"{t_hough:.2f} ms",
                "Curvature": f"{res_hough.get('curvature', 0):.2f} m",
                "Offset": f"{res_hough.get('offset', 0):.2f} m"
            })
            
        with col2:
            st.markdown("#### 2. Sliding Window Method")
            vis_sw = res_sw.get("visualization", img.copy())
            st.image(cv2.cvtColor(vis_sw, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.json({
                "Latency": f"{t_sw:.2f} ms",
                "Curvature": f"{res_sw.get('curvature', 0):.2f} m",
                "Offset": f"{res_sw.get('offset', 0):.2f} m"
            })

elif mode == "⚙️ Algorithm Config":
    st.subheader("System Parameters & Calibration")
    st.write("Tune edge detection & ROI parameters dynamically:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Canny Low Threshold", 10, 100, config.CANNY_LOW_THRESHOLD)
        st.slider("Canny High Threshold", 50, 250, config.CANNY_HIGH_THRESHOLD)
        st.slider("Hough Threshold", 10, 100, config.HOUGH_THRESHOLD)
    with col2:
        st.slider("Hough Min Line Length", 10, 150, config.HOUGH_MIN_LINE_LENGTH)
        st.slider("Hough Max Line Gap", 5, 100, config.HOUGH_MAX_LINE_GAP)
        st.slider("Confidence Threshold", 0.1, 1.0, config.CONFIDENCE_THRESHOLD)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #718096; font-size: 0.85rem;'>Autonomous Lane Detection System • Built with OpenCV, PyTorch & Streamlit</div>", unsafe_allow_html=True)
