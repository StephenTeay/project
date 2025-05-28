import streamlit as st
import os
import threading
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import logging
import tempfile
import av

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Cloud compatible paths
CACHE_DIR = Path.home() / ".cache" / "object_detector"
LOG_DIR = Path.home() / ".cache" / "logs"

# Thread-safe statistics storage
thread_stats_lock = threading.Lock()
thread_label_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "max_score": 0.0})

# Create directories
for dir_path in [CACHE_DIR, LOG_DIR]:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create {dir_path}: {e}")

# Set cache environment variables
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(CACHE_DIR)

LOG_FILE = LOG_DIR / "object_logs.txt"

# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'label_stats' not in st.session_state:
    st.session_state.label_stats = {}

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load the object detection model"""
    try:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        model.eval()
        return model, weights
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# Load model
with st.spinner("🤖 Loading AI model... This may take a moment on first run."):
    model, weights = load_model()

if model is None:
    st.error("❌ Model loading failed. Please refresh the page.")
    st.stop()

categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def detect_objects_in_image(image):
    """Detect objects in a single image"""
    try:
        # Preprocess image
        img_processed = img_preprocess(image)
        
        # Run inference
        with torch.no_grad():
            prediction = model(img_processed.unsqueeze(0))[0]
        
        labels = [categories[label] for label in prediction["labels"]]
        scores = prediction["scores"].detach().cpu().numpy()
        boxes = prediction["boxes"].detach().cpu().numpy()
        
        return {
            "labels": labels,
            "scores": scores,
            "boxes": boxes
        }
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return {"labels": [], "scores": [], "boxes": []}

def draw_boxes_on_image(image, prediction):
    """Draw bounding boxes on image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        # Try to use a better font
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    for i, (label, score, box) in enumerate(zip(
        prediction["labels"], 
        prediction["scores"], 
        prediction["boxes"]
    )):
        if score > 0.5:  # Only show high confidence detections
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            text = f"{label}: {score:.1%}"
            if font:
                bbox = draw.textbbox((x1, y1-25), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1-25), text, fill="white", font=font)
            else:
                draw.text((x1, y1-20), text, fill=color)
    
    return img_draw

def update_statistics(prediction):
    """Update detection statistics"""
    for label, score in zip(prediction["labels"], prediction["scores"]):
        if float(score) > 0.5:  # Only count high-confidence detections
            if label not in st.session_state.label_stats:
                st.session_state.label_stats[label] = {
                    "count": 0,
                    "total_score": 0.0,
                    "max_score": 0.0
                }
            st.session_state.label_stats[label]["count"] += 1
            st.session_state.label_stats[label]["total_score"] += float(score)
            st.session_state.label_stats[label]["max_score"] = max(
                st.session_state.label_stats[label]["max_score"], 
                float(score)
            )

def update_thread_statistics(prediction):
    """Update detection statistics in thread-safe manner"""
    global thread_label_stats, thread_stats_lock
    
    with thread_stats_lock:
        for label, score in zip(prediction["labels"], prediction["scores"]):
            if float(score) > 0.5:  # Only count high-confidence detections
                thread_label_stats[label]["count"] += 1
                thread_label_stats[label]["total_score"] += float(score)
                thread_label_stats[label]["max_score"] = max(
                    thread_label_stats[label]["max_score"], 
                    float(score)
                )

def sync_thread_stats_to_session():
    """Sync thread statistics to session state"""
    global thread_label_stats, thread_stats_lock
    
    with thread_stats_lock:
        for label, data in thread_label_stats.items():
            if label not in st.session_state.label_stats:
                st.session_state.label_stats[label] = {
                    "count": 0,
                    "total_score": 0.0,
                    "max_score": 0.0
                }
            
            # Add thread stats to session stats
            st.session_state.label_stats[label]["count"] += data["count"]
            st.session_state.label_stats[label]["total_score"] += data["total_score"]
            st.session_state.label_stats[label]["max_score"] = max(
                st.session_state.label_stats[label]["max_score"],
                data["max_score"]
            )
        
        # Clear thread stats after syncing
        thread_label_stats.clear()

def log_predictions():
    """Log prediction statistics to file"""
    try:
        if not st.session_state.label_stats:
            st.warning("No statistics to log")
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(LOG_FILE, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Detection Log - {timestamp}\n")
            f.write(f"{'='*50}\n")
            f.write(f"{'Object':<20} {'Count':<8} {'Avg Conf':<10} {'Max Conf':<10}\n")
            f.write(f"{'-'*50}\n")
            
            for label, data in sorted(st.session_state.label_stats.items()):
                avg_conf = data["total_score"] / data["count"]
                f.write(f"{label.title():<20} {data['count']:<8} {avg_conf:<10.1%} {data['max_score']:<10.1%}\n")
        
        st.success("✅ Statistics logged successfully!")
        logger.info("Statistics logged successfully")
    except Exception as e:
        error_msg = f"Logging failed: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)

# Enhanced RTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
        {"urls": ["stun:openrelay.metered.ca:80"]},
        {
            "urls": ["turn:relay1.expressturn.com:3480"],
            "username": "000000002063793301",
            "credential": "7Dfdi1EmR3bo6lc4JTfMqkNCqyI="
        }
    ],
    "iceCandidatePoolSize": 10,
    "iceTransportPolicy": "all"
})

def video_frame_callback(frame):
    """Process video frames and return modified frame with detections"""
    try:
        # Convert av.VideoFrame to PIL Image
        img = frame.to_image()
        
        # Run object detection
        prediction = detect_objects_in_image(img)
        
        # Update thread-safe statistics
        update_thread_statistics(prediction)
        
        # Draw bounding boxes on the image
        if prediction["labels"]:
            img_with_boxes = draw_boxes_on_image(img, prediction)
        else:
            img_with_boxes = img
        
        # Convert back to av.VideoFrame
        new_frame = av.VideoFrame.from_image(img_with_boxes)
        
        return new_frame
        
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return frame  # Return original frame if processing fails

# Alternative video processor class for better control
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.detection_interval = 5  # Process every 5th frame for performance
        
    def recv(self, frame):
        """Process incoming video frame"""
        try:
            self.frame_count += 1
            
            # Only process every nth frame to improve performance
            if self.frame_count % self.detection_interval == 0:
                # Convert to PIL Image
                img = frame.to_image()
                
                # Run detection
                prediction = detect_objects_in_image(img)
                
                # Update statistics
                update_thread_statistics(prediction)
                
                # Draw bounding boxes
                if prediction["labels"]:
                    img_with_boxes = draw_boxes_on_image(img, prediction)
                    # Convert back to frame
                    return av.VideoFrame.from_image(img_with_boxes)
            
            return frame
            
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return frame

# Main UI
st.title("🎥 AI Object Detection System")
st.markdown("*Powered by PyTorch Faster R-CNN*")

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Mode selection
    detection_mode = st.radio(
        "Detection Mode:",
        ["📤 Upload Images", "📹 Live Camera"],
        help="Upload mode works everywhere. Camera mode may have limitations on some platforms."
    )
    
    st.divider()
    
    # Video processing options (only show for camera mode)
    if detection_mode == "📹 Live Camera":
        st.subheader("⚙️ Video Settings")
        processing_mode = st.selectbox(
            "Processing Mode:",
            ["Standard", "Performance"],
            help="Performance mode processes fewer frames for better speed"
        )
    
    st.divider()
    
    # Statistics controls
    st.subheader("📊 Statistics")
    
    if st.button("💾 Save Log", use_container_width=True):
        log_predictions()
    
    if st.button("🗑️ Clear Stats", use_container_width=True):
        st.session_state.label_stats = {}
        st.rerun()
    
    if st.button("📜 View History", use_container_width=True):
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                log_content = f.read()
            if log_content.strip():
                st.text_area("Log History", log_content, height=300)
            else:
                st.info("No logs yet")
        else:
            st.info("No log file found")
    
    st.divider()
    
    # Model info
    st.subheader("ℹ️ Model Info")
    st.write(f"**Objects:** {len(categories)}")
    st.write(f"**Confidence:** 50%+")
    st.write(f"**Model:** Faster R-CNN")

# Main content area
if detection_mode == "📤 Upload Images":
    st.header("📤 Upload Image Detection")
    
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more images for object detection"
    )
    
    if uploaded_files:
        # Process each uploaded file
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"🖼️ Results for: {uploaded_file.name}")
            
            # Load image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Create columns for before/after
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image**")
                    st.image(image, use_container_width=True)
                
                # Detect objects
                with st.spinner("🔍 Detecting objects..."):
                    prediction = detect_objects_in_image(image)
                
                # Update statistics
                update_statistics(prediction)
                
                with col2:
                    st.write("**Detection Results**")
                    if prediction["labels"]:
                        # Draw bounding boxes
                        img_with_boxes = draw_boxes_on_image(image, prediction)
                        st.image(img_with_boxes, use_container_width=True)
                    else:
                        st.image(image, use_container_width=True)
                        st.info("No objects detected with high confidence")
                
                # Show detection details
                if prediction["labels"]:
                    high_conf_detections = [
                        (label, score) for label, score in 
                        zip(prediction["labels"], prediction["scores"]) 
                        if score > 0.5
                    ]
                    
                    if high_conf_detections:
                        st.write("**🎯 Detected Objects:**")
                        detection_data = []
                        for label, score in high_conf_detections:
                            detection_data.append({
                                "Object": label.title(),
                                "Confidence": f"{score:.1%}"
                            })
                        
                        df = pd.DataFrame(detection_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                if i < len(uploaded_files) - 1:  # Add separator between images
                    st.divider()
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

elif detection_mode == "📹 Live Camera":
    st.header("📹 Live Camera Detection")
    
    # Warning about camera limitations
    st.warning("⚠️ **Note:** Camera functionality may be limited on some cloud platforms. If it doesn't work, try Upload mode instead.")
    
    try:
        # Choose processor based on mode
        if 'processing_mode' in locals() and processing_mode == "Performance":
            processor = VideoProcessor()
            
            ctx = webrtc_streamer(
                key="object-detector-performance",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: processor,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 320, "ideal": 480, "max": 640},
                        "height": {"min": 240, "ideal": 360, "max": 480},
                        "frameRate": {"min": 5, "ideal": 8, "max": 12}
                    }, 
                    "audio": False
                },
                async_processing=True
            )
        else:
            ctx = webrtc_streamer(
                key="object-detector-standard",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 320, "ideal": 640, "max": 1280},
                        "height": {"min": 240, "ideal": 480, "max": 720},
                        "frameRate": {"min": 5, "ideal": 10, "max": 15}
                    }, 
                    "audio": False
                },
                async_processing=True
            )
        
        # Connection status
        status_placeholder = st.empty()
        
        if ctx.state.playing:
            status_placeholder.success("✅ Camera active - AI detection running in real-time!")
            
            # Real-time stats display
            stats_placeholder = st.empty()
            
            # Add sync button when camera is active
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Sync Stats", help="Sync detection statistics from video stream"):
                    sync_thread_stats_to_session()
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop & Save", help="Stop camera and save current statistics"):
                    sync_thread_stats_to_session()
                    if st.session_state.label_stats:
                        log_predictions()
                    st.rerun()
                
        elif ctx.state.signalling:
            status_placeholder.info("🔄 Connecting to camera... Please wait.")
        else:
            status_placeholder.info("📷 Click **START** to begin live detection")
            
    except Exception as e:
        st.error(f"❌ Camera initialization failed: {str(e)}")
        st.info("💡 **Suggestion:** Try refreshing the page or use Upload mode instead.")

# Statistics Display
if st.session_state.label_stats:
    st.header("📊 Detection Statistics")
    
    # Create statistics dataframe
    stats_data = []
    for label, data in st.session_state.label_stats.items():
        avg_confidence = data["total_score"] / data["count"]
        stats_data.append({
            "Object": label.title(),
            "Count": data["count"],
            "Avg Confidence": f"{avg_confidence:.1%}",
            "Max Confidence": f"{data['max_score']:.1%}"
        })
    
    # Sort by count
    stats_df = pd.DataFrame(stats_data).sort_values("Count", ascending=False)
    
    # Display as chart and table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        st.bar_chart(stats_df.set_index("Object")["Count"])
    
    with col2:
        # Statistics table
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Summary metrics
    total_objects = sum(data["count"] for data in st.session_state.label_stats.values())
    unique_objects = len(st.session_state.label_stats)
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Total Detections", total_objects)
    with metric_col2:
        st.metric("Unique Objects", unique_objects)

else:
    st.info("📈 **Statistics will appear here** after detecting objects in images or video")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 <strong>AI Object Detection</strong> | Built with Streamlit & PyTorch</p>
        <p><small>Detects 80+ object categories with state-of-the-art accuracy</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
