import streamlit as st
import av
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from pathlib import Path

# Set proper cache directories
CACHE_DIR = Path("/tmp/streamlit_cache")
TORCH_CACHE = CACHE_DIR / "torch_cache"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TORCH_CACHE, exist_ok=True)

# Set environment variables
os.environ["STREAMLIT_CACHE"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(TORCH_CACHE)

# Constants
LOG_INTERVAL = 300  # 5 minutes in seconds
LOG_FILE = "object_logs.txt"

# Initialize session state
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = datetime.now()
if 'label_stats' not in st.session_state:
    st.session_state.label_stats = {}

# Model setup
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision', 
                          'fasterrcnn_resnet50_fpn_v2',
                          pretrained=True,
                          cache_dir=str(TORCH_CACHE))  # Add this
    model.eval()
    return model

model = load_model()

def process_frame(frame):
    img = frame.to_image()
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    return {
        "labels": [categories[label] for label in prediction["labels"]],
        "scores": prediction["scores"].detach().cpu().numpy()
    }

def log_predictions():
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_score = sum(v["total_score"] for v in st.session_state.label_stats.values()) or 1  # Avoid division by zero
        
        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{timestamp}]\n")
            f.write("Label\t\tCount\tPercentage\n")
            f.write("-"*40 + "\n")
            for label, data in st.session_state.label_stats.items():
                percentage = (data["total_score"] / total_score * 100)
                f.write(f"{label.title()}\t\t{data['count']}\t{percentage:.1f}%\n")
        
        st.session_state.label_stats = {}
        st.success("Predictions logged successfully!")
    except Exception as e:
        st.error(f"Logging failed: {str(e)}")

# Streamlit UI
st.title("Real-Time Object Detector 🎥")

def video_frame_callback(frame):
    try:
        prediction = process_frame(frame)
        
        # Update statistics
        for label, score in zip(prediction["labels"], prediction["scores"]):
            if label not in st.session_state.label_stats:
                st.session_state.label_stats[label] = {
                    "count": 0,
                    "total_score": 0.0
                }
            st.session_state.label_stats[label]["count"] += 1
            st.session_state.label_stats[label]["total_score"] += float(score)
        
        # Automatic logging
        if (datetime.now() - st.session_state.last_log_time).seconds >= LOG_INTERVAL:
            log_predictions()
            st.session_state.last_log_time = datetime.now()
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    
    return frame

# WebRTC video streamer
ctx = webrtc_streamer(
    key="object-detector",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Display current statistics
if st.session_state.label_stats:
    st.subheader("Current Statistics")
    total_score = sum(v["total_score"] for v in st.session_state.label_stats.values()) or 1
    stats = []
    
    for label, data in st.session_state.label_stats.items():
        percentage = (data["total_score"] / total_score * 100)
        stats.append({
            "Label": label.title(),
            "Count": data["count"],
            "Confidence (%)": f"{percentage:.1f}%"
        })
    
    st.table(pd.DataFrame(stats))

# Manual log display
if st.button("Show Log History"):
    try:
        with open(LOG_FILE, "r") as f:
            st.text(f.read())
    except FileNotFoundError:
        st.warning("No logs available yet")

# Add reset button
if st.button("Reset Statistics"):
    st.session_state.label_stats = {}
    st.rerun()

# Required packages (save as requirements.txt)
"""
streamlit==1.28.0
torch==2.1.0
torchvision==0.16.0
numpy==1.26.0
pandas==2.1.1
Pillow==10.0.0
streamlit-webrtc==0.47.1
av==10.0.0
"""

