import streamlit as st
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(frame):
    img_processed = img_preprocess(frame)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def draw_predictions(frame, prediction):
    frame_tensor = torch.tensor(frame).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    frame_with_boxes = draw_bounding_boxes(
        frame_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]],
        width=2
    )
    return frame_with_boxes.permute(1, 2, 0).numpy()  # (3, H, W) -> (H, W, 3)

st.title("Real-Time Object Detection")

video_source = st.radio("Choose video source", ("Webcam", "Upload"))
if video_source == "Upload":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        cap = cv2.VideoCapture(video_file.name)
else:
    cap = cv2.VideoCapture(0)  # Use webcam

if cap.isOpened():
    stframe = st.empty()  # Placeholder for video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame for model compatibility
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = make_prediction(frame_rgb)
        frame_with_boxes = draw_predictions(frame_rgb, prediction)

        # Display frame
        stframe.image(frame_with_boxes, channels="RGB", use_column_width=True)
        
    cap.release()
else:
    st.error("Failed to load video.")
