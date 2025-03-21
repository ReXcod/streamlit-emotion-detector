import streamlit as st
import cv2
import numpy as np
import av
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Fix OpenCV issues in cloud environments
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

st.title("🎭 Real-time Emotion Detection with DeepFace")

# Define a video processor class for WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            if len(analysis) > 0:
                emotion = analysis[0]['dominant_emotion']
                cv2.putText(img, f"Emotion: {emotion}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam and process frames
webrtc_streamer(key="emotion-detection", video_processor_factory=VideoProcessor)
