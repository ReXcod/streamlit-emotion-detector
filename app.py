import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit App Title
st.title("Real-Time Emotion Detection")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            if len(analysis) > 0:
                emotion = analysis[0]['dominant_emotion']
                cv2.putText(img, f"Emotion: {emotion}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error: {e}")

        return img

# Activate webcam and process video
webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)
