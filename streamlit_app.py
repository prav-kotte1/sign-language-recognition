import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

# Load model
model = tf.keras.models.load_model("model.keras")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

st.title("Sign Language Recognition")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"]
)

def extract_frames(video_path, max_frames=30):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while len(frames) < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((224,224,3)))

    return np.array(frames)

if uploaded_video:

    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:

        tmp.write(uploaded_video.read())

        temp_video_path = tmp.name

    frames = extract_frames(temp_video_path)

    frames = np.expand_dims(frames, axis=0)

    prediction = model.predict(frames)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(
        f"Prediction: {labels[predicted_class]}"
    )

    st.write(f"Confidence: {confidence:.2f}")
