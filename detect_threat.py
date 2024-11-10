import cv2
import numpy as np
import tensorflow as tf
import requests
import ffmpeg

# Hyperparameters
image_size = (224, 224)  # Image size for input to the model
sequence_length = 30  # Number of frames per video clip

# Step 1: Function to get video stream (Works for YouTube and other video URLs)
def get_video_stream_url(url):
    try:
        # If the video is an online stream (e.g., mp4, rtsp, m3u8)
        # Use OpenCV to directly capture the stream
        return url
    except Exception as e:
        print(f"Error while processing video URL: {e}")
        return None

# Step 2: Extract frames from the video stream URL
def video_to_frames(video_stream_url, sequence_length, image_size):
    cap = cv2.VideoCapture(video_stream_url)
    frames = []
    count = 0
    while cap.isOpened() and count < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, image_size)
        frames.append(frame)
        count += 1
    cap.release()
    if len(frames) < sequence_length:
        frames += [np.zeros_like(frames[0])] * (sequence_length - len(frames))  # Pad with zeros if video is shorter
    return np.array(frames)

# Step 3: Threat detection function using your fine-tuned model
def detect_threat(model, video_stream_url):
    frames = video_to_frames(video_stream_url, sequence_length, image_size)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)

    if prediction > 0.5:  # If prediction > 0.5, classify as threat
        print("ALERT! Threat detected!")
    else:
        print("No threat detected.")

# Main logic to handle video stream and threat detection
if __name__ == "__main__":
    video_url = "https://path_to_any_video.mp4"  # Replace with actual video URL or stream

    # Get video stream URL (works for any valid video URL or stream)
    video_stream_url = get_video_stream_url(video_url)

    # Load your fine-tuned model
    model = tf.keras.models.load_model('fine_tuned_action_recognition_model.h5')

    # Detect threat in the video stream
    detect_threat(model, video_stream_url)
