import cv2
from ultralytics import YOLO  # type:ignore
from pytube import YouTube  # type:ignore
import ffmpeg
import sys

# Initialize the YOLO model (use pre-trained weights for fine-tuning)
model = YOLO('yolov8n.pt')

# Function to get the YouTube stream URL
def get_youtube_stream_url(url):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()  # Get the first mp4 stream
    return stream.url

# Function to handle video stream from URL (for other online video sources)
def get_video_stream_url(url):
    try:
        # For any other video source, we can use ffmpeg to handle stream processing
        # Example for streaming, modify according to the source type
        input_stream = ffmpeg.input(url)
        return input_stream
    except Exception as e:
        print(f"Error in streaming video: {e}")
        sys.exit()

# Check if the video source is YouTube or another link
video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"  # Replace with the actual URL
is_youtube = True  # Set to False if not YouTube

# Get the appropriate video stream URL
if is_youtube:
    video_stream_url = get_youtube_stream_url(video_url)
else:
    video_stream_url = get_video_stream_url(video_url)

# Open the video stream (works for YouTube or any video stream URL)
cap = cv2.VideoCapture(video_stream_url)

# Check if the video stream is opened properly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Run object detection and analyze the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break  # Exit when no frame is captured

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Run object detection on the frame
    results = model(frame)

    # Process detection results
    for result in results:
        if result.conf > 0.5 and result.name in ["person", "gun"]:  # Adjust for relevant classes
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates as integers
            label = f"{result.name} {result.conf:.2f}"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Optional alert for weapon detection
            if result.name == "gun":
                print("ALERT! Weapon detected.")

    # Display the frame with bounding boxes and labels
    cv2.imshow("Threat Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
