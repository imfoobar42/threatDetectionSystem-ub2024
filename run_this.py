import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load pre-trained YOLOv5 model (you can also use a custom-trained model if you have one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Change to your custom-trained model if needed

# Define a list of threat labels (we can add more labels as needed)
threat_labels = ['knife', 'gun', 'person']  # Customize based on your threat categories

# Open the video file
video_file = 'test-larger.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

frame_count = 0
threat_detected = False  # Flag to track if a threat is detected

while True:
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Process the frame with the YOLOv5 model
    results = model(frame)
    
    # Display the results (optional, you can comment this out if not needed)
    results.show()  # This will display the frame with bounding boxes
    
    # Extract prediction data (detection results)
    predictions = results.pandas().xywh[0]  # Getting predictions from YOLOv5 results
    
    # Iterate through predictions and check for threats
    for index, row in predictions.iterrows():
        label = row['name']  # Object detected
        confidence = row['confidence']  # Confidence score of the prediction
        
        # Check if the detected object is a threat and confidence is high enough
        if label in threat_labels and confidence > 0.5:  # You can adjust the confidence threshold
            print(f"Threat detected: {label} with confidence {confidence:.2f}")
            threat_detected = True
            break
    
    # Stop processing when a threat is detected
    if threat_detected:
        print("Threat detected in the video!")
        break
    
    frame_count += 1

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

