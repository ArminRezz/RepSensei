# Import necessary libraries for video processing and pose estimation
import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # MediaPipe for pose detection
import numpy as np  # NumPy for numerical operations
import os  # OS module for interacting with the operating system

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose  # Access the pose solution from MediaPipe
pose = mp_pose.Pose()  # Create a Pose object for pose estimation

# Path to the folder containing mp4 videos
video_folder = 'mp4s'  # Specify the folder where video files are stored
video_file = 'pushups1.mp4'  # Specify the video file you want to load
video_path = os.path.join(video_folder, video_file)  # Construct the full path to the video file
cap = cv2.VideoCapture(video_path)  # Open the video file for processing

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # Convert the first point to a NumPy array
    b = np.array(b)  # Convert the mid point to a NumPy array
    c = np.array(c)  # Convert the end point to a NumPy array
    # Calculate the angle in radians using arctangent
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # Convert radians to degrees and ensure the angle is between 0 and 360
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle  # Return the calculated angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose  # Access the pose solution from MediaPipe
pose = mp_pose.Pose()  # Create a Pose object for pose estimation

# Initialize variables for push-up counting and position tracking
push_up_count = 0  # Counter for push-ups
position = 0  # Variable to track the position of the push-up

# Process the selected video file
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is returned, exit the loop
        break

    # Convert the frame to RGB format for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)  # Process the image to detect pose landmarks

    # Draw landmarks if pose landmarks are detected
    if results.pose_landmarks:
        # Draw the detected pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get coordinates of key points (shoulder, elbow, wrist)
        landmarks = results.pose_landmarks.landmark  # Access the landmarks
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y  # Shoulder y-coordinate
        wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y  # Wrist y-coordinate

        # Calculate the position of the push-up
        position = shoulder_y - wrist_y  # Determine the position based on shoulder and wrist

        # Update push-up count based on position
        if position < 0.1:  # Threshold for bottom position
            push_up_count += 1  # Increment counter when reaching the bottom
        # Draw a bar indicating the push-up position
        cv2.rectangle(frame, (10, 50), (30, 50 + int(position * 200)), (255, 0, 0), -1)  # Draw a blue bar

    # Display the processed frame with landmarks and push-up count
    cv2.putText(frame, f'Push-ups: {push_up_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Push-up Tracker', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# Release the video capture object after processing
cap.release()

# Close all OpenCV windows after processing is complete
cv2.destroyAllWindows()