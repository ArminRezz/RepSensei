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

# Get all video files in the folder
# List comprehension to filter and collect all .mp4 files in the specified folder
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

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

# Process each video file
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)  # Construct the full path to the video file
    cap = cv2.VideoCapture(video_path)  # Open the video file for processing

    # Loop to read frames from the video
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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  # Shoulder coordinates
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]  # Elbow coordinates
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]  # Wrist coordinates

            # Calculate the angle at the elbow using shoulder, elbow, and wrist coordinates
            angle = calculate_angle(shoulder, elbow, wrist)

            # Provide feedback based on the calculated angle
            if angle > 160:  # If the angle is too large
                cv2.putText(frame, "Tip: Lower your body more!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif angle < 140:  # If the angle is too small
                cv2.putText(frame, "Tip: Keep your elbows close!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the processed frame with landmarks and feedback
        cv2.imshow('Push-up Tracker', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture object after processing
    cap.release()

# Close all OpenCV windows after processing is complete
cv2.destroyAllWindows()