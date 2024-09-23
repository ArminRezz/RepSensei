import cv2  # OpenCV for image processing
import mediapipe as mp  # MediaPipe for pose detection
import numpy as np  # NumPy for numerical operations
import os  # OS module for interacting with the operating system
import argparse  # Argument parser for command line options

class PushUpCounter:
    def __init__(self):
        # Initialize MediaPipe Pose with confidence thresholds
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.counter = 0  # Initialize push-up counter
        self.stage = None  # Initialize stage variable to track push-up position

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)  # Convert first point to NumPy array
        b = np.array(b)  # Convert middle point to NumPy array
        c = np.array(c)  # Convert end point to NumPy array

        # Calculate the angle in radians and convert to degrees
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        # Ensure the angle is between 0 and 360 degrees
        if angle > 180.0:
            angle = 360 - angle

        return angle  # Return the calculated angle

    def process_frame(self, image):
        """Process a single video frame for push-up detection."""
        # Convert the BGR image to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)  # Process the image to detect pose landmarks

        # Extract landmarks if available
        try:
            landmarks = results.pose_landmarks.landmark  # Access pose landmarks

            # Get coordinates for right shoulder, elbow, and wrist
            shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate angle at the elbow
            angle = self.calculate_angle(shoulder, elbow, wrist)

            # Visualize the calculated angle on the frame
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Push-up counter logic based on elbow angle
            if angle > 160:  # If elbow is straight
                self.stage = "up"  # Set stage to 'up'
            if angle < 90 and self.stage == 'up':  # If elbow is bent and stage was 'up'
                self.stage = "down"  # Set stage to 'down'
                self.counter += 1  # Increment push-up counter
                print(f"Push-ups: {self.counter}")  # Print current count

            # Render push-up counter on the frame
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)  # Draw background rectangle
            cv2.putText(image, 'Push-ups', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render current stage on the frame
            cv2.putText(image, 'Stage', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, self.stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error processing frame: {e}")  # Print error if frame processing fails

        return image  # Return the processed image

def list_videos(directory):
    """List all video files in the given directory."""
    supported_formats = ('.mp4')  # Define supported video formats
    videos = [f for f in os.listdir(directory) if f.lower().endswith(supported_formats)]  # List videos
    return videos  # Return the list of video files

def main():
    """Main function to run the push-up counter with video selection."""
    parser = argparse.ArgumentParser(description="Push-Up Counter using Video File")  # Create argument parser
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help='Directory containing video files')  # Add directory argument
    args = parser.parse_args()  # Parse the arguments

    video_dir = args.directory  # Get the directory from arguments

    # Check if the specified directory exists
    if not os.path.isdir(video_dir):
        print(f"Error: The directory '{video_dir}' does not exist.")
        return

    videos = list_videos(video_dir)  # List video files in the directory

    # Check if any videos were found
    if not videos:
        print(f"No video files found in directory '{video_dir}'.")
        return

    print("Available videos:")  # Print available videos
    for idx, video in enumerate(videos, start=1):
        print(f"{idx}. {video}")  # Display video options

    # Prompt user to select a video
    try:
        choice = int(input(f"Select a video to process (1-{len(videos)}): "))
        if choice < 1 or choice > len(videos):
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    selected_video = os.path.join(video_dir, videos[choice - 1])  # Get the selected video path
    print(f"Processing video: {selected_video}")  # Print the selected video

    cap = cv2.VideoCapture(selected_video)  # Open the video file
    pushup_counter = PushUpCounter()  # Create a PushUpCounter instance

    # Process video frames until the end of the video
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("End of video file reached.")  # Print message if video ends
            break

        # Process the frame and get the output
        output_frame = pushup_counter.process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Push-Up Counter', output_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()  # Run the main function