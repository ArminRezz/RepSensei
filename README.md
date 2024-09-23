#  RepSensei 🥋 

This project uses OpenCV and MediaPipe to count push-ups from a video file. It processes video frames to detect pose landmarks and calculates the angle at the elbow to determine the push-up count.

## Features 🏋️

- Detects push-ups using pose landmarks.
- Counts push-ups based on elbow angle.
- Displays the push-up count and current stage on the video frame.

## Requirements

- OpenCV
- MediaPipe
- NumPy

## Installation 💪 

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/push-up-counter.git
    cd push-up-counter
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python mediapipe numpy
    ```

## Usage

1. Place your video files in a directory.

2. Run the script with the directory containing the video files:
    ```sh
    python sensei.py -d /path/to/your/video/directory
    ```
