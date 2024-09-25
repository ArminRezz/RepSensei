import cv2
import mediapipe as mp
import numpy as np

class TricepExtensionCounter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
        self.counter = 0
        self.is_extending = False

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        elbow_angles = []

        if results.pose_landmarks:
            side = 'RIGHT'
            shoulder = results.pose_landmarks.landmark[getattr(self.mp_pose.PoseLandmark, f"{side}_SHOULDER").value]
            elbow = results.pose_landmarks.landmark[getattr(self.mp_pose.PoseLandmark, f"{side}_ELBOW").value]
            wrist = results.pose_landmarks.landmark[getattr(self.mp_pose.PoseLandmark, f"{side}_WRIST").value]

            shoulder_pt = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
            elbow_pt = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
            wrist_pt = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))

            angle = self.calculate_angle(shoulder_pt, elbow_pt, wrist_pt)
            elbow_angles.append(angle)

            if angle < 60:
                if not self.is_extending:
                    self.is_extending = True
            else:
                if self.is_extending:
                    self.counter += 1
                    self.is_extending = False

            cv2.line(image, shoulder_pt, elbow_pt, (255, 0, 0), 2)
            cv2.line(image, elbow_pt, wrist_pt, (0, 255, 0), 2)
            cv2.putText(image, f"{side} Elbow Angle: {int(angle)}°", (elbow_pt[0] + 10, elbow_pt[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(image, elbow_pt, 8, (0, 0, 255), -1)

            if elbow_angles:
                average_angle = sum(elbow_angles)
                cv2.putText(image, f"Average Elbow Angle: {int(average_angle)}°", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Tricep Extensions: {self.counter}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return image

def main():
    video_path = 'mp4s/tricepcables.mp4'
    cap = cv2.VideoCapture(video_path)
    tricep_counter = TricepExtensionCounter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = tricep_counter.process_frame(frame)
        cv2.imshow('Arm Pose Detection', output_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()