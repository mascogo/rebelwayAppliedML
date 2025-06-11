import cv2
import numpy as np
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

source = 0   # Webcam

cap = cv2.VideoCapture(source)

with mp_pose.Pose(min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if ret:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if results:
                mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

            cv2.imshow("Pose", image)

            if cv2.waitKey(5) == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()

