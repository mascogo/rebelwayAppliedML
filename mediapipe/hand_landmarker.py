import os
import cv2
import numpy as np
import mediapipe as mp

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose
mp_handpose = mp.solutions.hands

# source = 0   # Webcam

source = r"C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\myDataSets\hands"

# cap = cv2.VideoCapture(source)

with mp_handpose.Hands(max_num_hands=1, min_detection_confidence=0.45, min_tracking_confidence=0.6) as handpose:
    # while cap.isOpened():
    for f in os.listdir(source):
        # ret, image = cap.read()
        image_path = os.path.join(source, f)
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if image.any():
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = handpose.process(image)
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if results and results.multi_hand_landmarks:
                print("__multi_hand_landmarks__ ({}):\n{}\n-----------------".format(len(results.multi_hand_landmarks), results.multi_hand_landmarks))
                # print(
                #     "__multi_hand_world_landmarks__:\n{}\n-----------------".format(results.multi_hand_world_landmarks))
                # print("__multi_handedness__:\n{}\n-----------------".format(results.multi_handedness))


                mp_draw.draw_landmarks(image, results.multi_hand_landmarks[0], mp_handpose.HAND_CONNECTIONS, landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

            cv2.imshow("Pose", image)
            save_folder = os.path.join(source, "handpose")
            os.makedirs(save_folder, exist_ok=True)
            f_name, f_ext = os.path.splitext(f)
            save_path = os.path.join(save_folder,  "handpose_{}{}".format(f_name, f_ext))
            cv2.imwrite(save_path, image)
            # if cv2.waitKey(5) == ord('q'):
            key = cv2.waitKey(5)
            if key == 27:  # escape key
                break
        else:
            break

# cap.release()
cv2.destroyAllWindows()

