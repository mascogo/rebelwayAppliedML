import cv2
import numpy as np
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# source = 0   # Webcam
# source = r"C:\Mask&Magic\201702_LosIrvins_Y_El_Secreto_Del_Ultimo_Rey\io\finales\A007_C007_comp_1_v37_HD_G22_h264.mov"
# source = r"C:\Mask&Magic\201702_LosIrvins_Y_El_Secreto_Del_Ultimo_Rey\io\finales\A007_C007_comp_2_v38_HD_G22_h264.mov"
# source = r"C:\Mask&Magic\201702_LosIrvins_Y_El_Secreto_Del_Ultimo_Rey\io\finales\A012_C005_comp_v02_HD_G22_h264.mov"
# source = r"C:\Mask&Magic\201805_ESLAC_SinAliento_teaser_vfx\ref\ProRes 4444 16bpc 4k_1.mp4"
# source = r"C:\Mask&Magic\201805_ESLAC_SinAliento_teaser_vfx\ref\v6.mp4"
# source = r"C:\Mask&Magic\201702_LosIrvins_Y_El_Secreto_Del_Ultimo_Rey\vfx\A007_C014_0702D2\video\A007_C014_comp_v01_h264.mov"
source = r"C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\myDataSets\Obi Wan Kenobi trains Anakin Skywalker at the Jedi Temple (All Scenes) (540p_24fps_H264-128kbit_AAC).mp4"
cap = cv2.VideoCapture(source)

with mp_pose.Pose(min_tracking_confidence=0.95 ) as pose:
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

            # if cv2.waitKey(5) == ord('q'):
            key = cv2.waitKey(5)
            if key == 27:  # escape key
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()

