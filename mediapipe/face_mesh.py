import cv2
import numpy as np
import mediapipe as mp

# source = r"C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\dev\rebelwayAppliedML\mediapipe\4275756-uhd_4096_2160_25fps.mp4"
source = r"C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\dev\rebelwayAppliedML\mediapipe\3044691-uhd_3840_2160_24fps.mp4"

cap = cv2.VideoCapture(source)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2, static_image_mode=False)
drawSpec = mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius= 2)

scale_val = 0.7
while cap.isOpened():
    ret, img = cap.read()

    if ret:
        x1 = int(img.shape[1] * scale_val)
        x2 = int(img.shape[0] * scale_val)

        img = cv2.resize(img, (x1, x2))

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = faceMesh.process(imgRGB)

        if detections.multi_face_landmarks:
            for face_landmark in detections.multi_face_landmarks:
                mpDraw.draw_landmarks(img, face_landmark, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

        cv2.imshow("Face Mesh", img)

        if cv2.waitKey(1) == ord("q"):
            break


    else:
        break

cap.release()
cv2.destroyAllWindows()