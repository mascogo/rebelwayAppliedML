
import cv2
import easyocr
import numpy as np
import pandas as pd

# source = r"C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\dev\rebelwayAppliedML\text_detection\Start your next DnD campaign HERE (2160p_60fps_AV1-128kbit_AAC).mp4"
source = r"2796141-uhd_3840_2160_25fps.mp4"
# source = 0
output = "text_detected.avi"
device = "cuda"

cap = cv2.VideoCapture(source)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
reader = easyocr.Reader(["en", "es"], gpu=False)
threshold = 0.5

rows = []
num_frame = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output, fourcc, fps, (width, height))
while cap.isOpened():
    ret, img = cap.read()
    # print("img.shape: {}".format(img.shape))
    print("frame: {}".format(num_frame))
    if ret:
        result = reader.readtext(img, paragraph=False)

        for i, t in enumerate(result):

            bbox, text, score = t
            # print(f"{i}; {text}")
            # for n in range(len(bbox)-1):
            #     cv2.line(img, bbox[n], bbox[n+1], (0, 255, 0), 2)
            # cv2.line(img, bbox[n+1], bbox[0], (0, 255, 0), 2)
            points = np.array(bbox, np.int32)
            cv2.polylines(img, [points], True, (0, 255, 0))

            cv2.putText(img, text, points[0], cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, (255 // score), (255 // (1- score))), 2)  # bottomLeftOrigin=True)
            # cv2.rectangle(img, bbox[0],bbox[2], (0, 255, 0), 2)
            # b = sum(bbox, [])
            # print("b:",b)
            row = [num_frame] + [text]
            rows.append(row)
        cv2.imshow("Text detector", img)
        # print(img)
        out.write(img)


        if cv2.waitKey(1) == 27:  #  ord("q"):
            break

    else:
        break

    num_frame += 1

cap.release()

cv2.destroyAllWindows()
table = pd.DataFrame(rows)
table.to_csv("text_detected.csv", header=False, index=False)
print(table)
