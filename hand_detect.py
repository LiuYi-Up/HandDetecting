import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


rhd_seq = (0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

HAND_NAME = "/home/lab/Python_pro/Ly_pro/Dataset/RHD_v1-1/RHD_published_v2/training/color/"
file_list = [os.path.join(HAND_NAME, "00051.png") ]
output_idx=51
#  file_list = [os.path.join(HAND_NAME, "000{}.png".format(i)) for i in range(1, 2, 1)]


for idx, file in enumerate(file_list):

    # Read an image, flip it around y-axis for correct handedness output(see above)
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw han landmarks on the image
    if not results.multi_hand_landmarks:
        continue

    image_hight, image_width, _ = image.shape

    for hand_landmarks in results.multi_hand_landmarks:

        # connections = mp_hands.HAND_CONNECTIONS
        # mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
#  输出像素坐标

        uv_loc = list()
        for i in rhd_seq:
            px = hand_landmarks.landmark[i].x
            py =  hand_landmarks.landmark[i].y
            keypoin = mp_drawing._normalized_to_pixel_coordinates(px, py, image_hight,image_width)
            uv_loc.append(list((image_width-keypoin[0], keypoin[1])))
        # print(uv_loc)

hands.close()

