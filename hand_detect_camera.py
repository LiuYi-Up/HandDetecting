import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# cap = cv2.VideoCapture('F:\LIUYI\Python_pro\Hand_in_Video\\asl.mp4')
# cap = cv2.VideoCapture(0)  # VideoCapture()中参数是0，表示打开笔记本的内置摄像头,1为外接USB摄像头
#######################################
fig = plt.figure()
ax = Axes3D(fig)
###################################
# while cap.isOpened():  # 返回True表示摄像头打开成功，若返回False表示摄像头打开失败。
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)
while True:
    # success,image是获cap.read()方法的两个返回值。
    # 其中success是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
    # frame就是每一帧的图像，是个三维矩阵。
    # success, image = cap.read()
    # if not success:
    #     print("Ignoring empty camera frame.")

        # If loading a video, use 'break' instead of 'continue'
        # continue

    # plt.ion()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    if not color_frame:
        continue
    image = np.asanyarray(color_frame.get_data())

    # Flip the image horizontally for a later self-view display, and convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeabe to pass by reference
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
#################################################################################
    #         connections = mp_hands.HAND_CONNECTIONS
    #         plt.cla()
    #         for connection in connections:
    #             start_idx = connection[0]
    #             end_idx = connection[1]
    #             x = [hand_landmarks.landmark[start_idx].x, hand_landmarks.landmark[end_idx].x]
    #             y = [hand_landmarks.landmark[start_idx].y, hand_landmarks.landmark[end_idx].y]
    #             z = [abs(hand_landmarks.landmark[start_idx].z), abs(hand_landmarks.landmark[end_idx].z)]

    #             ax.plot3D(x, y, z, 'om-')
    #         plt.pause(0.00001)
    # plt.ioff()
    # # plt.show()
################################################################################
    cv2.imshow('Mediapipe Hands', image)
    # plt.show()

    if cv2.waitKey(5) & 0xFF == 27:
        break
# plt.show()
hands.close()
cap.release()
