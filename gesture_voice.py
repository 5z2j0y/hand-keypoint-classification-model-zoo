import os
import csv
import copy
import itertools
import threading
import pygame.mixer

import cv2
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

from utils.draw_hand import draw_bounding_rect, draw_landmarks, draw_info_text  

def init_pygame_mixer():
    pygame.mixer.init()
    sounds = {}
    sound_files = {
        0: 'voices/backward.mp3',
        1: 'voices/forward.mp3',
        2: 'voices/right.mp3',
        3: 'voices/left.mp3',
        4: 'voices/speedup.mp3',
        5: 'voices/speeddown.mp3'
    }
    for id, path in sound_files.items():
        if os.path.exists(path):
            sounds[id] = pygame.mixer.Sound(path)
    return sounds

def play_sound(hand_sign_id):
    try:
        if hand_sign_id in sounds and not pygame.mixer.get_busy():
            sounds[hand_sign_id].play()
    except Exception as e:
        print(f"播放声音时出错: {e}")

def main():
    cap_device = 1
    cap_width = 640
    cap_height = 480

    use_static_image_mode = 'store_true'
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    model_name = 'AladdinT'

    use_brect = not 'store_true'

    # 准备摄像头
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 加载模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    model_path = os.path.join('model', model_name, 'keypoint_classifier.tflite')
    keypoint_classifier = KeyPointClassifier(model_path=model_path)

    # 读取标签
    label_path = os.path.join('model', model_name, 'keypoint_classifier_label.csv')
    with open(label_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # 初始化声音
    global sounds
    sounds = init_pygame_mixer()

    # FPS计算模块
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cvFpsCalc.get()

        # 按键处理(ESC：退出)
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):  # ESC 或 q
            break

        # 摄像头捕获
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # 镜像显示
        debug_image = copy.deepcopy(image)

        # 进行检测
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # 关键点分类
        brect = None
        landmark_list = None
        handedness = None
        hand_sign_id = 0
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 计算外接矩形
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 计算关键点
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 转换为相对坐标和归一化坐标
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # 关键点分类
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # 播放声音
                threading.Thread(target=play_sound, args=(hand_sign_id,)).start()

        # 绘图
        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        debug_image = draw_landmarks(debug_image, landmark_list)
        debug_image = draw_info_text(
            debug_image,
            model_name,
            brect,
            handedness,
            keypoint_classifier_labels[hand_sign_id],
            fps,
        )

        cv2.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv2.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # 关键点
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 转换为相对坐标
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 转换为一维列表
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 归一化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 转换为相对坐标
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # 转换为一维列表
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

if __name__ == '__main__':
    main()