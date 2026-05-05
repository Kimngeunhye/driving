#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################################################
# 프로그램명 : hough_drive_c1.py
# 작 성 자 : (주)자이트론
# 생 성 일 : 2020년 07월 23일
# 본 프로그램은 상업 라이센스에 의해 제공되므로 무단 배포 및 상업적 이용을 금합니다.
##########################################################################
import rospy, rospkg, time
import numpy as np
import cv2, math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from std_msgs.msg import Int32MultiArray
from math import *
import signal
import sys
import os

def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

image = np.empty(shape=[0])
bridge = CvBridge()
motor = None
Width = 320
Height = 240
Offset = 160
Gap = 60
cam = False
cam_debug = False
sub_f = 0
time_c = 0
line_count = 0
ultra_msg = None
frame_cnt = 0
prev = "red"


class TrafficLightDetector:
    def __init__(self):
        self.frame_count = 0

    def img_callback(self, data):
        # 매 프레임을 수신할 때마다 이미지 변환
        self.frame_count += 1

        if self.frame_count % 6 == 0:  # 6프레임마다 처리
            signal_color = self.traffic_detect()
            print(f"Detected Traffic Light: {signal_color}")

    def traffic_detect(self, image):
        """
        주어진 이미지에서 초록색 및 빨간색 픽셀을 감지하여 신호등 상태를 예측합니다.
        Args:
            image (numpy.ndarray): 처리할 이미지.
        Returns:
            str: 예측된 신호 상태 ("green", "red", "zero").
        """
        best_green_threshold = 6
        best_red_threshold = 1
        pixel_sum_threshold = 3

        # 초록색의 HSV 색상 범위 설정
        lower_green = np.array([60, 57, 136])
        upper_green = np.array([65, 158, 255])

        # 빨간색의 HSV 색상 범위 설정
        lower_red = np.array([168, 57, 90])
        upper_red = np.array([184, 116, 255])

        # HSV 색 공간으로 변환
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 초록색 및 빨간색 마스크 적용
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # 초록색 및 빨간색 픽셀 수 계산
        green_pixel_count = cv2.countNonZero(green_mask)
        red_pixel_count = cv2.countNonZero(red_mask)
        total_pixel_count = green_pixel_count + red_pixel_count

        # 픽셀 수 출력
        print(f"Green pixel count: {green_pixel_count}")
        print(f"Red pixel count: {red_pixel_count}")
        print(f"Total pixel count: {total_pixel_count}")

        # 신호등 상태 판별
        if total_pixel_count < pixel_sum_threshold:
            return "green"
        elif green_pixel_count >= best_green_threshold:
            return "green"
        elif green_pixel_count < best_green_threshold and red_pixel_count < best_red_threshold:
            return "zero"
        else:
            return "red"


def img_callback(data):
    global image
    global sub_f
    global time_c
    sub_f += 1
    if time.time() - time_c > 1:
        # print("pub fps :", sub_f)
        time_c = time.time()
        sub_f = 0

    image = bridge.imgmsg_to_cv2(data, "bgr8")


def ultra_callback(data):
    global ultra_msg
    ultra_msg = data.data


# publish xycar_motor msg
def drive(Angle, Speed):
    global motor
    motor_msg = xycar_motor()
    motor_msg.angle = Angle
    motor_msg.speed = Speed
    motor.publish(motor_msg)


# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1 + Offset), (x2, y2 + Offset), (0, 255, 0), 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2
    center = int(center)

    cv2.rectangle(img, (lpos - 2, 7 + offset),
                  (lpos + 2, 12 + offset),
                  (0, 0, 0), 2)
    cv2.rectangle(img, (rpos - 2, 7 + offset),
                  (rpos + 2, 12 + offset),
                  (255, 0, 0), 2)
    cv2.rectangle(img, (center - 2, 7 + offset),
                  (center + 2, 12 + offset),
                  (0, 255, 0), 2)
    cv2.rectangle(img, (157, 7 + offset),
                  (162, 12 + offset),
                  (0, 0, 255), 2)
    return img


# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 20

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)

        if low_slope_threshold < abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []
    th = -10

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < Width / 2 - th):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width / 2 + th):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines


# get average m, b of line, sum of x, y, mget lpos, rpos
def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap, cam_debug

    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)

    m = 0
    b = 0

    if size != 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)

        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)

        m = m_sum / size
        b = y_avg - m * x_avg

    if m == 0 and b == 0:
        if left:
            pos = 0
        elif right:
            pos = Width
    else:
        y = Gap / 2

        pos = (y - b) / m

        if cam_debug:
            b += Offset
            xs = (Height - b) / float(m)
            xe = ((Height / 2) - b) / float(m)

            cv2.line(img, (int(xs), int(Height)), (int(xe), int(Height / 2)), (255, 0, 0), 3)

    return img, int(pos)


def stop(all_lines, flag, line_count, stop_time):
    line_len = all_lines
    print("all_lines", line_len)

    if (line_count == 0) and (line_len > 30):
        flag = 1
        line_count = 1
        stop_time = time.time() + 5
        print("Flag up for first time, stop time: ", stop_time)

    elif (line_count == 1) and (line_len > 30):
        # 두 번째 바퀴를 돌고 멈추는 조건
        flag = 0
        line_count = 2
        stop_time = time.time() + 5
        print("Flag up for the second time, stop time: ", stop_time)
    return line_count, flag, stop_time


def process_image(frame):
    global Width
    global Offset, Gap
    global cam, cam_debug, img

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[Offset: Offset + Gap, 0: Width]

    # blur
    kernel_size = 5
    standard_deviation_x = 3  # Kernel standard deviation along X-axis
    blur_gray = cv2.GaussianBlur(roi, (kernel_size, kernel_size), standard_deviation_x)

    # canny edge
    low_threshold = 170
    high_threshold = 200
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold, kernel_size)

    # HoughLinesP
    all_lines = cv2.HoughLinesP(edge_img, 1, math.pi / 180, 30, 30, 2)

    if cam:
        cv2.imshow('calibration', frame)
    # divide left, right lines
    if all_lines is None:
        return (Width) / 2, (Width) / 2, False
    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)

    if cam_debug:
        # draw lines
        frame = draw_lines(frame, left_lines)
        frame = draw_lines(frame, right_lines)
        frame = cv2.line(frame, (115, 117), (205, 117), (0, 255, 255), 2)

        # draw rectangle
        frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
        frame = cv2.rectangle(frame, (0, Offset), (int(Width), Offset + Gap), (255, 202, 204), 2)

    img = frame

    return lpos, rpos, len(all_lines), True


def draw_steer(steer_angle):
    global Width, Height, img

    if img is None or img.size == 0:
        return

    arrow = cv2.imread('/home/pi/xycar_ws/src/study/auto_drive/src/steer_arrow.png')

    if arrow is None:
        print("error:steer_arrow img empty")
        return

    origin_Height = arrow.shape[0]
    origin_Width = arrow.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height / 2
    arrow_Width = (arrow_Height * 462) / 728
    arrow_Height = int(arrow_Height)
    arrow_Width = int(arrow_Width)

    matrix = cv2.getRotationMatrix2D((origin_Width / 2, steer_wheel_center), (-steer_angle) * 1.5, 0.7)
    arrow = cv2.warpAffine(arrow, matrix, (origin_Width + 60, origin_Height))
    arrow = cv2.resize(arrow, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    if Height - arrow_Height < 0 or Width / 2 - arrow_Width / 2 < 0 or Width / 2 + arrow_Width / 2 > Width:
        print("오류: roi 범위가 이미지 크기를 벗어났습니다.")
        return

    arrow_roi = img[arrow_Height: int(Height), int(Width / 2 - arrow_Width / 2): int(Width / 2 + arrow_Width / 2)]
    arrow_roi = cv2.add(arrow, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow)
    img[int(Height - arrow_Height): int(Height),
    int(Width / 2 - arrow_Width / 2): int(Width / 2 + arrow_Width / 2)] = res

    cv2.imshow('steer', img)


def pid_angle(ITerm, error, b_angle, b_error, Cnt):
    angle = 0
    Kp = 0.92  # 0.5 good / if Kp high -> loss decrease+faster respone but incur overshoot
    Ki = 0.00065  # 0.0001 good #0.0002 / if Ki high
    # -> accumulated loss increase faster+faster response but incur overshoot
    Kd = 0.0925  # 1.0 good #2.0/ decrease the vibration
    # if Kd high -> decrease overshoot but when the signal changes rapidly
    # it can make the system destroy
    dt = 1

    PTerm = Kp * error
    ITerm += Ki * error * dt
    derror = error - b_error
    DTerm = Kd * (derror / dt)
    # angle = PTerm + ITerm + DTerm
    angle = PTerm + DTerm

    return angle, ITerm

flag = 0
line_count = 0

def start():
    global motor
    global image
    global Width, Height
    global img
    cam_record = False

    t_check = time.time()
    f_n = 0
    p_angle = 0
    global flag
    global line_count
    avoid_time = time.time() + 3.8
    turn_right = time.time()
    stop_time = time.time()

    b_angle = 0
    b_error = 0
    ITerm = 0
    Opt = 0
    Cnt = 0

    global line_count
    global frame_cnt
    global prev
    speed = 0
    traffic_detector = TrafficLightDetector()

    rospy.init_node('auto_drive')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback)

    image_sub = rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
    print("---------- Xycar C1 HD v1.0 ----------")
    time.sleep(3)

    if cam_record:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        path = '/home/pi/xycar_ws/src/base/cam_record'
        out = cv2.VideoWriter(os.path.join(path, 'test.avi'), fourcc, 25.0, (Width, Height))

    while not rospy.is_shutdown():
        while not image.size == (Width * Height * 3):
            continue

        draw_img = image.copy()

        if frame_cnt % 6 == 0:
            # 신호등 감지
            signal_color = traffic_detector.traffic_detect(draw_img)
            if signal_color == "red":
                print("Red light Detected")
                drive(0, 0)
                speed = 0
                time.sleep(3)
                prev = "red"
                continue
            elif signal_color == "green":
                print("Green Light Detected")
                speed = 35
                if prev == "red":
                    time.sleep(3)
                prev = "green"

        len_all_lines, go = process_image(draw_img)
        line_count, flag, stop_time = stop(len_all_lines, flag, line_count, time.time())

        f_n += 1
        if (time.time() - t_check) > 1:
            # print("fps : ", f_n)
            t_check = time.time()
            f_n = 0
        if cam_record:
            out.write(image)
        draw_img = image.copy()

        try:
            lpos, rpos, len_all_lines, go = process_image(draw_img)
        except:
            lpos, rpos, go = process_image(draw_img)

        if time.time() > stop_time:
            #print("stop_time", stop_time)
            line_count, flag, stop_time = stop(len_all_lines, flag, line_count, stop_time)
            #print("stop_time", stop_time)

        # stop
        if (line_count == 2):  # 라인 카운트 2개시 정지
            # time.sleep(1.5)
            drive(0, -50)
            time.sleep(60)
            cv2.waitKey(0)
            line_count = 0

        lpos, rpos = 120, 200
        diff = rpos - lpos
        center = (lpos + rpos) / 2
        error = (center - Width / 2)
        angle, ITerm = pid_angle(ITerm, error, b_angle, b_error, Cnt)

        if diff > 135 and diff < 142:
            #print("straight")
            pass
        else:
            #print("curve")
            pass

        if (lpos == 0):
            lpos = rpos - 130
        if (rpos > lpos + 145):
            rpos = lpos + 130

        # ##################  avoid car
        #
        # if time.time() > avoid_time and Opt == 0:
        #     Opt = 1
        #     print("------------------------OPT: ", Opt)
        #     print(f'Opt :{Opt}, left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #
        # if (ultra_msg[2] < 65 or ultra_msg[3] < 50) and Opt == 1:
        #     Opt = 2
        #     print(f'Opt :{Opt}, left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #
        #     #                 avoid_drive_right()
        #     max_time_end = time.time() + 0.4
        #     while True:
        #         drive(-110, 23)
        #         if time.time() > max_time_end:
        #             break
        #     print(f'Opt :{Opt}, step: 1,  left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #
        #     max_time_end = time.time() + 0.4  # start(True)
        #     while True:
        #         drive(130, 23)
        #         if time.time() > max_time_end:
        #             break
        #     print(f'Opt :{Opt}, step: 2,  left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #
        #     max_time_end = time.time() + 0.5  # changed line and to be stable
        #     while True:
        #         drive(30, 23)
        #         if time.time() > max_time_end:
        #             break
        #     print(f'Opt :{Opt}, step: 3,  left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #     turn_right = time.time() + 0.1
        #
        #     print(f'Opt :{Opt}, step: 4,  left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #
        #     max_time_end = time.time() + 0.7  # go back to the line
        #     while True:
        #         drive(130, 23)
        #         if time.time() > max_time_end:
        #             break
        #     print(f'Opt :{Opt}, step: 5,  left :{ultra_msg[3]}, center: {ultra_msg[2]}')
        #
        #     # max_time_end = time.time() + 0.2    #go back to the line
        #     # while True:
        #     #     drive(50,21)
        #     #     if time.time() > max_time_end:
        #     #         break
        #
        #     Opt = 3
        #     continue
        #
        # ##################
        #
        # if Opt == 3:
        #     ang = angle * 0.8
        #     drive(ang, 23)
        # else:
        #     drive(angle, 23)
        # ### avoid car end

        steer_angle = angle * 0.4
        # draw_steer(steer_angle)

        if angle < -17 or angle > 17:
            speed = 35
            print("angle:", angle, "speed:", speed, "car_curve", 'flag : ', flag, 'line_count : ', line_count)
        else:
            speed = 45
            print("angle:", angle, "speed", speed, "car_straight", 'flag : ', flag, 'line_count : ', line_count)

        # angle = angle - 10
        drive(angle, speed)
        cv2.waitKey(1)
        # sq.sleep()
        b_angle = angle
        b_error = error


if __name__ == '__main__':
    start()





