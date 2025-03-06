import cv2
import numpy as np

# 이미지 파일 경로
image_path = 'color_img/red4.jpg'

# 이미지를 BGR 색상으로 읽어들임
image_bgr = cv2.imread(image_path)

# BGR 이미지를 HSV 색상으로 변환
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# HSV 색상 범위 정의
lower_yellow = np.array([0, 0, 12])  # yellow range
upper_yellow = np.array([179, 255, 255])

# 색상 범위에 따른 마스크 생성
mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

# 배경을 검은색으로 설정
background_black = np.zeros_like(image_bgr)
result_yellow = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_yellow)
background_black = cv2.bitwise_and(background_black, background_black, mask=cv2.bitwise_not(mask_yellow))

# 결과 이미지 생성
result_with_black_background = cv2.add(result_yellow, background_black)

# 결과 이미지 표시
cv2.imshow('Original Image', image_bgr)
cv2.imshow('Filtered Result with Black Background', result_with_black_background)

# 키 입력을 기다리고 윈도우를 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()
