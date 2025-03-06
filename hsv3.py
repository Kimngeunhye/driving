import cv2
import numpy as np

# 이미지 파일 경로
image_path = 'color_img/red3.jpg'

# 이미지를 BGR 색상으로 읽어들임
image_bgr = cv2.imread(image_path)

# BGR 이미지를 HSV 색상으로 변환
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# HSV 배열의 최소값과 최대값 계산
h_min, s_min, v_min = np.min(image_hsv, axis=(0, 1))
h_max, s_max, v_max = np.max(image_hsv, axis=(0, 1))

# HSV 범위를 출력
print(f"HSV 범위 추출 결과:")
print(f"Hue 최소값: {h_min}, 최대값: {h_max}")
print(f"Saturation 최소값: {s_min}, 최대값: {s_max}")
print(f"Value 최소값: {v_min}, 최대값: {v_max}")

# HSV 범위를 사용하여 색상 범위 정의
lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

# 전체 이미지에서 색상 범위 마스크 생성
mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

# 원본 이미지에서 색상 범위 필터링
result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

# 결과 이미지 표시
cv2.imshow('Original Image', image_bgr)
cv2.imshow('Filtered Result', result)

# 키 입력을 기다리고 윈도우를 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()