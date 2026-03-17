import cv2
import numpy as np

# 이미지 파일 경로
image_path = 'color_img/red2.jpg'

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

# 자동으로 설정된 HSV 범위를 기반으로 색상 범위 정의
lower_bound = np.array([h_min, s_min, v_min])
upper_bound = np.array([h_max, s_max, v_max])

# 전체 이미지에서 색상 범위 마스크 생성
mask_all = cv2.inRange(image_hsv, lower_bound, upper_bound)

# 결과 이미지에서 자동으로 색상 범위 추출
# (HSV 범위를 기반으로 특정 색상 범위를 추출할 수 있도록 임계값 조정 필요)
lower_red = np.array([0, 150, 150])
upper_red = np.array([10, 200, 200])

lower_red2 = np.array([175, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Yellow color range in HSV
lower_yellow = np.array([15, 150, 150])
upper_yellow = np.array([35, 255, 255])

# lower_green = np.array([30, 150, 150])  # 초록색 범위 조정
# upper_green = np.array([90, 255, 255])
lower_green = np.array([35, 125, 125])  # Hue, Saturation, Value 범위 조정
upper_green = np.array([85, 200, 200])

# 각 색상 범위에 따른 마스크 생성
mask_red1 = cv2.inRange(image_hsv, lower_red, upper_red)
mask_red2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

# 색상 범위에서 빨간색 영역을 제외
mask_yellow_no_red = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_red))
mask_green_no_red = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_red))

# 색상 영역 윤곽선 찾기
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_yellow, _ = cv2.findContours(mask_yellow_no_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green_no_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 색상 영역에 체크박스 그리기 (사각형)
for contour in contours_red:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 빨간색 사각형

for contour in contours_yellow:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 255), 2)  # 노란색 사각형

for contour in contours_green:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록색 사각형

# 결과 이미지 표시
cv2.imshow('Original Image with Bounding Boxes', image_bgr)
cv2.imshow('Red Areas', cv2.bitwise_and(image_bgr, image_bgr, mask=mask_red))
cv2.imshow('Yellow Areas', cv2.bitwise_and(image_bgr, image_bgr, mask=mask_yellow))
cv2.imshow('Green Areas', cv2.bitwise_and(image_bgr, image_bgr, mask=mask_green))

# 키 입력을 기다리고 윈도우를 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()
