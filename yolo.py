import cv2
import numpy as np
import torch

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 파일 경로
image_path = 'color_img/red3.jpg'
# 이미지를 BGR 색상으로 읽어들임
image_bgr = cv2.imread(image_path)

# BGR 이미지를 HSV 색상으로 변환
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# HSV 범위 정의
lower_red = np.array([0, 150, 150])
upper_red = np.array([10, 200, 200])

lower_red2 = np.array([175, 100, 100])
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([15, 150, 150])
upper_yellow = np.array([35, 255, 255])

lower_green = np.array([35, 125, 125])
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

# 색상 영역에 체크박스와 텍스트 추가
for contour in contours_red:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 빨간색 사각형
    cv2.putText(image_bgr, 'Red', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for contour in contours_yellow:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 노란색 사각형
    cv2.putText(image_bgr, 'Yellow', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

for contour in contours_green:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 초록색 사각형
    cv2.putText(image_bgr, 'Green', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# YOLOv5 모델로 객체 탐지
results = model(image_path)

# 결과를 Pandas DataFrame으로 변환
detections = results.pandas().xyxy[0]

# 신호등의 가장 밝은 색상 찾기
for index, row in detections.iterrows():
    if row['class'] == 9:  # YOLOv5에서 신호등 클래스 ID가 9인 경우
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # 신호등 영역 추출
        traffic_light_roi = image_hsv[y1:y2, x1:x2]

        # 각 색상 범위의 평균 Value 채널 계산
        mean_red = np.mean(
            traffic_light_roi[np.bitwise_and(mask_red[y1:y2, x1:x2] > 0, traffic_light_roi[:, :, 1] > 0)])
        mean_yellow = np.mean(
            traffic_light_roi[np.bitwise_and(mask_yellow[y1:y2, x1:x2] > 0, traffic_light_roi[:, :, 1] > 0)])
        mean_green = np.mean(
            traffic_light_roi[np.bitwise_and(mask_green[y1:y2, x1:x2] > 0, traffic_light_roi[:, :, 1] > 0)])

        # 가장 밝은 색상 찾기
        colors_mean = {'Red': mean_red, 'Yellow': mean_yellow, 'Green': mean_green}
        brightest_color = max(colors_mean, key=colors_mean.get)

        # 신호등 박스 그리기
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 신호등 클래스와 신뢰도 레이블 추가
        cv2.putText(image_bgr, f'Traffic Light: {row["confidence"]:.2f}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        # 가장 밝은 색상 레이블 추가
        cv2.putText(image_bgr, f'Brightest Color: {brightest_color}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow('Combined Result', image_bgr)

# 키 입력을 기다리고 윈도우를 닫음
cv2.waitKey(0)
cv2.destroyAllWindows()
