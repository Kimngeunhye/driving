import cv2
import torch

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 파일 경로
image_path = 'color_img/green7.jpg'

# 이미지에서 신호등 탐지
results = model(image_path)

# 결과를 Pandas DataFrame으로 변환
detections = results.pandas().xyxy[0]

# DataFrame 열 이름 출력
print(detections.columns)

# 검출 결과를 원본 이미지에 표시
# 열 이름 확인 후 열 이름을 코드에 맞게 수정
for index, row in detections.iterrows():
    if row['class'] == 9:  # YOLOv5에서 신호등 클래스 ID가 9인 경우
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        # 이미지 로드
        img = cv2.imread(image_path)
        # 신호등 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 신호등 클래스와 신뢰도 레이블 추가
        cv2.putText(img, f'Confidence: {row["confidence"]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
