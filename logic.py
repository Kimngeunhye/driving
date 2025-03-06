import cv2
import numpy as np

# 빨간색과 초록색의 HSV 색상 범위 설정
lower_red = np.array([0, 70, 120])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 120])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([45, 50, 50])  # 초록색의 채도(Saturation)와 명도(Value) 범위
upper_green = np.array([90, 255, 255])

# 신호등 상태를 실시간으로 감지하는 함수
def traffic_detect(frame):
    # HSV 색 공간으로 변환
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 마스크 (두 범위 합침)
    red_mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2  # 두 범위를 결합

    # 초록색 마스크
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 빨간색과 초록색 픽셀 수 계산
    red_pixel_count = cv2.countNonZero(red_mask)
    green_pixel_count = cv2.countNonZero(green_mask)

    # 빨간색과 초록색 픽셀의 합이 20 이하이면 zero 출력
    if (red_pixel_count + green_pixel_count) <= 20:
        signal_state = "zero"
    else:
        # 초록색 픽셀의 60%를 기준으로 판단
        green_threshold = green_pixel_count * 0.6
        if red_pixel_count > green_threshold:
            signal_state = "Red light"
        else:
            signal_state = "Green light"

    return signal_state, red_pixel_count, green_pixel_count

# 실시간 비디오 스트림을 통해 신호등 상태 감지
def main():
    # 라즈베리파이에서 카메라 연결 (0은 기본 카메라 장치)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 신호등 상태 감지
        signal_state, red_pixel_count, green_pixel_count = traffic_detect(frame)

        # 결과 출력 (콘솔 출력)
        print(f"Signal: {signal_state}, Red Pixels: {red_pixel_count}, Green Pixels: {green_pixel_count}")

        # 결과를 프레임에 표시
        cv2.putText(frame, f"Signal: {signal_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Red Pixels: {red_pixel_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Green Pixels: {green_pixel_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 프레임을 화면에 표시
        cv2.imshow('Traffic Light Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 시 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
