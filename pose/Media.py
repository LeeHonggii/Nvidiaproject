import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 동영상 파일 열기
cap = cv2.VideoCapture('test.mp4')

# Pose 모듈 사용
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 포즈 추정 수행
        results = pose.process(image)

        # 이미지 다시 쓰기 가능으로 설정
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 관절 값 추출 및 시각화
        if results.pose_landmarks:
            # 각 관절 값을 텍스트로 표시 (손목과 손 관절 제외)
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # 손목과 손가락 관절 인덱스 (왼쪽 손목: 15, 오른쪽 손목: 16, 21~32는 손가락 관절)
                if idx in [15, 16] or (idx >= 21 and idx <= 32):
                    continue

                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(image, f'{idx}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # 손목과 손을 제외한 관절 연결선만 그리기
            connections = [conn for conn in mp_pose.POSE_CONNECTIONS if conn[0] not in [15, 16] and conn[1] not in [15, 16]]
            mp_drawing.draw_landmarks(image, results.pose_landmarks, connections)

        # 결과 영상 보기
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
