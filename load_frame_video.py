import cv2  # OpenCV 라이브러리 임포트


def show_frame(video_path, frame_number):
    # 비디오 파일을 읽기 위한 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 총 프레임 수를 가져옵니다
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 요청한 프레임 번호가 비디오의 프레임 수를 초과하지 않는지 확인
    if frame_number >= total_frames:
        print("Error: Frame number exceeds total number of frames in the video.")
        cap.release()
        return

    # 비디오에서 특정 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 프레임을 읽음
    ret, frame = cap.read()

    # 프레임을 성공적으로 읽었는지 확인
    if not ret:
        print("Failed to read the frame")
        cap.release()
        return

    # 읽은 프레임을 화면에 표시
    cv2.imshow(f"Frame at {frame_number}", frame)
    cv2.waitKey(0)  # 유저가 키를 누를 때까지 대기
    cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫음

    # 자원 해제
    cap.release()


# 사용 예
video_path = "video/ive_baddie_1.mp4"  # 비디오 파일 경로
frame_number = 2928  # 보고 싶은 프레임 번호
show_frame(video_path, frame_number)
