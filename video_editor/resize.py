from moviepy.editor import VideoFileClip

# 동영상 파일 경로 설정
video_path = 'data/ive_iam_2.mp4'

# 동영상 불러오기
video_clip = VideoFileClip(video_path)
video_clip = video_clip.subclip(15, 35)

# 비디오 확대 (가로와 세로 길이를 각각 1.5배로 확대)
# 여기서는 가로와 세로 길이를 각각 1.5배로 확대합니다.
# 비디오 내용은 확대되지만 실제 비디오의 크기는 변경되지 않습니다.
video_clip_resized = video_clip.resize(newsize=(1280, 720))

# 비디오를 원본 크기로 자르기
# 여기서는 비디오를 원본 크기로 자릅니다.
# 자르는 영역을 지정하지 않으면 자동으로 원본 크기로 잘립니다.
video_clip_resized = video_clip_resized.crop(x1=0, y1=0, x2=video_clip.w, y2=video_clip.h)
# video_clip_resized = video_clip.crop(x1=0, y1=0, x2=video_clip.w*0.5, y2=video_clip.h*0.5)
# video_clip_resized = video_clip.crop(x1=video_clip.w*0.25, y1=video_clip.h*0.25, x2=video_clip.w*0.75, y2=video_clip.h*0.75)

# 크기가 조정된 비디오 클립을 파일로 저장
output_path = 'data/resized_video2.mp4'
video_clip_resized.write_videofile(output_path, codec='libx264', audio_codec='aac')

# 메모리에서 동영상 클립 제거 (선택 사항)
video_clip.close()
video_clip_resized.close()
