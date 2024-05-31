# from moviepy.editor import VideoFileClip
#
# video_path = 'data/short_video.mp4'
#
# # 동영상 불러오기
# video_clip = VideoFileClip(video_path)
#
# # 화면을 5% 우측으로 이동
# w, h = video_clip.size
# moved_clip = video_clip.set_position(0.30 * w, 'center')
#
# # 이어붙인 동영상을 파일로 저장
# output_path = 'data/moved_video.mp4'
# moved_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
#
# # 메모리에서 동영상 클립 제거
# video_clip.close()
# moved_clip.close()

# from moviepy.editor import VideoFileClip, CompositeVideoClip
#
# # 동영상 파일 경로 설정
# video_path = 'data/video.mp4'
#
# # 동영상 불러오기
# video_clip = VideoFileClip(video_path)
#
# # 화면 크기 설정 (예: 1920x1080)
# screen_width, screen_height = 1280, 720
#
# # 중앙에 위치시키기
# moved_clip_center = CompositeVideoClip([video_clip.set_position(('center', 'center'))], size=(screen_width, screen_height))
#
# # 왼쪽 위에 위치시키기
# moved_clip_top_left = CompositeVideoClip([video_clip.set_position((0, 0))], size=(screen_width, screen_height))
#
# # 오른쪽 아래에 위치시키기
# moved_clip_bottom_right = CompositeVideoClip([video_clip.set_position(('right', 'bottom'))], size=(screen_width, screen_height))
#
# # 화면 중앙에 백분율로 위치시키기
# moved_clip_percentage = CompositeVideoClip([video_clip.set_position(('50%', '50%'))], size=(screen_width, screen_height))
#
# # 위치가 이동된 클립을 각각 파일로 저장
# moved_clip_center.write_videofile('data/moved_video_center.mp4', codec='libx264', audio_codec='aac')
# moved_clip_top_left.write_videofile('data/moved_video_top_left.mp4', codec='libx264', audio_codec='aac')
# moved_clip_bottom_right.write_videofile('data/moved_video_bottom_right.mp4', codec='libx264', audio_codec='aac')
# moved_clip_percentage.write_videofile('data/moved_video_percentage.mp4', codec='libx264', audio_codec='aac')
#
# # 메모리에서 동영상 클립 제거
# video_clip.close()
# moved_clip_center.close()
# moved_clip_top_left.close()
# moved_clip_bottom_right.close()
# moved_clip_percentage.close()

from moviepy.editor import VideoFileClip, CompositeVideoClip

# 동영상 파일 경로 설정
video_path = 'data/video.mp4'

# 동영상 불러오기
video_clip = VideoFileClip(video_path)

# 화면 크기 설정 (예: 원본 영상의 크기)
screen_width, screen_height = video_clip.size

# 클립을 5픽셀 우측으로 이동시키기
moved_clip = video_clip.set_position((int(screen_width*0.1), 0))

# CompositeVideoClip을 사용하여 화면을 지정하고 이동된 클립을 배치
composite_clip = CompositeVideoClip([moved_clip], size=(screen_width, screen_height))

# 이동된 클립을 파일로 저장
output_path = 'data/moved_video.mp4'
composite_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# 메모리에서 동영상 클립 제거
video_clip.close()
moved_clip.close()
composite_clip.close()