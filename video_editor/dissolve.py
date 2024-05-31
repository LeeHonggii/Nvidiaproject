from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip

# 동영상 파일 읽기
clip1 = VideoFileClip("data/video1.mp4")
clip2 = VideoFileClip("data/video2.mp4")
clip1 = clip1.subclip(5, 10)
clip2 = clip2.subclip(5, 10)

# # 페이드 효과 설정 (2초 동안 교차 페이드)
# fade_duration = 0.1
#
# # 첫 번째 클립의 마지막 부분과 두 번째 클립의 첫 부분을 페이드 처리
# clip1 = clip1.crossfadeout(fade_duration)
# clip2 = clip2.crossfadein(fade_duration)
#
# # 클립들을 연결
# final_clip = concatenate_videoclips([clip1, clip2], method="compose")
#
# # 결과를 파일로 저장
# final_clip.write_videofile("data/fade_video.mp4", codec="libx264", fps=24)
#
# from moviepy.editor import VideoFileClip, concatenate_videoclips
#
# # 동영상 파일 읽기
# clip1 = VideoFileClip("video1.mp4")
# clip2 = VideoFileClip("video2.mp4")

# 페이드 효과 설정 (2초 동안 겹쳐서 전환)
# fade_duration = 3
#
# # 첫 번째 클립의 길이
# duration1 = clip1.duration
#
# # 두 번째 클립의 시작 부분을 잘라내어 첫 번째 클립에 겹치기
# clip2 = clip2.set_start(duration1 - fade_duration)
#
# # 두 번째 클립의 길이를 전체 길이로 확장
# final_duration = duration1 + clip2.duration - fade_duration
#
# # CompositeVideoClip을 사용하여 두 클립을 겹쳐서 연결
# final_clip = CompositeVideoClip([clip1, clip2]).set_duration(final_duration)
#
# # 결과를 파일로 저장
# final_clip.write_videofile("data/fade_video.mp4", codec="libx264", fps=24)
#

# from moviepy.editor import VideoFileClip, concatenate_videoclips
#
# # 동영상 파일 읽기
# clip1 = VideoFileClip("video1.mp4")
# clip2 = VideoFileClip("video2.mp4")

dissolve_duration = 2

# 디졸브 효과를 적용하여 클립들을 연결
final_clip = concatenate_videoclips([clip1, clip2], method="chain", padding=-dissolve_duration)

# 결과를 파일로 저장
final_clip.write_videofile("data/desolve_video.mp4", codec="libx264", fps=24)

# padding = 2
#
# video_clips = [clip1, clip2]
#
# video_fx_list = [video_clips[0]]
#
# idx = video_clips[0].duration - padding
# for video in video_clips[1:]:
#     video_fx_list.append(video.set_start(idx).crossfadein(padding))
#     idx += video.duration - padding
#
# # final_video = CompositeVideoClip(video_fx_list)
# # final_video.write_videofile('data/desolve_video.mp4', fps=clip1.fps) # add any remaining params
# # CompositeVideoClip을 사용하여 두 클립을 겹쳐서 연결
# final_clip = CompositeVideoClip(video_fx_list)
#
# # 결과를 파일로 저장
# final_clip.write_videofile("data/fade_video.mp4", codec="libx264", fps=24)
#
#
# custom_padding = 2
# final_video = concatenate(
#     [
#         clip1,
#         clip2.crossfadein(custom_padding),
#         clip3.crossfadein(custom_padding)
#     ],
#     padding=-custom_padding,
#     method="chain"
# )
# final_video.write_videofile(target_path, fps=clip.fps) # add any remaining params