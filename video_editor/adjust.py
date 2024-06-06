
#
# # scaled_clip = clip.resize(0.5)
# moved_clip = clip.set_position((30, 50))
#
# final_clip = CompositeVideoClip([moved_clip])
#
# final_clip.write_videofile("data/adjust_video.mp4", codec='libx264')
#

from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import math
maxx = max
length = lambda w, h: (w**2 + h**2)**(1/2)


clip1 = VideoFileClip("data/line1.mp4") # (100, 100) to (300, 100)
clip2 = VideoFileClip("data/line2.mp4") # (100, 100) to (200, 100)

width, height = clip1.size

def case_1():
    x1, y1, w1, h1 = 100, 100, 200, 0
    x2, y2, w2, h2 = 100, 100, 100, 0

    distance = length(x2-x1, y2-y1)
    length_1 = length(w1, h1)
    length_2 = length(w2, h2)
    ratio = length_1 / length_2
    print(distance, length_1, length_2, ratio)

    new_x = x2 * ratio
    new_y = y2 * ratio
    move_x = x2 - new_x
    move_y = y2 - new_y
    print(new_x, new_y, move_x, move_y)

    resize_clip = clip2.resize(ratio)
    moved_clip = resize_clip.set_position((move_x, move_y))
    composed_clip = CompositeVideoClip([moved_clip])
    adjust_clip = composed_clip.crop(x1=0, y1=0, x2=width, y2=height)

x1, y1, x12, y12 = 100, 100, 300, 200
x2, y2, x22, y22 = 100, 100, 200, 200

w1, h1 = x12 - x1, y12 - y1
w2, h2 = x22 - x2, y22 - y2

distance = length(x2-x1, y2-y1)
length_1 = length(w1, h1)
length_2 = length(w2, h2)
ratio = length_1 / length_2
print(x1, y1, w1, h1, "to", x2, y2, w2, h2)
print(f"distance: {distance:.2f}, length_1: {length_1:.2f}, length_2: {length_2:.2f}, ratio: {ratio:.2f}")

new_x = int(x2 * ratio)
new_y = int(y2 * ratio)
move_x = int(x2 - new_x)
move_y = int(y2 - new_y)
print(new_x, new_y, move_x, move_y)



# 내적 계산 함수
dot_product = lambda v1, v2: v1[0] * v2[0] + v1[1] * v2[1]
cross_product = lambda v1, v2: v1[0] * v2[1] - v1[1] * v2[0]

# 벡터 크기 계산 함수
length = lambda v: (v[0]**2 + v[1]**2)**(1/2)

# 각도 계산 함수
# def calculate_angle(v1, v2):
#     dot_prod = dot_product(v1, v2)
#     length_v1 = length(v1)
#     length_v2 = length(v2)
#     # 코사인 값 계산
#     cos_theta = dot_prod / (length_v1 * length_v2)
#     # 각도 계산 (라디안 단위)
#     angle_radians = math.acos(cos_theta)
#     # 라디안을 도로 변환
#     angle_degrees = math.degrees(angle_radians)
#     return angle_degrees
def calculate_signed_angle(v1, v2):
    dot_prod = dot_product(v1, v2)
    length_v1 = length(v1)
    length_v2 = length(v2)
    # 코사인 값 계산
    cos_theta = dot_prod / (length_v1 * length_v2)
    # 각도 계산 (라디안 단위)
    angle_radians = math.acos(cos_theta)
    # 라디안을 도로 변환
    angle_degrees = math.degrees(angle_radians)
    # 외적을 이용하여 부호 결정
    cross_prod = cross_product(v1, v2)
    if cross_prod < 0:
        angle_degrees = -angle_degrees
    return angle_degrees

# 예시 벡터
# v1 = (3, 4)  # (x2 - x1, y2 - y1)
# v2 = (4, 3)  # (w2 - w1, h2 - h1)
# 예시 벡터
# v1 = (3, 4)  # (x2 - x1, y2 - y1)
# v2 = (6, 8)  # (w2 - w1, h2 - h1)
v1 = (w1, h1)  # (x2 - x1, y2 - y1)
v2 = (w2, h2)  # (w2 - w1, h2 - h1)
# v2 = (w1, h1)  # (x2 - x1, y2 - y1)
# v1 = (w2, h2)  # (w2 - w1, h2 - h1)

# 각도 계산
angle = calculate_signed_angle(v1, v2)
print(f"Signed angle between vectors: {angle} degrees")

# 각도 계산
# angle = calculate_angle(v1, v2)
# print(f"Angle between vectors: {angle} degrees")

clip = clip2.rotate(angle)
# clip = clip.resize(1.5)
# clip = clip.on_color(size=(1920, 1080), color=(0, 0, 0), pos=('center', 'center'))
clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-25, -370))
# clip = clip.set_position((25, -350))
# clip = CompositeVideoClip([clip])


rotated_clip = clip.resize(newsize=(1280, 720))

length = lambda w, h: (w**2 + h**2)**(1/2)

distance = length(x2 - x1, y2 - y1)
length_1 = length(w1, h1)
length_2 = length(w2, h2)
ratio = length_1 / length_2
print(distance, length_1, length_2, ratio)

new_x = x2 * ratio
new_y = y2 * ratio
move_x = x2 - new_x
move_y = y2 - new_y
print(new_x, new_y, move_x, move_y)

resize_clip = rotated_clip.resize(ratio)
moved_clip = resize_clip.set_position((move_x, move_y))
composed_clip = CompositeVideoClip([moved_clip])
adjust_clip = composed_clip.crop(x1=0, y1=0, x2=width, y2=height)

# clip = clip2
# clip = clip.rotate(angle)
# # clip = clip2.resize(ratio)
# # clip = clip.set_position((move_x, move_y))
# angle_radians = math.radians(angle)
# # move_x = height * math.cos(angle_radians)
# move_y = width * math.sin(angle_radians)
# move_x = 0
# print(move_x, move_y)
# clip = clip.set_position((-move_x, -move_y))
#
# composed_clip = CompositeVideoClip([clip])
# adjust_clip = composed_clip.crop(x1=0, y1=0, x2=width, y2=height)
# resize_clip = clip2.resize(ratio)
# moved_clip = resize_clip.set_position((move_x, move_y))
# rotated_clip = moved_clip.rotate(angle)
# composed_clip = CompositeVideoClip([rotated_clip])
# adjust_clip = composed_clip.crop(x1=0, y1=0, x2=width, y2=height)


# adjust_clip = moved_clip.crop(x1=0, y1=0, x2=width, y2=height)

# clip1 = VideoFileClip("data/line1.mp4") # (100, 100) to (200, 100)
# clip2 = VideoFileClip("data/line2.mp4") # (130, 150) to (230, 150)
#
# width, height = clip1.size
#
# move_x = 30
# move_y = 50
# move_x_ratio = move_x / width
# move_y_ratio = move_y / height
# move_ratio = maxx(move_x_ratio, move_y_ratio)
#
# print(width, height, move_x, move_y, f"{move_x_ratio:.3f}", f"{move_y_ratio:.3f}", f"> movie_ratio: {move_ratio:.3f}")
#
# scaled_clip = clip1.resize(1 + move_ratio)
#
# moved_clip = scaled_clip.set_position((30-30*move_ratio, 50-50*move_ratio))
#
# adjust_clip = CompositeVideoClip([moved_clip])

# video_clip_resized = video_clip_resized.crop(x1=0, y1=0, x2=video_clip.w, y2=video_clip.h)


#
# # 비디오 클립의 크기를 확대
# scaled_clip = clip.resize(1.5)  # 비디오를 1.5배 확대
#
# # 비디오 클립의 위치 조정
# moved_clip = scaled_clip.set_position((30, 50))
#
# # 조정된 비디오 클립을 합성 비디오 클립으로 생성
# final_clip = CompositeVideoClip([moved_clip])
#
# # 최종 합성 비디오를 새 파일로 작성
# final_clip.write_videofile("data/adjust_video.mp4", codec='libx264')


#
# clip1 = VideoFileClip("data/line1.mp4")
# clip2 = VideoFileClip("data/line2.mp4")
#
final = concatenate_videoclips([clip1, adjust_clip])
final.write_videofile("data/adjust_video.mp4", codec='libx264')
#
#
clip1.close()
clip2.close()
adjust_clip.close()
final.close()