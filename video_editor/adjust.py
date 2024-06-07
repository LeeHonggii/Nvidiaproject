
#
# # scaled_clip = clip.resize(0.5)
# moved_clip = clip.set_position((30, 50))
#
# final_clip = CompositeVideoClip([moved_clip])
#
# final_clip.write_videofile("data/adjust_video.mp4", codec='libx264')
#

from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
import math_lib as ml
import math
maxx = max
length = lambda w, h: (w**2 + h**2)**(1/2)


clip1 = VideoFileClip("data/line1.mp4") # (100, 100) to (300, 100)
clip2 = VideoFileClip("data/line2.mp4") # (100, 100) to (200, 100)

width, height = clip1.size
center_x = width / 2
center_y = height / 2

# # def case_1():
# x1, y1, w1, h1 = 100, 100, 200, 0
# x2, y2, w2, h2 = 100, 100, 100, 0
#
# distance = length(x2-x1, y2-y1)
# length_1 = length(w1, h1)
# length_2 = length(w2, h2)
# ratio = length_1 / length_2
# print(distance, length_1, length_2, ratio)
#
# new_x = x2 * ratio
# new_y = y2 * ratio
# moved_x = x2 - new_x
# moved_y = y2 - new_y
# print(new_x, new_y, moved_x, moved_y)
#
# clip = clip2.resize(ratio)
# ww1, hh1 = clip.size
# print(ww1,hh1)
# ad_x = int(abs(ww1-width)/2)
# ad_y = int(abs(hh1-height)/2)
# # total_x = moved_x + ad_x
# # total_y = moved_y + ad_y
# total_x = -moved_x
# total_y = -moved_y
# print(f"move_x: {moved_x:.2f} ad_x: {ad_x:.2f} total: {total_x: .2f}")
# print(f"move_y: {moved_y:.2f} ad_x: {ad_y:.2f} total: {total_y: .2f}")
#
# clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-total_x, -total_y))
#
# # clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-move_x, -move_y))
# # clip = clip.set_position((-move_x, -move_y))
# # clip = CompositeVideoClip([clip])
# adjust_clip = clip.resize(newsize=(1280, 720))


# resize_clip = clip2.resize(ratio)
# moved_clip = resize_clip.set_position((move_x, move_y))
# composed_clip = CompositeVideoClip([moved_clip])
# adjust_clip = composed_clip.crop(x1=0, y1=0, x2=width, y2=height)


x1, y1, x12, y12 = 100, 100, 300, 200
x2, y2, x22, y22 = 150, 100, 250, 200

w1, h1 = x12 - x1, y12 - y1
w2, h2 = x22 - x2, y22 - y2

diff_x = x2 - x1
diff_y = y2 - y1
distance = length(diff_x, diff_y)
length_1 = length(w1, h1)
length_2 = length(w2, h2)
ratio = length_1 / length_2
print(x1, y1, w1, h1, "to", x2, y2, w2, h2)
print(f"distance: {distance:.2f}, length_1: {length_1:.2f}, length_2: {length_2:.2f}, ratio: {ratio:.2f}")

smove_x = int(x1 * (ratio - 1))
smove_y = int(y1 * (ratio - 1))
dmove_x = int(diff_x * ratio)
dmove_y = int(diff_y * ratio)
print(smove_x, smove_y, dmove_x, dmove_y)


v1 = (w1, h1)  # (x2 - x1, y2 - y1)
v2 = (w2, h2)  # (w2 - w1, h2 - h1)

rotate_angle = ml.calculate_signed_angle(v1, v2)
distance_from_center = length(x1-center_x, y1-center_y)
opposite_side = distance_from_center * math.sin(math.radians(((abs(rotate_angle) / 2))))
near_side = distance_from_center * math.cos(math.radians(((abs(rotate_angle) / 2))))
# print(f"abs(rotate_angle) / 2: {abs(rotate_angle) / 2:.2f}")
# print(f"math.cos((abs(rotate_angle) / 2): {math.cos((abs(rotate_angle) / 2)):.2f}")
solution = ml.find_intersection(x1, y1, opposite_side, center_x, center_y, near_side)

xxx, yyy = solution[0]
# print(solution[0])
rotated_x = (xxx - x1) *2
rotated_y = (yyy - y1) *2
# print(xxx, yyy, rotated_x, rotated_y)

# move_x = (solution[0][0] - x1) * 2
# move_y = (solution[0][1] - y1) * 2

print(f"Signed angle between vectors: {rotate_angle} degrees")
print(f"distance_from_center: {distance_from_center:.2f} opposite_side: {opposite_side:.2f}, near_side: {near_side:.2f}")
print(solution)
print(f"move_x: {rotated_x:.2f}, move_y: {rotated_y:.2f}")

clip = clip2.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-diff_x, -diff_y))
ww4, hh4 = clip.size
print(ww4, hh4)

clip = clip.rotate(rotate_angle)
# clip = clip2.rotate(rotate_angle)
ww1, hh1 = clip.size
print(ww1,hh1)
ad_x = int(abs(ww1-width)/2)
ad_y = int(abs(hh1-height)/2)

# new_x = x2 * ratio
# new_y = y2 * ratio
clip = clip.resize(ratio)
ww2, hh2 = clip.size
print(ww2,hh2)

# total_x = rotated_x + ad_x - smove_x
# total_y = rotated_y + ad_y - smove_y
# total_x = (rotated_x + ad_x) * ratio + smove_x + dmove_x
# total_y = (rotated_y + ad_y) * ratio + smove_y + dmove_y
total_x = (rotated_x + ad_x) * ratio + smove_x
total_y = (rotated_y + ad_y) * ratio + smove_y
print(f"move_x: {rotated_x:.2f} ad_x: {ad_x:.2f} smove_x: {smove_x:.2f} total: {total_x:.2f}")
print(f"move_y: {rotated_y:.2f} ad_x: {ad_y:.2f} smove_y: {smove_y:.2f} total: {total_y:.2f}")

# clip = clip.resize(1.5)
# clip = clip.on_color(size=(1920, 1080), color=(0, 0, 0), pos=('center', 'center'))
# clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=('center', 'center'))
# clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-25, -370))
# clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(0, 0))
clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-total_x, -total_y))

# clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-move_x, -move_y))
# clip = clip.set_position((-move_x, -move_y))
# clip = CompositeVideoClip([clip])
adjust_clip = clip.resize(newsize=(1280, 720))

# resize_clip = rotated_clip.resize(ratio)
# moved_clip = resize_clip.set_position((move_x, move_y))
# composed_clip = CompositeVideoClip([moved_clip])
# adjust_clip = composed_clip.crop(x1=0, y1=0, x2=width, y2=height)

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