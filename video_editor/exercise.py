# from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
# import random
# #
# # scaled_clip = clip.subclip(5, 15).resize(newsize=(1280, 720))
# #
# # # final_clip = CompositeVideoClip([scaled_clip.set_position("center")], size=(1280, 720))
# #
# # scaled_clip.write_videofile("data/output_video.mp4", codec='libx264')
#
#
# input_file = "data/short_video.mp4"
# output_file = f"data/output_video_{random.randint(0,999):3d}.mp4"
#
# clip = VideoFileClip(input_file)
# duration = clip.duration
#
# clip = clip.rotate(5)
# # clip = clip.resize(1.7)
# # clip = clip.on_color(size=(1920, 1080), color=(0, 0, 0), pos=('center', 'center'))
# clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=('center', 'center'))
# # clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-100, -100))
# final_clip = clip.resize(newsize=(1280, 720))
#
# rotated_clip = clip.rotate(5)
# rotated_size = rotated_clip.size
# background = rotated_clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=('center', 'center'))
# final_clip = background.resize(newsize=(1280, 720))
# final_clip.write_videofile(output_file, codec='libx264')
#
#
#
# # original_size = clip.size
# #
# # # 1920x1080 크기의 검정색 배경을 만듭니다.
# # background = clip.resize(newsize=(1920, 1080)).on_color(size=(1920, 1080), color=(0, 0, 0))
# #
# # # 원본 동영상을 중앙에 위치시킵니다.
# # final_clip = background.set_position(('center', 'center'))
# #
# # # 변환된 동영상을 파일로 저장합니다.
# # final_clip.write_videofile(output_file, codec='libx264')
#
#
# # final_clip = resize_clip.rotate(30)
#
# final_clip.write_videofile(output_file, codec='libx264')
#
#

import numpy as np
from sympy import symbols, Eq, solve

def find_intersection(x1, y1, a, x2, y2, b):
    # Define the symbols
    x, y = symbols('x y')

    # Define the equations for the two circles
    eq1 = Eq((x - x1)**2 + (y - y1)**2, a**2)
    eq2 = Eq((x - x2)**2 + (y - y2)**2, b**2)

    # Solve the system of equations
    solutions = solve((eq1, eq2), (x, y))

    return solutions

# Example usage
x1, y1, a = 0, 0, 5
x2, y2, b = 4, 0, 3

solutions = find_intersection(x1, y1, a, x2, y2, b)
print("Solutions:", solutions)

import math

def find_other_sides(x, a, b):
    # 각도를 라디안으로 변환
    a_rad = math.radians(a)
    b_rad = math.radians(b)

    # y는 a에 대한 인접변, z는 a에 대한 반대변
    y = x * math.cos(a_rad)
    z = x * math.sin(a_rad)

    return y, z

# 빗변의 길이와 각도 입력
x = float(input("빗변의 길이를 입력하세요: "))
a = float(input("첫 번째 각도를 입력하세요 (도): "))
b = float(input("두 번째 각도를 입력하세요 (도): "))

# 다른 두 변의 길이 구하기
y, z = find_other_sides(x, a, b)

print(f"첫 번째 변의 길이 (인접변): {y}")
print(f"두 번째 변의 길이 (반대변): {z}")
