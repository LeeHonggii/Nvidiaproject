from moviepy.editor import VideoFileClip, concatenate_videoclips
from math import radians, degrees, sin, cos, acos
from sympy import symbols, Eq, solve
from line_add import prepare_sample

dot_product = lambda v1, v2: v1[0] * v2[0] + v1[1] * v2[1]
cross_product = lambda v1, v2: v1[0] * v2[1] - v1[1] * v2[0]
length = lambda v: (v[0]**2 + v[1]**2)**(1/2)
dist = lambda w, h: (w**2 + h**2)**(1/2)


def calculate_signed_angle(v1, v2):
    dot_prod = dot_product(v1, v2)
    length_v1 = length(v1)
    length_v2 = length(v2)

    cos_theta = dot_prod / (length_v1 * length_v2)
    angle_radians = acos(cos_theta)
    angle_degrees = degrees(angle_radians)

    cross_prod = cross_product(v1, v2)
    if cross_prod < 0:
        angle_degrees = -angle_degrees
    return angle_degrees


def find_intersection(x1, y1, a, x2, y2, b):
    x, y = symbols('x y')

    eq1 = Eq((x - x1)**2 + (y - y1)**2, a**2)
    eq2 = Eq((x - x2)**2 + (y - y2)**2, b**2)

    solutions = solve((eq1, eq2), (x, y))

    return solutions


def get_adjusted_clip(clip, v1, v2):
    x1, y1, x12, y12 = v1
    x2, y2, x22, y22 = v2

    width, height = clip.size
    center_x = width / 2
    center_y = height / 2

    w1, h1 = x12 - x1, y12 - y1
    w2, h2 = x22 - x2, y22 - y2
    diff_x = x2 - x1
    diff_y = y2 - y1
    distance = dist(diff_x, diff_y)
    length_1 = dist(w1, h1)
    length_2 = dist(w2, h2)
    ratio = length_1 / length_2
    print("(", x1, y1, w1, h1, ") to (", x2, y2, w2, h2, ")")
    print(f"From [p1({x1}, {y1}), v1({w1}, {h1})] To [p2({x2}, {y2}), v2({w2}, {h2})]")
    print(f"distance: {distance:.2f}, length_1: {length_1:.2f}, length_2: {length_2:.2f}, ratio: {ratio:.2f}")

    # 1) moving
    clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-diff_x, -diff_y))

    # 2) rotation
    rotate_angle = calculate_signed_angle((w1, h1), (w2, h2))
    distance_from_center = dist(x1 - center_x, y1 - center_y)
    opposite_side = distance_from_center * sin(radians((abs(rotate_angle) / 2)))
    near_side = distance_from_center * cos(radians((abs(rotate_angle) / 2)))
    solution = find_intersection(x1, y1, opposite_side, center_x, center_y, near_side)
    print(f"Signed angle between vectors: {rotate_angle:.2f} degrees")
    # print(f"distance_from_center: {distance_from_center:.2f} opposite_side: {opposite_side:.2f}, near_side: {near_side:.2f}")
    # print(solution)
    # print(f"rotated_move_x: {rotated_move_x:.2f}, rotated_move_x: {rotated_move_y:.2f}")

    clip = clip.rotate(rotate_angle)

    i = 0 if rotate_angle > 0 else 1
    rotated_move_x = (solution[i][0] - x1) * 2
    rotated_move_y = (solution[i][1] - y1) * 2

    rotated_window_width, rotated_window_height = clip.size
    print(f"rotated window size: {rotated_window_width} x {rotated_window_height}")
    window_move_x = abs(rotated_window_width - width) / 2
    window_move_y = abs(rotated_window_height - height) / 2

    # 3) resize
    clip = clip.resize(ratio)

    scaled_move_x = x1 * (ratio - 1)
    scaled_move_y = y1 * (ratio - 1)

    # 4) total correction
    total_move_x = (rotated_move_x + window_move_x) * ratio + scaled_move_x
    total_move_y = (rotated_move_y + window_move_y) * ratio + scaled_move_y
    print(f"rotated_move_x: {rotated_move_x:.1f} + window_move_x: {window_move_x:.1f} + scaled_move_x: {scaled_move_x:.1f} = total_move_x: {total_move_x:.1f}")
    print(f"rotated_move_y: {rotated_move_y:.1f} + window_move_y: {window_move_y:.1f} + scaled_move_y: {scaled_move_y:.1f} = total_move_y: {total_move_y:.1f}")

    clip = clip.on_color(size=(1280, 720), color=(0, 0, 0), pos=(-total_move_x, -total_move_y))
    clip = clip.resize(newsize=(1280, 720))

    return clip


if __name__ == "__main__":
    clip1 = VideoFileClip("data/line1.mp4")
    clip2 = VideoFileClip("data/line2.mp4")

    line1 = (100, 100, 220, 100)
    line2 = (120, 120, 200, 140)
    # prepare_sample(line1, line2)

    adjust_clip = get_adjusted_clip(clip2, line1, line2)

    final_clip = concatenate_videoclips([clip1, adjust_clip])
    final_clip.write_videofile("data/adjust_video.mp4", codec='libx264', audio_codec='aac')

    clip1.close()
    clip2.close()
    adjust_clip.close()
    final_clip.close()
