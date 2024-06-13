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
    # print(cos_theta, dot_prod, length_v1, length_v2)
    if cos_theta > 1:
        cos_theta = 1
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


def vector_interpolation(v1, v2):
    # diff_x, diff_y, ratio, angle, px, py, pw, ph, _, _, _, _ = vector_difference(prev_vector, curr_vector)
    # print("vector_difference called", prev_vector, curr_vector)
    # px, py, pw, ph = prev_vector
    # cx, cy, cw, ch = clurr_vector
    # px2, py2 = px + pw, py + ph
    # cx2, cy2 = cx + cw, cy + ch
    # adjust_vector = int(px + (diff_x / 2)), int(py + (diff_y / 2)), int(pw * ((1/ratio) ** 0.5)), int(ph * ((1/ratio) ** 0.5))
    x1, y1, w1, h1 = v1
    x2, y2, w2, h2 = v2
    x12, y12 = x1 + w1, y1 + h1
    x22, y22 = x2 + w2, y2 + h2
    ix1, iy1, ix2, iy2 = (x1 + x2) / 2, (y1 + y2) / 2, (x12 + x22) / 2, (y12 + y22) / 2

    return [ix1, iy1, ix2 - ix1, iy2 - iy1]


def get_adjusted_clip(clip, v1, v2):
    x1, y1, w1, h1 = v1
    x2, y2, w2, h2 = v2
    x12, y12 = x1 + w1, y1 + h1
    x22, y22 = x2 + w2, y2 + h2
    # x1, y1, x12, y12 = v1
    # x2, y2, x22, y22 = v2
    # w1, h1 = x12 - x1, y12 - y1
    # w2, h2 = x22 - x2, y22 - y2

    diff_x = x2 - x1
    diff_y = y2 - y1
    # distance = dist(diff_x, diff_y)
    length_1 = dist(w1, h1)
    length_2 = dist(w2, h2)
    ratio = length_1 / length_2
    rotate_angle = calculate_signed_angle((w1, h1), (w2, h2))
    # return diff_x, diff_y, ratio, rotate_angle, x1, y1, w1, h1, x2, y2, w2, h2
    # diff_x, diff_y, ratio, rotate_angle, x1, y1, w1, h1, x2, y2, w2, h2 = vector_difference(v1, v2)

    # print(f"From [p1({x1}, {y1}), v1({w1}, {h1})] To [p2({x2}, {y2}), v2({w2}, {h2})]")
    # print(f"diff_x: {diff_x:.2f}, diff_y: {diff_y:.2f}, ratio: {ratio:.2f}, angle: {rotate_angle:.2f}")
    # # print(f"distance: {distance:.2f}, diff_x: {diff_x:.2f}, diff_y: {diff_y:.2f}, ratio: {ratio:.2f}, angle: {rotate_angle:.2f}")

    width, height = clip.size
    center_x = int(width / 2)
    center_y = int(height / 2)

    # 1) moving
    clip = clip.on_color(size=(width, height), color=(0, 0, 0), pos=(-diff_x, -diff_y))

    # 2) rotation
    distance_from_center = dist(x1 - center_x, y1 - center_y)
    if distance_from_center == 0:
        rotated_move_x = 0
        rotated_move_y = 0
    else:
        opposite_side = distance_from_center * sin(radians((abs(rotate_angle) / 2)))
        near_side = distance_from_center * cos(radians((abs(rotate_angle) / 2)))
        solution = find_intersection(x1, y1, opposite_side, center_x, center_y, near_side)
        # print(f"Signed angle between vectors: {rotate_angle:.2f} degrees")
        # print(solution)
        # print(f"distance_from_center: {distance_from_center:.2f} opposite_side: {opposite_side:.2f}, near_side: {near_side:.2f}")
        # print(solution)
        # print(f"rotated_move_x: {rotated_move_x:.2f}, rotated_move_x: {rotated_move_y:.2f}")

        i = 0 if rotate_angle > 0 else 1
        if len(solution) == 1:
            i = 0
        rotated_move_x = (solution[i][0] - x1) * 2
        rotated_move_y = (solution[i][1] - y1) * 2

    clip = clip.rotate(rotate_angle)

    rotated_window_width, rotated_window_height = clip.size
    # print(f"rotated window size: {rotated_window_width} x {rotated_window_height}")
    window_move_x = abs(rotated_window_width - width) / 2
    window_move_y = abs(rotated_window_height - height) / 2

    # 3) resize
    clip = clip.resize(ratio)

    scaled_move_x = x1 * (ratio - 1)
    scaled_move_y = y1 * (ratio - 1)

    # 4) total correction
    total_move_x = (rotated_move_x + window_move_x) * ratio + scaled_move_x
    total_move_y = (rotated_move_y + window_move_y) * ratio + scaled_move_y
    # print(f"rotated_move_x: {rotated_move_x:.1f} + window_move_x: {window_move_x:.1f} + scaled_move_x: {scaled_move_x:.1f} = total_move_x: {total_move_x:.1f}")
    # print(f"rotated_move_y: {rotated_move_y:.1f} + window_move_y: {window_move_y:.1f} + scaled_move_y: {scaled_move_y:.1f} = total_move_y: {total_move_y:.1f}")

    clip = clip.on_color(size=(width, height), color=(0, 0, 0), pos=(-total_move_x, -total_move_y))
    clip = clip.resize(newsize=(width, height))

    return clip

def line_to_vector(line1, line2):
    x1, y1, x12, y12 = line1
    x2, y2, x22, y22 = line2
    v1 = (x1, y1, x12 - x1, y12 - y1)
    v2 = (x2, y2, x22 - x2, y22 - y2)
    return v1, v2


if __name__ == "__main__":
    clip1 = VideoFileClip("data/line1.mp4")
    clip2 = VideoFileClip("data/line2.mp4")

    # width, height = clip1.size
    # center_x = int(width / 2)
    # center_y = int(height / 2)
    #
    # # line1 = x1, y1, x12, y12 = (100, 100, 220, 100)
    # # line2 = x2, y2, x22, y22 = (120, 120, 200, 140)
    # # line1 = (center_x, center_y, center_x + 200, center_y)
    # # line2 = (center_x, center_y, center_x + 200, center_y + 100)
    # # prepare_sample(line1, line2)
    # # quit()

    line1 = (100, 100, 220, 100)
    line2 = (120, 120, 200, 140)
    v1, v2 = line_to_vector(line1, line2)
    x1, y1, w1, h1 = v1
    x2, y2, w2, h2 = v2
    # v1 = (x1, y1, x12 - x1, y12 - y1)
    # v2 = (x2, y2, x22 - x2, y22 - y2)
    # a = 720 / 1080
    # v1 = (int(788*a), int(229*a), int(395*a), int(395*a))
    # v2 = (int(790*a), int(233*a), int(406*a), int(406*a))
    # x1, y1, w1, h1 = v1
    # x2, y2, w2, h2 = v2
    # line1 = (x1, y1, x1 + w1, y1 + h1)
    # line2 = (x2, y2, x2 + w2, y2 + h2)
    # prepare_sample(line1, line2)

    # line1 = (center_x, center_y, center_x + 200, center_y)
    # line2 = (center_x, center_y, center_x + 200, center_y + 100)
    # v1, v2 = line_to_vector(line1, line2)

    ratio = 2.0
    v3 = x1, y1, w1*ratio, h1*ratio
    temp_clip = get_adjusted_clip(clip2, v1, v2)
    adjust_clip1 = get_adjusted_clip(temp_clip, v3, v1)
    adjust_clip2 = get_adjusted_clip(clip2, v3, v2)

    final_clip = concatenate_videoclips([clip1, adjust_clip1, adjust_clip2])
    final_clip.write_videofile("data/adjust_video.mp4", codec='libx264', audio_codec='aac')

    clip1.close()
    clip2.close()
    adjust_clip1.close()
    adjust_clip2.close()
    final_clip.close()
