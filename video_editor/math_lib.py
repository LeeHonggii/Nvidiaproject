from math import radians, degrees, sin, cos, acos
from sympy import symbols, Eq, solve

# 내적 계산 함수
dot_product = lambda v1, v2: v1[0] * v2[0] + v1[1] * v2[1]
cross_product = lambda v1, v2: v1[0] * v2[1] - v1[1] * v2[0]
vlength = lambda v: (v[0]**2 + v[1]**2)**(1/2)

def calculate_signed_angle(v1, v2):
    dot_prod = dot_product(v1, v2)
    length_v1 = vlength(v1)
    length_v2 = vlength(v2)

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

# # Example usage
# x1, y1, a = 0, 0, 5
# x2, y2, b = 4, 0, 3
#
# solutions = find_intersection(x1, y1, a, x2, y2, b)
# print("Solutions:", solutions)


def find_other_sides(x, a, b):
    y = x * cos(radians(a))
    z = x * sin(radians(b))
    return y, z


# # 다른 두 변의 길이 구하기
# y, z = find_other_sides(x, a, b)
#
# print(f"첫 번째 변의 길이 (인접변): {y}")
# print(f"두 번째 변의 길이 (반대변): {z}")
