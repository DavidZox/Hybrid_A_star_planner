import math


def norm_angle(a):
    """把角度正規化到 [-pi, pi)。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
