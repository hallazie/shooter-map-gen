# coding:utf-8
# @author: Xiao Shanghua
# @file: vars.py
# @time: 2025/4/14 10:16
# @desc:

from enum import Enum

import math


class Vector2:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, factor: float):
        return Vector2(self.x * factor, self.y * factor)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        if self.x == 0 and self.y == 0:
            return Vector2(0, 0)
        mag = self.length()
        return Vector2(self.x / mag, self.y / mag)


class TileType(Enum):
    NULL = 0
    WALL = 1
    GROUND = 2
    GLASS = 3
    DOOR = 4
    ENV_BLOCK = 5
    ENV_HALF = 6


if __name__ == '__main__':
    a = Vector2(0, 3)
    b = Vector2(5, 6)
    print(a+b, a-b, b*4)
