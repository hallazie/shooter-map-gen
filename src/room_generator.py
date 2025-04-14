# coding:utf-8
# @author: Xiao Shanghua
# @file: room_generator.py
# @time: 2025/4/14 9:53
# @desc:

from src.vars import TileType, Vector2

import numpy as np


class Room:
    def __init__(self, width: int, height: int, center: Vector2):
        self._width = width
        self._height = height
        self._center = center
        self.grid = np.ones((width, height)) * int(TileType.GROUND.value)
        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0
        self._update_xxyy()
        self.velocity = Vector2(0, 0)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self._update_xxyy()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._update_xxyy()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._update_xxyy()

    def _update_xxyy(self):
        self.x0 = self._center.x - self._width // 2
        self.x1 = self.x0 + self._width
        self.y0 = self._center.y - self._height // 2
        self.y1 = self.y0 + self._height

    def init_basic(self):
        for i in range(self.width):
            for j in range(self.height):
                if i == 0 or j == 0 or i == self.width - 1 or j == self.height - 1:
                    self.grid[i, j] = TileType.WALL.value

    def overlaps(self, other: 'Room', allow_overlap_ratio: float = 0, exclude_wall: bool = True):
        if exclude_wall:
            x_overlap = max(0, min(self.x1 - 1, other.x1 - 1) - max(self.x0 + 1, other.x0 + 1))
            y_overlap = max(0, min(self.y1 - 1, other.y1 - 1) - max(self.y0 + 1, other.y0 + 1))
        else:
            x_overlap = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
            y_overlap = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        a_overlap = x_overlap * y_overlap
        if a_overlap / min(self.area(), other.area()) > allow_overlap_ratio:
            return True
        else:
            return False

    def border(self, other):
        if self.overlaps(other):
            return True
        if self.x0 == other.x1 or self.x1 == other.x0:
            return True
        elif self.y0 == other.y1 or self.y1 == other.y0:
            return True
        else:
            return False

    def too_close(self, other: 'Room'):
        close_x = abs(self.center.x - other.center.x) < (self.width + other.width)
        close_y = abs(self.center.y - other.center.y) < (self.height + other.height)
        return close_x or close_y

    def area(self):
        return self.width * self.height

    def generate(self):
        self.init_basic()
        return self.grid


