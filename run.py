# coding:utf-8
# @author: Xiao Shanghua
# @file: run.py
# @time: 2025/4/14 9:53
# @desc:

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Room:
    x1: int
    y1: int
    x2: int
    y2: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def distance_to(self, other: 'Room') -> float:
        """计算两个房间中心点之间的欧氏距离"""
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        return np.sqrt(dx ** 2 + dy ** 2)

class RoomGenerator:
    def __init__(self, map_size: Tuple[int, int]):
        self.map_size = map_size
        self.grid = np.zeros(map_size, dtype=bool)  # 空间占用网格
        self.room_grid = np.zeros(map_size, dtype=int)  # 房间ID网格
        self.rooms = []
        self.next_room_id = 1

    def _mark_area(self, x1: int, y1: int, w: int, h: int, buffer: int = 1):
        """标记区域及其缓冲区为已占用"""
        x_start = max(0, x1 - buffer)
        y_start = max(0, y1 - buffer)
        x_end = min(self.map_size[0], x1 + w + buffer)
        y_end = min(self.map_size[1], y1 + h + buffer)
        self.grid[x_start:x_end, y_start:y_end] = True

    def _find_valid_position(self, w: int, h: int, min_spacing: int = 2, max_attempts: int = 100) -> Tuple[
        int, int]:
        """使用空间跳跃算法寻找有效位置"""
        for _ in range(max_attempts):
            # 优先尝试空白区域
            if len(self.rooms) > 0 and np.random.rand() < 0.7:
                # 在现有房间附近生成
                anchor = np.random.choice(self.rooms)
                offset_range = max(w, h) * 2
                x = anchor.x1 + np.random.randint(-offset_range, offset_range)
                y = anchor.y1 + np.random.randint(-offset_range, offset_range)
            else:
                # 随机生成位置
                x = np.random.randint(0, self.map_size[0] - w)
                y = np.random.randint(0, self.map_size[1] - h)

            # 检查边界和间距
            if (x + w >= self.map_size[0]) or (y + h >= self.map_size[1]):
                continue

            # 检查空间占用
            if not np.any(self.grid[x:x + w + min_spacing, y:y + h + min_spacing]):
                return x, y
        return None

    def generate_room(self, min_size: int = 20, max_size: int = 40,
                      min_spacing: int = 2, max_attempts: int = 100) -> bool:
        """生成单个房间"""
        w = np.random.randint(min_size, max_size)
        h = np.random.randint(min_size, max_size)

        if pos := self._find_valid_position(w, h, min_spacing, max_attempts):
            x, y = pos  # 创建并记录房间
            room = Room(x1=x, y1=y, x2=x + w, y2=y + h, w=w, h=h)
            self.rooms.append(room)
            self.room_grid[x:x + w, y:y + h] = self.next_room_id
            self.next_room_id += 1
            self._mark_area(x, y, w, h, buffer=min_spacing)
            return True
        return False

def generate_rooms(map_size: Tuple[int, int], num_rooms: int,
                   min_size: int = 3, max_size: int = 8,
                   min_spacing: int = 2) -> List[Room]:
    """改进的房间生成器"""
    generator = RoomGenerator(map_size)

    # 首先生成核心房间
    for _ in range(num_rooms):
        success = generator.generate_room(min_size, max_size, min_spacing)
        if not success:
            # 降低要求重试
            generator.generate_room(
                min_size=max(2, min_size - 1),
                max_size=max_size,
                min_spacing=max(1, min_spacing - 1)
            )

    # 确保最小房间数量
    while len(generator.rooms) < num_rooms * 0.8:  # 允许80%完成度
        generator.generate_room(
            min_size=2,
            max_size=6,
            min_spacing=1,
            max_attempts=500
        )

    return generator.rooms


import matplotlib.pyplot as plt


def visualize_rooms(rooms: List[Room], map_size: Tuple[int, int]):
    """可视化房间布局"""
    grid = np.zeros(map_size)
    for i, room in enumerate(rooms):
        grid[room.x1:room.x2, room.y1:room.y2] = i + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.T, cmap='tab20', interpolation='none')
    plt.colorbar(label='Room ID')
    plt.title(f"Generated {len(rooms)} Rooms")
    plt.show()


def connect_rooms(grid, room1, room2):
    # 在房间之间生成走廊
    path_x = np.linspace(room1.center_x, room2.center_x, dtype=int)
    path_y = np.linspace(room1.center_y, room2.center_y, dtype=int)
    grid[path_x, path_y] = 1


# 生成并可视化
rooms = generate_rooms(map_size=(50, 50), num_rooms=10)
visualize_rooms(rooms, (50, 50))
