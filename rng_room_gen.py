# coding:utf-8
# @author: Xiao Shanghua
# @file: rng_room_gen.py
# @time: 2025/4/16 18:15
# @desc:

from matplotlib.collections import LineCollection

import random
import math
import numpy as np
import matplotlib.pyplot as plt


def compute_rng(points):
    """计算相对邻域图的边"""
    n = points.shape[0]
    distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            d_ij = distances[i, j]
            mask = np.ones(n, dtype=bool)
            mask[[i, j]] = False
            cond_i = distances[i] < d_ij
            cond_j = distances[j] < d_ij
            valid_k = np.where(mask & cond_i & cond_j)[0]
            if not valid_k.size:
                edges.append((i, j))
    return edges


def generate_rooms(points, edges, width, height):
    """基于RNG图生成房间的边界"""
    n = points.shape[0]
    adjacency = [[] for _ in range(n)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    rooms = []
    for i in range(n):
        x, y = points[i]
        # 处理x轴左右边界
        left_nodes = [points[j] for j in adjacency[i] if points[j][0] < x]
        right_nodes = [points[j] for j in adjacency[i] if points[j][0] > x]
        left = (x + max(p[0] for p in left_nodes)) / 2 if left_nodes else x - 0.5
        right = (x + min(p[0] for p in right_nodes)) / 2 if right_nodes else x + 0.5

        # 处理y轴上下边界
        down_nodes = [points[j] for j in adjacency[i] if points[j][1] < y]
        up_nodes = [points[j] for j in adjacency[i] if points[j][1] > y]
        bottom = (y + max(p[1] for p in down_nodes)) / 2 if down_nodes else y - 0.5
        top = (y + min(p[1] for p in up_nodes)) / 2 if up_nodes else y + 0.5

        left = math.floor(left)
        right = math.ceil(right)
        top = math.ceil(top)
        bottom = math.floor(bottom)
        left = int(left)
        right = int(right)
        top = int(top)
        bottom = int(bottom)
        left = left if left >= 0 else 0
        right = right if right < width else width - 1
        top = top if top >= 0 else 0
        bottom = bottom if bottom < height else height - 1

        rooms.append((left, right, bottom, top))
    return rooms


def plot_combined(points, edges, rooms):
    """组合可视化RNG图和房间布局"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制RNG边
    rng_segments = [[points[i], points[j]] for i, j in edges]
    lc_rng = LineCollection(rng_segments, colors='gray', linewidths=1, zorder=1)
    ax.add_collection(lc_rng)

    # 收集所有唯一墙体线段
    wall_segments = set()
    for room in rooms:
        left, right, bottom, top = room
        # 左墙
        seg = ((left, bottom), (left, top))
        wall_segments.add(tuple(sorted(seg)))
        # 右墙
        seg = ((right, bottom), (right, top))
        wall_segments.add(tuple(sorted(seg)))
        # 底墙
        seg = ((left, bottom), (right, bottom))
        wall_segments.add(tuple(sorted(seg)))
        # 顶墙
        seg = ((left, top), (right, top))
        wall_segments.add(tuple(sorted(seg)))

    # 转换线段格式并绘制
    walls = [np.array(seg) for seg in wall_segments]
    lc_walls = LineCollection(walls, colors='black', linewidths=2, zorder=2)
    ax.add_collection(lc_walls)

    # 绘制节点
    ax.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=3)

    ax.set_title("Connected Rooms with RNG")
    plt.axis('equal')
    plt.show()


def generate_random_points(width=16, height=10, portion=0.1):
    """生成稀疏化网格点"""
    width -= 2
    height -= 2
    index_list = [i for i in range(width * height)]
    random.shuffle(index_list)
    index_list = index_list[:int(width * height * portion)]
    coord_list = np.array([(i // width, i % width) for i in index_list], dtype=np.float64)
    for i in range(len(coord_list)):
        coord = coord_list[i]
        for j in range(i, len(coord_list)):
            if i == j:
                continue
            other = coord_list[j]
            if coord[0] > other[0] and coord[0] - other[0] <= 1:
                coord_list[j][0] += 1
            elif coord[0] < other[0] and other[0] - coord[0] <= 1:
                coord_list[j][0] -= 1
            if coord[1] > other[1] and coord[1] - other[1] <= 1:
                coord_list[j][1] += 1
            elif coord[1] < other[1] and other[1] - coord[1] <= 1:
                coord_list[j][1] -= 1
    for i in range(len(coord_list)):
        coord_list[i][0] += 1
        coord_list[i][1] += 1
    return coord_list


def expand_non_connected_rooms(points, edges, rooms):
    return rooms


def room_to_blocks(rooms):
    blocks = [[] for _ in range(len(rooms))]
    for i in range(len(rooms)):
        room = rooms[i]
        left, right, bottom, top = room
        for x in range(left, right):
            for y in range(bottom, top):
                blocks[i].append((x, y))
    return blocks


def resolve_blocks_overlapping(block_list):
    occupied = []
    block_resolve_list = []
    for block in block_list:
        block_resolve = []
        for pos in block:
            if pos not in occupied:
                block_resolve.append(pos)
        block_resolve_list.append(block_resolve)
    return block_resolve_list


def refine_blocks(block_list):
    refined_list = []
    for block in block_list:
        refined = refine_room_with_walls(block)
        refined_list.append(refined)
    return refined_list


def refine_room_with_walls(original_cells):
    """将房间细化为4x4网格并提取边界墙"""
    # 生成所有细化后的单元格
    refined = set()
    for x, y in original_cells:
        for dx in range(4):
            for dy in range(4):
                refined.add((x * 4 + dx, y * 4 + dy))

    # 检测原始边界
    walls = set()
    grounds = set()
    for x, y in refined:
        # 检查四个方向的邻居
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        borders = [
            ((x, y), (x + 1, y)),  # 右墙
            ((x, y), (x, y + 1)),  # 上墙
            ((x, y), (x - 1, y)),  # 左墙 (转换为相邻单元格的左墙)
            ((x, y), (x, y - 1))  # 下墙 (转换为相邻单元格的上墙)
        ]

        for i, (nx, ny) in enumerate(neighbors):
            if (nx, ny) not in refined:
                # 标准化墙体方向
                a, b = sorted([borders[i][0], borders[i][1]])
                # walls.add((a, b))
                walls.add(a)

    # # 合并相邻墙体
    # merged = []
    # horizontal = {}  # key: y坐标，value: (min_x, max_x)
    # vertical = {}  # key: x坐标，value: (min_y, max_y)
    #
    # for (x1, y1), (x2, y2) in walls:
    #     if y1 == y2:  # 水平墙
    #         y = y1
    #         x_start = min(x1, x2)
    #         x_end = max(x1, x2)
    #         if y not in horizontal:
    #             horizontal[y] = []
    #         horizontal[y].append((x_start, x_end))
    #     else:  # 垂直墙
    #         x = x1
    #         y_start = min(y1, y2)
    #         y_end = max(y1, y2)
    #         if x not in vertical:
    #             vertical[x] = []
    #         vertical[x].append((y_start, y_end))
    #
    # # 合并水平墙
    # for y in horizontal:
    #     segments = sorted(horizontal[y])
    #     current_start, current_end = segments[0]
    #     for s, e in segments[1:]:
    #         if s <= current_end:
    #             current_end = max(current_end, e)
    #         else:
    #             merged.append(((current_start, y), (current_end, y)))
    #             current_start, current_end = s, e
    #     merged.append(((current_start, y), (current_end, y)))
    #
    # # 合并垂直墙
    # for x in vertical:
    #     segments = sorted(vertical[x])
    #     current_start, current_end = segments[0]
    #     for s, e in segments[1:]:
    #         if s <= current_end:
    #             current_end = max(current_end, e)
    #         else:
    #             merged.append(((x, current_start), (x, current_end)))
    #             current_start, current_end = s, e
    #     merged.append(((x, current_start), (x, current_end)))
    #
    # return merged
    return refined, walls


def run():
    # 生成随机点集
    width = 20
    height = 10
    points = generate_random_points(width=width, height=height)

    # 计算RNG边
    edges = compute_rng(points)

    # 生成房间布局
    rooms = generate_rooms(points, edges, width, height)
    # rooms = []

    rooms = expand_non_connected_rooms(points, edges, rooms)

    from src.draw_map import draw_basic_color_blocks, draw_basic_rooms
    blocks = room_to_blocks(rooms)
    blocks = resolve_blocks_overlapping(blocks)
    refined = refine_blocks(blocks)
    draw_basic_rooms(width * 4, height * 4, refined)
    # draw_basic_color_blocks(width, height, blocks)

    # 组合可视化
    plot_combined(points, edges, rooms)


if __name__ == '__main__':
    run()
