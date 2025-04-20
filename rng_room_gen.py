# coding:utf-8
# @author: Xiao Shanghua
# @file: rng_room_gen.py
# @time: 2025/4/16 18:15
# @desc:

from matplotlib.collections import LineCollection
from collections import defaultdict

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


def has_adjacent_edge(A, B, min_length=2):
    """
    判断两个矩形是否存在至少一条重叠或邻接的边，且接触长度≥指定值
    :param A: 矩形A的(left, right, bottom, top)
    :param B: 矩形B的(left, right, bottom, top)
    :param min_length: 要求的最小接触长度，默认4
    :return: 布尔值表示是否满足条件
    """
    # 解包矩形坐标
    a_l, a_r, a_b, a_t = A
    b_l, b_r, b_b, b_t = B

    # 检查垂直边对齐（左右边）
    vertical_checks = [
        (a_l, b_l),  # A左 vs B左
        (a_l, b_r),  # A左 vs B右
        (a_r, b_l),  # A右 vs B左
        (a_r, b_r)  # A右 vs B右
    ]

    for x1, x2 in vertical_checks:
        if x1 == x2:
            # 计算y轴重叠范围
            y_overlap_start = max(a_b, b_b)
            y_overlap_end = min(a_t, b_t)
            if y_overlap_end - y_overlap_start >= min_length:
                return True

    # 检查水平边对齐（上下边）
    horizontal_checks = [
        (a_b, b_b),  # A底 vs B底
        (a_b, b_t),  # A底 vs B顶
        (a_t, b_b),  # A顶 vs B底
        (a_t, b_t)  # A顶 vs B顶
    ]

    for y1, y2 in horizontal_checks:
        if y1 == y2:
            # 计算x轴重叠范围
            x_overlap_start = max(a_l, b_l)
            x_overlap_end = min(a_r, b_r)
            if x_overlap_end - x_overlap_start >= min_length:
                return True

    return False


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


def generate_random_points(width=16, height=10, nodes=10, border_reserve=4, min_gap=4, max_gap=8):
    """生成稀疏化网格点"""
    width -= border_reserve
    height -= border_reserve
    index_list = [i for i in range(width * height)]
    point_list = [(i // width ,i % width) for i in range(width * height)]
    random.shuffle(index_list)
    # point_list = [point_list[i] for i in index_list]
    random.shuffle(point_list)
    coord_list = [(width//2, height//2)]
    print(f'adding random points: {point_list[0]}')
    for _ in range(10):
        random.shuffle(point_list)
        if len(coord_list) > nodes:
            break
        for other in point_list:
            if other in coord_list:
                continue
            if len(coord_list) > nodes:
                break
            min_valid = True
            max_valid = False
            for coord in coord_list:
                if abs(coord[0] - other[0]) + abs(coord[1] - other[1]) < max_gap * 2:
                    max_valid = True
                if abs(coord[0] - other[0]) + abs(coord[1] - other[1]) < min_gap * 2:
                    min_valid = False
                    break
            if min_valid and max_valid:
                coord_list.append(other)
    coord_list = np.array(coord_list)
    coord_list += np.array([border_reserve//2, border_reserve//2])
    return coord_list


def generate_random_points_ds(width=16, height=10, nodes=10, border_reserve=4, min_gap=4, max_gap=8):
    """生成稀疏化网格点"""
    adjusted_width = width - border_reserve
    adjusted_height = height - border_reserve

    # 生成所有可能的网格点（调整后的坐标系）
    point_list = [(y, x) for y in range(adjusted_height) for x in range(adjusted_width)]
    random.shuffle(point_list)

    # 初始化点集合：从中心点开始
    start_y = adjusted_height // 2
    start_x = adjusted_width // 2
    coord_list = [(start_y, start_x)]

    for _ in range(10):  # 最多尝试10次填充循环
        random.shuffle(point_list)
        if len(coord_list) >= nodes:
            break

        for other in point_list:
            if len(coord_list) >= nodes:
                break
            if other in coord_list:
                continue

            # 检查最小间距条件
            min_valid = True
            for coord in coord_list:
                dy = abs(coord[0] - other[0])
                dx = abs(coord[1] - other[1])
                if dy + dx <= min_gap:
                    min_valid = False
                    break
            if not min_valid:
                continue

            # 检查最大邻接条件
            max_valid = False
            for coord in coord_list:
                dy = abs(coord[0] - other[0])
                dx = abs(coord[1] - other[1])
                if dy + dx <= max_gap:
                    max_valid = True
                    break
            if max_valid:
                coord_list.append(other)

    # 将坐标调整回原始坐标系
    coord_list = np.array(coord_list) + border_reserve // 2
    return coord_list


def expand_non_connected_rooms(room_list, edge_list, point_list):
    edge_map = defaultdict(list)
    for p1, p2 in edge_list:
        edge_map[p1].append(p2)
        edge_map[p2].append(p1)
    center_list = [(int(x), int(y)) for x, y in point_list]
    for i in range(len(center_list)):
        room_border = list(room_list[i])
        room_center = center_list[i]
        for j in edge_map[i]:
            neighbor_border = list(room_list[j])
            neighbor_center = center_list[j]
            for _ in range(10):
                valid_adj = has_adjacent_edge(room_border, neighbor_border)
                # print(f'{room_center}({room_border})-{neighbor_center}({neighbor_border}) has adj border: {valid_adj}')
                if valid_adj:
                    room_list[i] = tuple(room_border)
                    break
                expand_horz = random.randint(-10, 10) > 0
                if expand_horz:
                    expand_step = 1 if room_center[0] < neighbor_center[0] else -1
                    if expand_step == 1:
                        room_border[1] += expand_step
                    else:
                        room_border[0] += expand_step
                else:
                    expand_step = 1 if room_center[1] < neighbor_center[1] else -1
                    if expand_step == 1:
                        room_border[3] += expand_step
                    else:
                        room_border[2] += expand_step
    return room_list


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
    block_list = sorted(block_list, key=lambda x: len(x), reverse=False)
    for block in block_list:
        block_resolve = []
        for pos in block:
            if pos not in occupied:
                occupied.append(pos)
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
                walls.add((x, y))

    shifted = set()
    for x, y in walls:
        y += 1 if y % 2 != 0 else 0
        x += 1 if x % 2 != 0 else 0
        shifted.add((x, y))
    fill = set()
    for x, y in shifted:
        if (x+2, y) in shifted:
            fill.add((x+1, y))
        if (x, y+2) in shifted:
            fill.add((x, y+1))
    shifted |= fill

    return refined, shifted


def run():
    # 生成随机点集
    width = 48
    height = 48
    points = generate_random_points(width=width, height=height, nodes=12, border_reserve=4)

    # 计算RNG边
    edges = compute_rng(points)

    # 生成房间布局
    rooms = generate_rooms(points, edges, width, height)
    # rooms = expand_non_connected_rooms(rooms, edges, points)
    # rooms = []

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
