# coding:utf-8
# @author: Xiao Shanghua
# @file: rng_room_gen.py
# @time: 2025/4/16 18:15
# @desc:

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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


def generate_rooms(points, edges):
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

        rooms.append((left, right, bottom, top))
    return rooms


def generate_rooms_by_expansion(points, edges):
    pass


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
    index_list = [i for i in range(width * height)]
    random.shuffle(index_list)
    index_list = index_list[:int(width * height * portion)]
    return np.array([(i // width, i % width) for i in index_list], dtype=np.float64)


def run():
    # 生成随机点集
    points = generate_random_points()

    # 计算RNG边
    edges = compute_rng(points)

    # 生成房间布局
    # rooms = generate_rooms(points, edges)
    rooms = []

    # 组合可视化
    plot_combined(points, edges, rooms)


if __name__ == '__main__':
    run()
