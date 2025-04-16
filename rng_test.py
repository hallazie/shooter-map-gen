# coding:utf-8
# @author: Xiao Shanghua
# @file: rng_test.py
# @time: 2025/4/16 17:41
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
            # 创建掩码排除i和j自身
            mask = np.ones(n, dtype=bool)
            mask[[i, j]] = False
            # 检查是否存在k使得d_ik < d_ij 且 d_jk < d_ij
            cond_i = distances[i] < d_ij
            cond_j = distances[j] < d_ij
            valid_k = np.where(mask & cond_i & cond_j)[0]
            if not valid_k.size:
                edges.append((i, j))
    return edges


def plot_rng(points, edges):
    """可视化RNG"""
    segments = [[points[i], points[j]] for i, j in edges]
    lc = LineCollection(segments, colors='gray', linewidths=1, zorder=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=2)
    ax.autoscale()
    plt.title("Relative Neighborhood Graph")
    plt.axis('equal')
    plt.show()


def generate_random_points(width=16, height=10, portion=0.1):
    index_list = [i for i in range(width * height)]
    random.shuffle(index_list)
    index_list = index_list[:int(width * height * portion)]
    coord_list = []
    for index in index_list:
        x = index // width
        y = index % width
        coord_list.append((x, y))
    print(coord_list)
    return np.array(coord_list)


def run():
    # 生成随机点
    # np.random.seed(42)
    # points = np.random.rand(20, 2)

    points = generate_random_points()

    # 计算RNG边
    edges = compute_rng(points)

    # 可视化结果
    plot_rng(points, edges)


if __name__ == '__main__':
    run()
