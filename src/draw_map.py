import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from enum import Enum
from typing import List, Union


def visualize_enum_matrix(matrix: Union[np.ndarray, List[List[int]]],
                          enum_class: type = None,
                          title: str = "Enum Matrix Visualization",
                          color_palette: List[str] = None):
    """
    可视化枚举类型矩阵为彩色网格图，其中0值固定显示为白色

    参数:
        matrix: 二维矩阵，元素为int或enum类型
        enum_class: 可选的枚举类，用于获取枚举值和名称
        title: 图表标题
        color_palette: 可选的颜色名称列表(不包括0值)，如['maroon', 'orange', 'green', ...]
    """
    # 转换为numpy数组
    matrix = np.array(matrix)

    # 获取所有非零唯一值
    unique_values = np.unique(matrix)
    unique_values = unique_values[unique_values != 0]
    num_unique = len(unique_values)

    # 创建颜色映射
    if enum_class is not None and issubclass(enum_class, Enum):
        # 如果有枚举类，使用枚举名称作为标签
        value_to_name = {e.value: e.name for e in enum_class}
        labels = ['BACKGROUND'] + [value_to_name.get(v, str(v)) for v in unique_values]
    else:
        labels = ['0'] + [str(v) for v in unique_values]

    # 设置颜色调色板
    if color_palette is None:
        # 默认使用一组美观的Matplotlib命名颜色(不包括白色)
        default_colors = [
            'maroon', 'orange', 'green', 'royalblue', 'purple',
            'gold', 'teal', 'crimson', 'navy', 'olive',
            'orchid', 'tan', 'lime', 'indigo', 'salmon'
        ]
        # 确保有足够的颜色
        color_palette = default_colors[:num_unique]
    else:
        # 确保用户提供的调色板足够长
        if len(color_palette) < num_unique:
            raise ValueError(f"需要至少 {num_unique} 种颜色，但只提供了 {len(color_palette)} 种")

    # 在颜色列表开头插入白色(用于0值)
    colors = ['white'] + color_palette

    # 创建映射字典：值 -> 颜色索引
    value_to_index = {0: 0}
    for i, val in enumerate(unique_values, start=1):
        value_to_index[val] = i

    # 创建映射后的矩阵
    mapped_matrix = np.vectorize(lambda x: value_to_index[x])(matrix)

    # 创建颜色映射
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(len(colors) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 绘制图像
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mapped_matrix, cmap=cmap, norm=norm)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(colors)) + 0.5)
    cbar.ax.set_yticklabels(labels)

    # 添加标题和调整布局
    plt.title(title)
    plt.tight_layout()
    plt.show()