# coding:utf-8
# @author: Xiao Shanghua
# @file: basic_explore.py
# @time: 2025/4/14 9:59
# @desc:

from src.draw_map import visualize_enum_matrix
from src.layer_generator import Layer
from config import CANVAS_SIZE, ROOM_RANGE, ROOM_NUMBER


layer = Layer(CANVAS_SIZE[0], CANVAS_SIZE[1], ROOM_NUMBER)
grid = layer.generate()
visualize_enum_matrix(grid)
