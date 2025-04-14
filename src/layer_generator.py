# coding:utf-8
# @author: Xiao Shanghua
# @file: layer_generator.py
# @time: 2025/4/14 9:53
# @desc:

from src.room_generator import Room
from src.vars import Vector2
from config import ROOM_RANGE

import random
import numpy as np


class Layer:
    def __init__(self, width, height, room_count):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.room_count = room_count
        self.room_list = []

        self.room_max_size = ROOM_RANGE[1]
        self.room_min_size = ROOM_RANGE[0]

    def generate_rooms(self):
        for i in range(self.room_count):
            # if len(self.room_list) > 0:
            #     index = random.randint(0, len(self.room_list) - 1)
            #     offset_x = (-1 if random.randint(-10, 10) > 0 else 1) * random.randint(self.room_min_size, self.room_max_size)
            #     offset_y = (-1 if random.randint(-10, 10) > 0 else 1) * random.randint(self.room_min_size, self.room_max_size)
            #     x = self.room_list[index].center.x + offset_x
            #     y = self.room_list[index].center.y + offset_y
            # else:
            #     x = random.randint(self.room_max_size // 2, self.width - self.room_max_size // 2)
            #     y = random.randint(self.room_max_size // 2, self.height - self.room_max_size // 2)

            x = random.randint(self.width // 2 - self.room_max_size, self.width // 2 + self.room_max_size)
            y = random.randint(self.width // 2 - self.room_max_size, self.height // 2 + self.room_max_size)

            center = Vector2(x, y)
            width = random.randint(self.room_min_size // 2, self.room_max_size // 2) * 2
            height = random.randint(self.room_min_size // 2, self.room_max_size // 2) * 2
            room = Room(width, height, center)
            room.generate()
            self.room_list.append(room)

    def apply_separation_steering_behaviour(self):

        def find_neighbors_within_range(room_list, target: Room, within_range: float):
            neighbor_list = []
            for other in room_list:
                if target.overlaps(other):
                    neighbor_list.append(other)
            return neighbor_list

        def calculate_separation_factor(room_list, target, sep_distance):
            # offset_x = 0
            # offset_y = 0
            # for other in room_list:
            #     offset_x += (-1 if other.center.x > target.center.x else 1) * ((other.width + target.width) - abs(other.x1 - target.x0))
            #     offset_y += (-1 if other.center.y > target.center.y else 1) * ((other.height + target.height) - abs(other.y1 - target.y0))
            # return int(offset_x), int(offset_y)

            force = Vector2(0, 0)
            for other in room_list:
                if other == target:
                    continue
                if not target.overlaps(other, 0.1):
                    continue
                diff = target.center - other.center
                distance = diff.length()
                if distance < sep_distance:
                    eps = 1e-5
                    force += diff.normalize() * (sep_distance - distance) * (1. / max(eps, distance))
            return force

        def calculate_alignment_factor(room_list, target, radius):
            avg_velocity = Vector2(0, 0)
            count = 0
            for other in room_list:
                if other is target:
                    continue
                if (target.center - other.center).length() < radius:
                    avg_velocity += other.velocity
                    count += 1
            if count > 0:
                avg_velocity *= (1. / count)
                steering = avg_velocity - target.velocity
                return steering
            return Vector2(0, 0)

        def calculate_coherent_factor(room_list, target, radius):
            center_mass = Vector2(0, 0)
            count = 0
            for other in room_list:
                if other == target:
                    continue
                if other.too_close(other):
                    continue
                if (target.center - other.center).length() < radius:
                    center_mass += other.center
                    count += 1
            if count > 0:
                center_mass *= (1. / count)
                steering = center_mass - target.center
                return steering
            return Vector2(0, 0)

        epoch = 20
        for e in range(epoch):
            overlap_flag = [False for _ in self.room_list]
            for idx, room in enumerate(self.room_list):
                # neighbors = find_neighbors_within_range(self.room_list, room, self.room_max_size * 2)
                # if not neighbors:
                #     overlap_flag[idx] = True
                #     continue
                offset_sep = calculate_separation_factor(self.room_list, room, 80)
                offset_ali = calculate_alignment_factor(self.room_list, room, 40)
                offset_coh = calculate_coherent_factor(self.room_list, room, 40)

                force = offset_sep * 0.8 + offset_ali * 0 + offset_coh * 0.4
                max_force = 10
                if force.length() > max_force:
                    force = force.normalize() * max_force

                room.velocity = force

                print(f'room {room.center}, velo: {force}, sep: {offset_sep}, ali: {offset_ali}, coh: {offset_coh}')
                # room.center += (offset_sep * 0.8 + offset_coh * 0.2)
                room.center += room.velocity

            if all(overlap_flag):
                break

    def merge_rooms(self):
        for room in self.room_list:
            try:
                self.grid[
                    int(room.center.x-room.width//2): int(room.center.x+room.width//2),
                    int(room.center.y-room.height//2): int(room.center.y+room.height//2)
                ] = room.grid
            except Exception as e:
                pass

    def generate(self):
        self.generate_rooms()
        self.apply_separation_steering_behaviour()
        self.merge_rooms()
        return self.grid

    def test(self):
        from src.draw_map import visualize_enum_matrix
        self.generate_rooms()
        for i in range(20):
            self.apply_separation_steering_behaviour()
            self.grid = np.zeros((self.width, self.height))
            self.merge_rooms()
            visualize_enum_matrix(self.grid)
        return self.grid


if __name__ == '__main__':
    from config import CANVAS_SIZE, ROOM_RANGE, ROOM_NUMBER
    from src.draw_map import visualize_enum_matrix

    layer = Layer(200, 200, 20)
    g = layer.generate()
    visualize_enum_matrix(g)
