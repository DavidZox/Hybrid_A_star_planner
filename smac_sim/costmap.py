"""2D 代價地圖（簡報 p.2「一、Costmap」）。

- 網格化離散空間：cell (i, j) 對應世界座標 x = x_min + i * resolution, y = y_min + j * resolution
- 距離場與膨脹半徑：
    碰撞區  dist <= radius                         -> cost = 200
    緩衝區  radius < dist <= radius + inflation     -> cost = 100 * (1 - (dist - radius) / inflation)
    安全區  dist > radius + inflation               -> cost = 0
- 車輛碰撞檢驗（圓形包覆模型）：d <= R_robot + R_obstacle 即碰撞
"""
import math

import numpy as np

LETHAL_COST = 200.0
INFLATED_COST = 100.0


class Costmap2D:
    def __init__(self, x_min=-0.5, y_min=-1.5, width=6.0, height=4.0, resolution=0.05):
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.resolution = float(resolution)
        self.nx = int(round(width / resolution))
        self.ny = int(round(height / resolution))
        self.grid = np.zeros((self.ny, self.nx))          # grid[j, i]
        self.obstacles = []                               # (x, y, radius, inflation_radius)

    @property
    def extent(self):
        return [self.x_min, self.x_min + self.nx * self.resolution,
                self.y_min, self.y_min + self.ny * self.resolution]

    def add_obstacle(self, x, y, radius, inflation_radius):
        self.obstacles.append((float(x), float(y), float(radius), float(inflation_radius)))
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        wx = self.x_min + ii * self.resolution
        wy = self.y_min + jj * self.resolution
        dist = np.hypot(wx - x, wy - y)
        cost = np.where(dist <= radius, LETHAL_COST,
                        np.where(dist <= radius + inflation_radius,
                                 INFLATED_COST * (1.0 - (dist - radius) / inflation_radius), 0.0))
        self.grid = np.maximum(self.grid, cost)

    def world_to_map(self, x, y):
        return (int(math.floor((x - self.x_min) / self.resolution)),
                int(math.floor((y - self.y_min) / self.resolution)))

    def get_cost(self, x, y):
        """查詢 cost(x, y)；地圖外視為碰撞區。"""
        i, j = self.world_to_map(x, y)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return float(self.grid[j, i])
        return LETHAL_COST

    def is_collision(self, x, y, robot_radius):
        for ox, oy, r, _ in self.obstacles:
            if math.hypot(x - ox, y - oy) <= r + robot_radius:
                return True
        return False

    def clearance(self, x, y, robot_radius=0.0):
        """機器人外緣到最近障礙物表面的距離（安全距離評估），無障礙物時回傳 inf。"""
        if not self.obstacles:
            return math.inf
        return min(math.hypot(x - ox, y - oy) - r - robot_radius for ox, oy, r, _ in self.obstacles)
