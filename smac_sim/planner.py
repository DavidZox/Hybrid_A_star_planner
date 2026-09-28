"""Smac Hybrid-A* 規劃器（含 ROI 限制與解析擴展），對應簡報 p.2「二、Smac Hybrid-A* 規劃器與 ROI 區域限制」。

- ROI：x_min = min(x_start, x_goal) - margin …，只在 ROI 內搜尋
- 狀態 (x, y, θ) ∈ SE(2)，封閉集以 (x/0.05, y/0.05, θ/8°) 離散化
- 6 種運動原語：前進/倒車 × {直行, 左 12°, 右 12°}
    new_θ = θ + steer * move_dir
    new_x = x + move_dir * step_size * cos(new_θ)
    new_y = y + move_dir * step_size * sin(new_θ)
- 累積代價 g = g_parent + 步長 + 地圖代價 + 倒車懲罰 + 換檔懲罰 + 打角懲罰
- 啟發函數 h = 到終點距離 + 0.8 * |Δθ_goal|，f = g + h
- 終點條件：dist < 0.1 且 angle_diff < 45° 且 goal_angle_diff < 45°，
  之後以解析擴展（直線）連到終點姿態；直線段必須無碰撞
- 迭代數超過 max_iterations 時，回傳目前 h 最小節點的路徑（best-effort）

註：原始程式已遺失，以上細節中「代價的組合方式、goal_angle_diff 的定義、max_iterations」
是依簡報 11 張結果圖反推而得，推導過程見 README。
"""
import heapq
import math
import time
from dataclasses import dataclass, field

from .geometry import norm_angle


@dataclass
class PlannerParams:
    step_size: float = 0.06
    steer_angle_deg: float = 12.0
    angle_bin_deg: float = 8.0
    xy_resolution: float = 0.05
    reverse_penalty: float = 0.3
    change_dir_penalty: float = 0.8
    steering_penalty: float = 0.1
    cost_penalty: float = 0.01          # 地圖代價權重：g += cost_penalty * cost(x, y)
    heading_weight: float = 0.8         # h 中角度差的權重
    goal_dist_tolerance: float = 0.1
    goal_angle_tolerance_deg: float = 45.0
    roi_margin: float = 1.5
    robot_radius: float = 0.22
    max_iterations: int = 30000


@dataclass
class Node:
    x: float
    y: float
    theta: float
    g: float
    direction: int                      # +1 前進、-1 倒車、0 起點
    parent: "Node" = None


@dataclass
class PlanResult:
    path: list                          # [(x, y, theta, direction), ...]
    found: bool                         # True：滿足終點條件；False：迭代上限後的 best-effort 路徑
    iterations: int
    plan_time: float
    roi: tuple
    expanded: list = field(default_factory=list)   # 依展開順序的 (x, y, theta, direction)
    end_node_h: float = 0.0


class SmacHybridAStar:
    def __init__(self, costmap, params=None):
        self.costmap = costmap
        self.p = params or PlannerParams()

    # ---- 基本元件 -------------------------------------------------------------
    def compute_roi(self, start, goal):
        m = self.p.roi_margin
        return (min(start[0], goal[0]) - m, max(start[0], goal[0]) + m,
                min(start[1], goal[1]) - m, max(start[1], goal[1]) + m)

    def state_key(self, x, y, theta):
        r = self.p.xy_resolution
        return (math.floor(x / r), math.floor(y / r),
                math.floor(norm_angle(theta) / math.radians(self.p.angle_bin_deg)))

    def heuristic(self, x, y, theta, goal):
        return (math.hypot(goal[0] - x, goal[1] - y)
                + self.p.heading_weight * abs(norm_angle(goal[2] - theta)))

    def step_cost(self, parent, x, y, direction, steer):
        p = self.p
        cost = p.step_size                                     # 基礎步長成本
        if direction < 0:
            cost *= 1.0 + p.reverse_penalty                     # 倒車懲罰
        if steer != 0.0:
            cost += p.steering_penalty * p.step_size            # 打角懲罰
        if parent.direction != 0 and direction != parent.direction:
            cost += p.change_dir_penalty                        # 換檔懲罰（尖點）
        cost += p.cost_penalty * self.costmap.get_cost(x, y)    # 地圖代價懲罰
        return cost

    def reached_goal(self, node, goal):
        p = self.p
        tol = math.radians(p.goal_angle_tolerance_deg)
        dist = math.hypot(node.x - goal[0], node.y - goal[1])
        angle_diff = abs(norm_angle(node.theta - goal[2]))                     # 車頭與終點朝向
        travel_heading = node.theta if node.direction >= 0 else node.theta + math.pi
        goal_angle_diff = abs(norm_angle(travel_heading - goal[2]))            # 行進方向與終點朝向
        return dist < p.goal_dist_tolerance and angle_diff < tol and goal_angle_diff < tol

    def analytic_expansion(self, node, goal):
        """以直線把節點接到終點；途中碰撞則回傳 None。"""
        dist = math.hypot(goal[0] - node.x, goal[1] - node.y)
        n_check = max(1, math.ceil(dist / (0.5 * self.p.step_size)))
        for i in range(1, n_check + 1):
            t = i / n_check
            if self.costmap.is_collision(node.x + t * (goal[0] - node.x), node.y + t * (goal[1] - node.y),
                                         self.p.robot_radius):
                return None
        n = max(1, math.ceil(dist / self.p.step_size))
        seg = []
        for i in range(1, n + 1):
            t = i / n
            seg.append((node.x + t * (goal[0] - node.x), node.y + t * (goal[1] - node.y),
                        goal[2] if i == n else node.theta, node.direction))
        return seg

    @staticmethod
    def reconstruct(node):
        path = []
        while node is not None:
            path.append((node.x, node.y, node.theta, node.direction if node.direction != 0 else 1))
            node = node.parent
        path.reverse()
        if len(path) > 1:                                   # 起點沿用第一步的行進方向
            path[0] = path[0][:3] + (path[1][3],)
        return path

    # ---- 搜尋 -----------------------------------------------------------------
    def plan(self, start, goal):
        p = self.p
        t0 = time.perf_counter()
        roi = self.compute_roi(start, goal)
        xmn, xmx, ymn, ymx = roi
        steers = [math.radians(-p.steer_angle_deg), 0.0, math.radians(p.steer_angle_deg)]

        start_node = Node(start[0], start[1], start[2], 0.0, 0, None)
        open_list = [(self.heuristic(*start, goal), 0, start_node)]
        closed = set()
        counter = 0
        iterations = 0
        best_node, best_h = start_node, math.inf
        expanded = []
        found_path = None

        while open_list:
            _, _, cur = heapq.heappop(open_list)
            key = self.state_key(cur.x, cur.y, cur.theta)
            if key in closed:
                continue
            closed.add(key)
            iterations += 1
            if iterations > p.max_iterations:
                iterations -= 1
                break
            expanded.append((cur.x, cur.y, cur.theta, cur.direction))

            if self.reached_goal(cur, goal):
                tail = self.analytic_expansion(cur, goal)
                if tail is not None:
                    found_path = self.reconstruct(cur) + tail
                    best_node = cur
                    break

            h_cur = self.heuristic(cur.x, cur.y, cur.theta, goal)
            if h_cur < best_h:
                best_h, best_node = h_cur, cur

            for direction in (1, -1):
                for steer in steers:
                    nt = norm_angle(cur.theta + steer * direction)
                    nx = cur.x + direction * p.step_size * math.cos(nt)
                    ny = cur.y + direction * p.step_size * math.sin(nt)
                    if not (xmn <= nx <= xmx and ymn <= ny <= ymx):
                        continue                                    # ROI 外不搜尋
                    if self.costmap.is_collision(nx, ny, p.robot_radius):
                        continue
                    ng = cur.g + self.step_cost(cur, nx, ny, direction, steer)
                    counter += 1
                    heapq.heappush(open_list, (ng + self.heuristic(nx, ny, nt, goal), counter,
                                               Node(nx, ny, nt, ng, direction, cur)))

        found = found_path is not None
        path = found_path if found else self.reconstruct(best_node)
        return PlanResult(path=path, found=found, iterations=iterations,
                          plan_time=time.perf_counter() - t0, roi=roi, expanded=expanded,
                          end_node_h=self.heuristic(best_node.x, best_node.y, best_node.theta, goal))
