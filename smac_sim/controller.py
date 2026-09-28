"""軌跡分段與 Pure Pursuit 追蹤控制（簡報 p.2「三、追蹤控制器與路徑分段執行」）。

狀態機：START -> TRACKING_SMAC_SEGMENT（依序追蹤 Smac 子段，處理倒車／轉向／對位）
        -> TRACKING_MAIN（追蹤主路徑）-> COMPLETED
"""
import math
from dataclasses import dataclass

import numpy as np

from .geometry import norm_angle

START = 'START'
TRACKING_SMAC_SEGMENT = 'TRACKING_SMAC_SEGMENT'
TRACKING_MAIN = 'TRACKING_MAIN'
COMPLETED = 'COMPLETED'


def split_path_into_segments(path):
    """依 move_dir（+1 前進 / -1 倒車）切分路徑，方向改變處開始新的一段。

    path: [(x, y, theta, move_dir), ...]；尖點同時是前一段終點與下一段起點。
    回傳 [(xy ndarray (N, 2), move_dir), ...]。
    """
    segments = []
    cur = [path[0]]
    for p in path[1:]:
        if p[3] != cur[-1][3]:
            segments.append(cur)
            cur = [tuple(cur[-1][:3]) + (p[3],), p]
        else:
            cur.append(p)
    segments.append(cur)
    return [(np.array([[q[0], q[1]] for q in seg]), int(seg[-1][3])) for seg in segments]


@dataclass
class PurePursuit:
    lookahead: float = 0.2
    default_speed: float = 0.15
    k_alpha: float = 2.5
    w_max: float = 1.0

    def command(self, pose, xy, move_dir):
        """① 最近路徑點 ② 往後找第一個距離 >= lookahead 的預瞄點 ③ 夾角 α ④ (v, w)。"""
        x, y, th = pose
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        nearest = int(np.argmin(d))
        target = len(xy) - 1
        for k in range(nearest, len(xy)):
            if d[k] >= self.lookahead:
                target = k
                break
        tx, ty = xy[target]
        heading = th if move_dir > 0 else th + math.pi          # 倒車時以車尾方向為行進方向
        alpha = norm_angle(math.atan2(ty - y, tx - x) - heading)
        v = self.default_speed * move_dir
        w = float(np.clip(self.k_alpha * alpha, -self.w_max, self.w_max))
        return v, w


class TrackingSimulator:
    """差速車運動學 x' = v cos θ, y' = v sin θ, θ' = w 的離散模擬。"""

    def __init__(self, controller, dt=0.1, segment_tolerance=0.15, goal_tolerance=0.1, max_steps=4000):
        self.ctrl = controller
        self.dt = dt
        self.segment_tolerance = segment_tolerance
        self.goal_tolerance = goal_tolerance
        self.max_steps = max_steps              # 每一段的步數上限

    def run(self, start_pose, smac_path, main_path):
        x, y, th = start_pose
        traj = [(x, y, th)]
        states = [START]
        segments = split_path_into_segments(smac_path) if len(smac_path) > 1 else []
        phases = [(xy, d, TRACKING_SMAC_SEGMENT, self.segment_tolerance) for xy, d in segments]
        phases.append((np.asarray(main_path, dtype=float), 1, TRACKING_MAIN, self.goal_tolerance))
        completed = True
        for xy, move_dir, state, tol in phases:
            for _ in range(self.max_steps):
                if math.hypot(xy[-1, 0] - x, xy[-1, 1] - y) < tol:
                    break
                v, w = self.ctrl.command((x, y, th), xy, move_dir)
                x += v * math.cos(th) * self.dt
                y += v * math.sin(th) * self.dt
                th = norm_angle(th + w * self.dt)
                traj.append((x, y, th))
                states.append(state)
            else:
                completed = False                   # 該段超過步數上限仍未到達
        return np.array(traj), states, len(segments), completed
