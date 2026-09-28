"""把 Costmap、Smac Hybrid-A* 與 Pure Pursuit 串成一次完整模擬（簡報 p.2「總結」流程）：
感知障礙物 → Costmap → ROI → Hybrid-A* 搜尋 → 路徑分段 → Pure Pursuit 追蹤 → 抵達主路徑終點。
"""
import math

import numpy as np

from .controller import PurePursuit, TrackingSimulator
from .costmap import Costmap2D
from .metrics import path_metrics, tracking_metrics
from .planner import PlannerParams, SmacHybridAStar


def build_main_path(cfg):
    mp = cfg['main_path']
    x0, y0 = mp['start']
    x1, y1 = mp['end']
    n = int(round(math.hypot(x1 - x0, y1 - y0) / mp['spacing']))
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]


def run_scenario(cfg):
    cm_cfg = cfg['costmap']
    costmap = Costmap2D(cm_cfg['x_min'], cm_cfg['y_min'], cm_cfg['width'], cm_cfg['height'], cm_cfg['resolution'])
    ob = cfg['obstacle']
    costmap.add_obstacle(ob['x'], ob['y'], ob['radius'], ob['inflation_radius'])

    main_path = build_main_path(cfg)
    entry = main_path[0]
    goal_heading = math.atan2(main_path[1][1] - entry[1], main_path[1][0] - entry[0])
    goal = (entry[0], entry[1], goal_heading)
    st = cfg['start']
    start = (st['x'], st['y'], math.radians(st['theta_deg']))

    params = PlannerParams(**cfg['planner'])
    planner = SmacHybridAStar(costmap, params)
    plan = planner.plan(start, goal)

    tr = cfg['tracking']
    ctrl = PurePursuit(lookahead=tr['lookahead'], default_speed=tr['default_speed'],
                       k_alpha=tr['k_alpha'], w_max=tr['w_max'])
    sim = TrackingSimulator(ctrl, dt=tr['dt'], segment_tolerance=tr['segment_tolerance'],
                            goal_tolerance=tr['goal_tolerance'], max_steps=tr['max_steps_per_phase'])
    traj, states, n_segments, completed = sim.run(start, plan.path, main_path)

    metrics = dict(found=plan.found, iterations=plan.iterations, plan_time_ms=plan.plan_time * 1000.0,
                   segments=n_segments, completed=completed)
    metrics.update(path_metrics(plan.path, costmap, params.robot_radius, goal))
    metrics.update(tracking_metrics(traj, states, plan.path, main_path, costmap, params.robot_radius, tr['dt']))
    return dict(costmap=costmap, roi=plan.roi, main_path=np.array(main_path), entry=entry, goal=goal,
                start=start, smac_path=np.array(plan.path), plan=plan, trajectory=traj, states=states,
                robot_radius=params.robot_radius, dt=tr['dt'], metrics=metrics)
