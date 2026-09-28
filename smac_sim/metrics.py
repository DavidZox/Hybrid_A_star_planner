"""規劃與追蹤結果的量化指標。"""
import math

import numpy as np

from .geometry import norm_angle


def _seg_lengths(xy):
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return np.zeros(0)
    return np.hypot(*np.diff(xy, axis=0).T)


def point_to_polyline(points, poly):
    """每個點到折線的最短距離。"""
    P = np.asarray(points, dtype=float)[:, None, :]
    A = np.asarray(poly, dtype=float)[:-1][None]
    B = np.asarray(poly, dtype=float)[1:][None]
    AB = B - A
    t = np.clip(((P - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0, 1)
    proj = A + t[..., None] * AB
    return np.hypot(*(P - proj).transpose(2, 0, 1)).min(axis=1)


def path_metrics(path, costmap, robot_radius, goal):
    """path: [(x, y, theta, direction), ...]（direction: +1 前進 / -1 倒車）。"""
    P = np.asarray(path, dtype=float)
    seg = _seg_lengths(P[:, :2])
    dirs = P[1:, 3]
    headings = np.unwrap(P[:, 2])
    cusps = int(np.sum(P[1:, 3] != P[:-1, 3]))
    costs = [costmap.get_cost(x, y) for x, y in P[:, :2]]
    clear = [costmap.clearance(x, y, robot_radius) for x, y in P[:, :2]]
    end = P[-1]
    return dict(
        path_length=float(seg.sum()),
        forward_length=float(seg[dirs > 0].sum()),
        reverse_length=float(seg[dirs < 0].sum()),
        cusps=cusps,
        heading_change_deg=float(np.degrees(np.abs(np.diff(headings)).sum())),
        min_clearance=float(min(clear)),
        max_cost=float(max(costs)),
        mean_cost=float(np.mean(costs)),
        end_pos_error=float(math.hypot(end[0] - goal[0], end[1] - goal[1])),
        end_heading_error_deg=float(abs(math.degrees(norm_angle(end[2] - goal[2])))),
    )


def tracking_metrics(traj, phases, smac_path, main_path, costmap, robot_radius, dt):
    """traj: (N,3) 位姿；phases: 每一步所處的狀態字串。"""
    T = np.asarray(traj, dtype=float)
    phases = np.asarray(phases)
    smac_xy = np.asarray(smac_path, dtype=float)[:, :2]
    main_xy = np.asarray(main_path, dtype=float)
    on_smac = phases == 'TRACKING_SMAC_SEGMENT'
    on_main = phases == 'TRACKING_MAIN'
    e_smac = point_to_polyline(T[on_smac, :2], smac_xy) if on_smac.any() else np.zeros(1)
    e_main = point_to_polyline(T[on_main, :2], main_xy) if on_main.any() else np.zeros(1)
    clear = [costmap.clearance(x, y, robot_radius) for x, y in T[:, :2]]
    k_entry = int(np.argmax(on_main)) if on_main.any() else len(T) - 1
    ex, ey, eth = T[k_entry]
    fx, fy, fth = T[-1]
    return dict(
        sim_time=float((len(T) - 1) * dt),
        track_err_smac_mean=float(e_smac.mean()),
        track_err_smac_max=float(e_smac.max()),
        track_err_main_mean=float(e_main.mean()),
        track_err_main_max=float(e_main.max()),
        traj_min_clearance=float(min(clear)),
        entry_pos_error=float(math.hypot(ex - main_xy[0, 0], ey - main_xy[0, 1])),
        entry_heading_error_deg=float(abs(math.degrees(norm_angle(eth - math.atan2(
            main_xy[1, 1] - main_xy[0, 1], main_xy[1, 0] - main_xy[0, 0]))))),
        final_pos_error=float(math.hypot(fx - main_xy[-1, 0], fy - main_xy[-1, 1])),
        final_heading_error_deg=float(abs(math.degrees(norm_angle(fth - math.atan2(
            main_xy[-1, 1] - main_xy[-2, 1], main_xy[-1, 0] - main_xy[-2, 0]))))),
    )
