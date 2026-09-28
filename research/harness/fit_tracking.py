"""研究用：以吻合度高的頁面擬合 Pure Pursuit 參數。用法：python fit_tracking.py 5,6,8,9,10,12 '{"v":[0.1,0.15,0.2],"dt":[0.1],"lookahead":[0.15,0.2],"seg_tol":[0.1,0.15],"main_ds":[0.1]}'"""
import itertools, json, math, sys
import numpy as np
from scipy.spatial import cKDTree
import evalsim, base_sim as sim, variant_planner as sim2
KW = {"goal_ang2_mode":"motion_vs_goal","angle_bin_deg":8,"key_dir":False,"rev_mode":"mult","steer_mode":"mult","map_w":0.01,"max_iter":30000}
slides = sys.argv[1].split(',')
grid = json.loads(sys.argv[2])
goal = (1.5, -0.5, 0.0); th0 = math.radians(75)
paths = {}
for s in slides:
    sc = dict(evalsim.SCEN[s]); ox = sc.pop('ox'); k2 = dict(KW); k2.update(sc)
    cm = sim.Costmap(); cm.add_obstacle(ox, 0.1, 0.2, 0.35)
    pl = sim2.Planner2(cm, **k2); pl.plan((1.0, 1.0, th0), goal)
    paths[s], _ = pl.result(goal, k2['max_iter'], 'best_h')
main_opts = {0.05: [(1.5 + i * 0.05, -0.5) for i in range(71)], 0.1: [(1.5 + i * 0.1, -0.5) for i in range(36)]}
res = []
keys = list(grid)
for vals in itertools.product(*[grid[k] for k in keys]):
    tk = dict(zip(keys, vals)); mds = tk.pop('main_ds', 0.05)
    tot = 0; det = {}
    for s in slides:
        traj = sim.simulate(paths[s], main_opts[mds], (1.0, 1.0, th0), tk)
        T = evalsim.TREES[s]
        Q = traj[:, :2]; Q = Q[Q[:, 0] < 2.6]
        tQ = cKDTree(evalsim.densify(Q))
        e = tQ.query(T['b'])[0].mean() + T['bt'].query(Q)[0].mean()
        det[s] = round(e, 3); tot += e
    tk['main_ds'] = mds
    res.append((round(tot, 4), tk, det, round(float(traj[-1, 0]), 3)))
res.sort(key=lambda r: r[0])
for r in res[:15]:
    print(r)
