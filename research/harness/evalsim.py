"""研究用：11 組簡報情境、讀取 reference/digitized 曲線、以及路徑誤差計算（g=原圖→重現，gcov=重現→原圖）。"""
import json, math, sys, time
import numpy as np
from scipy.spatial import cKDTree
import base_sim as sim

from pathlib import Path
_REF = Path(__file__).resolve().parents[2] / 'reference' / 'digitized'
DIG = {}
for _f in sorted(_REF.glob('slide*.json')):
    _d = json.loads(_f.read_text())
    DIG[str(_d['slide'])] = dict(green=_d['green'], blue=_d['blue'])
SCEN = {
    '3': dict(ox=0.1), '4': dict(ox=0.5), '5': dict(ox=0.9), '6': dict(ox=1.2), '7': dict(ox=1.5),
    '8': dict(ox=0.9, steering_penalty=0.8), '9': dict(ox=0.9, steering_penalty=2.0),
    '10': dict(ox=0.9, reverse_penalty=2.0), '11': dict(ox=0.9, reverse_penalty=10.0),
    '12': dict(ox=0.9, change_dir_penalty=0.1), '13': dict(ox=0.9, change_dir_penalty=0.1, goal_dist=1.0),
}
TREES = {}
for s, v in DIG.items():
    g = np.array(v['green']); b = np.array(v['blue'])
    g = g[~((g[:, 0] > 2.9) & (g[:, 1] > 1.2))]; b = b[~((b[:, 0] > 2.9) & (b[:, 1] > 1.2))]
    # restrict to local region (planning area) for green; blue keep x<2.6 for the interesting part
    TREES[s] = dict(g=g, b=b[b[:, 0] < 2.6], gt=cKDTree(g), bt=cKDTree(b[b[:, 0] < 2.6]))


def densify(xy, ds=0.01):
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        L = np.hypot(*(b - a)); n = max(1, int(L / ds))
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return np.array(out)


def run(s, theta0=math.pi / 2, plan_kw=None, track_kw=None, main_ds=0.05):
    sc = dict(SCEN[s]); ox = sc.pop('ox')
    kw = dict(plan_kw or {}); kw.update(sc)
    cm = sim.Costmap(); cm.add_obstacle(ox, 0.1, 0.2, 0.35)
    pl = sim.Planner(cm, **kw)
    start = (1.0, 1.0, theta0); goal = (1.5, -0.5, 0.0)
    t0 = time.time(); path = pl.plan(start, goal); dt = time.time() - t0
    n = int(round(3.5 / main_ds))
    main = [(1.5 + i * main_ds, -0.5) for i in range(n + 1)]
    traj = sim.simulate(path, main, start, track_kw) if path else None
    return dict(path=path, traj=traj, iters=pl.iterations, found=pl.found, time=dt)


def score(s, r):
    T = TREES[s]
    if r['path'] is None:
        return dict(g=9, gcov=9, b=9, bcov=9)
    P = densify(np.array([[p[0], p[1]] for p in r['path']]))
    tP = cKDTree(P)
    e_g = tP.query(T['g'])[0].mean()          # digitized green -> my path
    e_gcov = T['gt'].query(P)[0].mean()       # my path -> digitized green
    out = dict(g=e_g, gcov=e_gcov)
    if r['traj'] is not None:
        Q = r['traj'][:, :2]; Q = Q[Q[:, 0] < 2.6]
        if len(Q) > 1:
            tQ = cKDTree(densify(Q))
            out['b'] = tQ.query(T['b'])[0].mean(); out['bcov'] = T['bt'].query(Q)[0].mean()
        else:
            out['b'] = out['bcov'] = 9
    return out


if __name__ == '__main__':
    th = math.radians(float(sys.argv[1])) if len(sys.argv) > 1 else math.pi / 2
    tot = 0
    for s in SCEN:
        r = run(s, th)
        sc = score(s, r)
        tot += sc['g'] + sc['gcov']
        print(s, 'found' if r['found'] else 'NOTFOUND', 'it=%d t=%.2fs' % (r['iters'], r['time']),
              ' '.join('%s=%.3f' % (k, v) for k, v in sc.items()))
    print('total green err %.3f' % tot)
