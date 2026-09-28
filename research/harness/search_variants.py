"""研究用：列舉規劃器實作變體（grids/*.json），一次搜尋即可評估所有 max_iterations 與退回規則。

用法：python search_variants.py ../grids/grid8.json ../runs/batch8_out.json 12
"""
import itertools, json, math, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
import evalsim, base_sim as sim, variant_planner as sim2

MS = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000,
      18000, 20000, 25000, 30000, 40000, 50000]
FALLBACKS = ['best_h', 'min_dist', 'last']
SLIDES = list(evalsim.SCEN)


def gerr(s, path):
    if not path:
        return 2.0
    T = evalsim.TREES[s]
    P = evalsim.densify(np.array([[p[0], p[1]] for p in path]))
    return cKDTree(P).query(T['g'])[0].mean() + T['gt'].query(P)[0].mean()


def job(args):
    th, kw = args
    goal = (1.5, -0.5, 0.0)
    per = {}  # s -> {(M, fb): err}
    info = {}
    for s in SLIDES:
        sc = dict(evalsim.SCEN[s]); ox = sc.pop('ox')
        k2 = dict(kw); k2.update(sc); k2['max_iter'] = MS[-1]
        cm = sim.Costmap(); cm.add_obstacle(ox, 0.1, 0.2, 0.35)
        pl = sim2.Planner2(cm, **k2)
        pl.plan((1.0, 1.0, math.radians(th)), goal)
        info[s] = pl.iterations if pl.found_node is not None else -1
        cache = {}
        d = {}
        for M in MS:
            for fb in FALLBACKS:
                path, ok = pl.result(goal, M, fb)
                key = (ok, id(path[-1]) if path else None, len(path) if path else 0, path[-1][:2] if path else None)
                if key not in cache:
                    cache[key] = gerr(s, path)
                d[(M, fb)] = cache[key]
        per[s] = d
    best = None
    for M in MS:
        for fb in FALLBACKS:
            tot = sum(per[s][(M, fb)] for s in SLIDES)
            if best is None or tot < best[0]:
                best = (tot, M, fb, {s: round(per[s][(M, fb)], 3) for s in SLIDES})
    return dict(th=th, kw=kw, tot=round(best[0], 3), M=best[1], fb=best[2], det=best[3], found_it=info)


if __name__ == '__main__':
    grid = json.loads(open(sys.argv[1]).read())
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, vals)); th = d.pop('theta0')
        combos.append((th, d))
    print('combos', len(combos), flush=True)
    t0 = time.time(); out = []
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 10) as p:
        for i, r in enumerate(p.imap_unordered(job, combos, chunksize=1)):
            out.append(r)
            if i % 20 == 0:
                print(i, '%.0fs' % (time.time() - t0), flush=True)
    out.sort(key=lambda r: r['tot'])
    json.dump(out, open(sys.argv[2], 'w'), indent=0, default=str)
    for r in out[:30]:
        print(r['tot'], r['th'], r['M'], r['fb'], r['kw'], r['det'], r['found_it'])
