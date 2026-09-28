"""研究用：把某個變體的 11 組路徑疊在原圖數位化曲線上。用法：python overlay_variant.py 75 '<變體 JSON>' 30000 best_h out.png"""
import json, math, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evalsim, base_sim as sim, variant_planner as sim2, search_variants as batch2

def run2(s, th, kw, M, fb):
    sc = dict(evalsim.SCEN[s]); ox = sc.pop('ox')
    k2 = dict(kw); k2.update(sc); k2['max_iter'] = M
    cm = sim.Costmap(); cm.add_obstacle(ox, 0.1, 0.2, 0.35)
    pl = sim2.Planner2(cm, **k2)
    goal = (1.5, -0.5, 0.0)
    pl.plan((1.0, 1.0, math.radians(th)), goal)
    path, ok = pl.result(goal, M, fb)
    return pl, path, ok

def overlay(th, kw, M, fb, out, track=None):
    slides = list(evalsim.SCEN)
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    tot = 0
    for ax, s in zip(np.array(axs).flat, slides):
        pl, path, ok = run2(s, th, kw, M, fb)
        e = batch2.gerr(s, path); tot += e
        T = evalsim.TREES[s]
        ax.plot(T['b'][:, 0], T['b'][:, 1], '.', color='#9ab', ms=1)
        ax.plot(T['g'][:, 0], T['g'][:, 1], '.', color='#6c6', ms=1)
        if path:
            P = np.array(path); fw = P[:, 3] > 0
            ax.plot(P[:, 0], P[:, 1], 'k-', lw=0.8)
            ax.plot(P[fw, 0], P[fw, 1], 'r.', ms=2); ax.plot(P[~fw, 0], P[~fw, 1], 'm.', ms=2)
            if track is not None:
                main = [(1.5 + i * 0.05, -0.5) for i in range(71)]
                traj = sim.simulate(path, main, (1.0, 1.0, math.radians(th)), track)
                ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=0.8)
        ox = evalsim.SCEN[s]['ox']
        ax.add_patch(plt.Circle((ox, 0.1), 0.42, fill=False, ls='--', color='r'))
        ax.add_patch(plt.Circle((ox, 0.1), 0.55, fill=False, ls=':', color='orange'))
        ax.set_xlim(-0.2, 2.3); ax.set_ylim(-1.0, 1.4); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title('s%s %s it=%d(found@%s) e=%.3f' % (s, 'ok' if ok else 'FB', min(pl.iterations, M), pl.iterations if pl.found_node else '-', e), fontsize=10)
    fig.suptitle('th=%s M=%s fb=%s total=%.3f %s' % (th, M, fb, tot, kw), fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=55); plt.close(fig)
    return tot

if __name__ == '__main__':
    th = float(sys.argv[1]); kw = json.loads(sys.argv[2]); M = int(sys.argv[3]); fb = sys.argv[4]; out = sys.argv[5]
    track = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
    print(overlay(th, kw, M, fb, out, track))
