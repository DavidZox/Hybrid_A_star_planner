"""研究用：以 stdin 逐行 JSON 覆寫參數，快速比較少數變體。

用法：echo '{"theta0":75}' | python quick_eval.py '{"goal_ang2_mode":"motion_vs_goal","angle_bin_deg":8,"key_dir":false,"rev_mode":"mult","steer_mode":"mult","map_w":0.01}' 3,9,13
"""
import json, math, sys
import evalsim, base_sim as sim, variant_planner as sim2, search_variants as batch2
def run(th, kw, slides, M=60000, fb='best_h'):
    res = {}
    for s in slides:
        sc = dict(evalsim.SCEN[s]); ox = sc.pop('ox')
        k2 = dict(kw); k2.update(sc); k2['max_iter'] = M
        cm = sim.Costmap(); cm.add_obstacle(ox, 0.1, 0.2, 0.35)
        pl = sim2.Planner2(cm, **k2); goal = (1.5, -0.5, 0.0)
        pl.plan((1.0, 1.0, math.radians(th)), goal)
        path, ok = pl.result(goal, M, fb)
        res[s] = (round(batch2.gerr(s, path), 3), pl.iterations if pl.found_node else -1)
    return res
if __name__ == '__main__':
    base = json.loads(sys.argv[1]); slides = sys.argv[2].split(',')
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        ch = json.loads(line); th = ch.pop('theta0', 78)
        kw = dict(base); kw.update(ch)
        print(th, ch, run(th, kw, slides), flush=True)
