"""Planner variant with iteration trace (to evaluate any max_iter / fallback rule after one run)."""
import heapq, math
import numpy as np
from base_sim import Costmap, norm_ang, split_segments, simulate

DEF = dict(
    step_size=0.06, reverse_penalty=0.3, change_dir_penalty=0.8, steering_penalty=0.1,
    steer_deg=12.0, angle_bin_deg=8.0, xy_res=0.05, heading_w=0.8,
    goal_dist=0.1, goal_ang_deg=45.0, goal_ang2_deg=45.0, goal_ang2_mode='bearing_vs_heading',
    rev_mode='mult', steer_mode='abs', map_w=0.0, map_mode='add', start_dir=0,
    robot_radius=0.22, roi_margin=1.5, max_iter=60000,
    key_xy='floor', key_th='floor', closed_on='pop', goal_on='pop', dedup='none',
    lethal_block=False, analytic_cc=True, key_dir=False, key_mode2='xyt', h_mode='goal_heading', dir_order=(1, -1), steer_order=(-1, 0, 1), tie='fifo',
)


class Node:
    __slots__ = ('x', 'y', 't', 'g', 'd', 'parent', 's')
    def __init__(self, x, y, t, g, d, parent, s=0.0):
        self.x, self.y, self.t, self.g, self.d, self.parent, self.s = x, y, t, g, d, parent, s


class Planner2:
    def __init__(self, cm, **kw):
        self.cm = cm; self.p = dict(DEF); self.p.update(kw)

    def key(self, x, y, t):
        p = self.p; r = p['xy_res']; ab = math.radians(p['angle_bin_deg'])
        kx = p['key_xy']
        if kx == 'floor':
            ix, iy = math.floor(x / r), math.floor(y / r)
        elif kx == 'round':
            ix, iy = round(x / r), round(y / r)
        elif kx == 'int':
            ix, iy = int(x / r), int(y / r)
        elif kx == 'grid':
            ix, iy = int((x - self.cm.x_min) / r), int((y - self.cm.y_min) / r)
        kt = p['key_th']
        if kt == 'floor':
            it = math.floor(norm_ang(t) / ab)
        elif kt == 'int':
            it = int(norm_ang(t) / ab)
        elif kt == 'mod':
            it = int((t % (2 * math.pi)) / ab)
        elif kt == 'round':
            it = round(norm_ang(t) / ab) % int(round(360 / p['angle_bin_deg']))
        if p['key_mode2'] == 'xy':
            return (ix, iy)
        return (ix, iy, it)

    def h(self, x, y, t, goal):
        d = math.hypot(goal[0] - x, goal[1] - y)
        m = self.p['h_mode']
        if m == 'goal_heading':
            return d + self.p['heading_w'] * abs(norm_ang(goal[2] - t))
        b = math.atan2(goal[1] - y, goal[0] - x)
        if m == 'bearing':
            return d + self.p['heading_w'] * abs(norm_ang(b - t))
        if m == 'both':
            return d + self.p['heading_w'] * (abs(norm_ang(b - t)) + abs(norm_ang(goal[2] - t)))
        if m == 'bearing_goal':
            return d + self.p['heading_w'] * abs(norm_ang(b - goal[2]))

    def is_goal(self, x, y, t, d, goal):
        p = self.p
        dist = math.hypot(x - goal[0], y - goal[1])
        if dist >= p['goal_dist']:
            return False
        if abs(norm_ang(t - goal[2])) >= math.radians(p['goal_ang_deg']):
            return False
        m = p['goal_ang2_mode']
        if m == 'none':
            return True
        b = math.atan2(goal[1] - y, goal[0] - x)
        if m == 'bearing_vs_heading':
            g2 = abs(norm_ang(b - t))
        elif m == 'bearing_vs_goal':
            g2 = abs(norm_ang(b - goal[2]))
        elif m == 'bearing_dir':
            g2 = abs(norm_ang(b - (t if d >= 0 else t + math.pi)))
        elif m == 'motion_vs_goal':
            g2 = abs(norm_ang((t if d >= 0 else t + math.pi) - goal[2]))
        elif m == 'motion_and_bearing':
            g2 = max(abs(norm_ang((t if d >= 0 else t + math.pi) - goal[2])), abs(norm_ang(b - t)))
        if g2 >= math.radians(p['goal_ang2_deg']):
            return False
        return True

    def line_free(self, x, y, goal):
        dist = math.hypot(goal[0] - x, goal[1] - y)
        n = max(1, int(math.ceil(dist / (self.p['step_size'] * 0.5))))
        for i in range(1, n + 1):
            t = i / n
            if self.cm.collision(x + t * (goal[0] - x), y + t * (goal[1] - y), self.p['robot_radius']):
                return False
        return True

    def plan(self, start, goal):
        p = self.p; cm = self.cm; m = p['roi_margin']
        self.roi = (min(start[0], goal[0]) - m, max(start[0], goal[0]) + m,
                    min(start[1], goal[1]) - m, max(start[1], goal[1]) + m)
        xmn, xmx, ymn, ymx = self.roi
        st = Node(start[0], start[1], start[2], 0.0, p['start_dir'], None)
        openl = [(self.h(*start, goal), 0, 0, st)]; cnt = 0
        closed = set(); best_g = {}
        if p['closed_on'] == 'push':
            closed.add(self.key(*start) + ((p['start_dir'],) if p['key_dir'] else ()))
        steers = [k * math.radians(p['steer_deg']) for k in p['steer_order']]
        self.trace_h = []; self.trace_d = []; self.expanded = []
        bh = 1e9; bd = 1e9
        it = 0; found = None; step = p['step_size']
        while openl:
            f, _, _, cur = heapq.heappop(openl)
            if p['closed_on'] == 'pop':
                k = self.key(cur.x, cur.y, cur.t) + ((cur.d,) if p['key_dir'] else ())
                if k in closed:
                    continue
                closed.add(k)
            it += 1
            if it > p['max_iter']:
                break
            self.expanded.append(cur)
            if p['goal_on'] == 'pop' and self.is_goal(cur.x, cur.y, cur.t, cur.d, goal) and (not p['analytic_cc'] or self.line_free(cur.x, cur.y, goal)):
                found = cur; break
            hc = self.h(cur.x, cur.y, cur.t, goal)
            if hc < bh:
                bh = hc; self.trace_h.append((it, cur))
            dc = math.hypot(cur.x - goal[0], cur.y - goal[1])
            if dc < bd:
                bd = dc; self.trace_d.append((it, cur))
            for d in p['dir_order']:
                for s in steers:
                    nt = norm_ang(cur.t + s * d)
                    nx = cur.x + d * step * math.cos(nt)
                    ny = cur.y + d * step * math.sin(nt)
                    if not (xmn <= nx <= xmx and ymn <= ny <= ymx):
                        continue
                    if cm.collision(nx, ny, p['robot_radius']):
                        continue
                    c = cm.get_cost(nx, ny)
                    if p['lethal_block'] and c >= 200:
                        continue
                    k = None
                    if p['closed_on'] == 'push' or p['dedup'] != 'none':
                        k = self.key(nx, ny, nt) + ((d,) if p['key_dir'] else ())
                    if p['closed_on'] == 'push' and k in closed:
                        continue
                    if d == -1:
                        rm = p['rev_mode']
                        if rm == 'mult':
                            g = step * (1 + p['reverse_penalty'])
                        elif rm == 'add':
                            g = step + p['reverse_penalty']
                        elif rm == 'mult_only':
                            g = step * p['reverse_penalty']
                    else:
                        g = step
                    sm = p['steer_mode']
                    if sm == 'change':
                        if s != cur.s:
                            g += p['steering_penalty']
                    elif sm == 'change_abs':
                        g += p['steering_penalty'] * abs(s - cur.s)
                    elif sm == 'abs_change_abs':
                        g += p['steering_penalty'] * (abs(s) + abs(s - cur.s))
                    elif sm == 'add_change':
                        if s != 0:
                            g += p['steering_penalty']
                        if s != cur.s:
                            g += p['steering_penalty']
                    elif s != 0:
                        if sm == 'add':
                            g += p['steering_penalty']
                        elif sm == 'mult':
                            g += step * p['steering_penalty']
                        elif sm == 'abs':
                            g += p['steering_penalty'] * abs(s)
                    if cur.d != 0 and d != cur.d:
                        g += p['change_dir_penalty']
                    if p['map_mode'] == 'add':
                        g += p['map_w'] * c
                    else:
                        g += step * p['map_w'] * c
                    ng = cur.g + g
                    if p['dedup'] == 'best_g':
                        if k in best_g and best_g[k] <= ng:
                            continue
                        best_g[k] = ng
                    n = Node(nx, ny, nt, ng, d, cur, s)
                    if p['closed_on'] == 'push':
                        closed.add(k)
                    if p['goal_on'] == 'gen' and self.is_goal(nx, ny, nt, d, goal) and (not p['analytic_cc'] or self.line_free(nx, ny, goal)):
                        found = n; break
                    cnt += 1
                    hn = self.h(nx, ny, nt, goal)
                    tb = cnt if p['tie'] == 'fifo' else (-cnt if p['tie'] == 'lifo' else (hn if p['tie'] == 'h' else ng))
                    heapq.heappush(openl, (ng + hn, tb, cnt, n))
                if found is not None and p['goal_on'] == 'gen':
                    break
            if found is not None and p['goal_on'] == 'gen':
                break
        self.iterations = it
        self.found_node = found
        return found

    @staticmethod
    def chain(n):
        path = []
        while n is not None:
            path.append((n.x, n.y, n.t, n.d if n.d != 0 else 1)); n = n.parent
        path.reverse()
        if len(path) > 1:
            path[0] = (path[0][0], path[0][1], path[0][2], path[1][3])
        return path

    def result(self, goal, max_iter, fallback='best_h', analytic=True):
        """Path as if the search had been limited to max_iter expansions."""
        found = self.found_node
        if found is not None and self.iterations <= max_iter:
            path = self.chain(found)
            if analytic:
                ex, ey, et, ed = path[-1]
                dist = math.hypot(goal[0] - ex, goal[1] - ey)
                nseg = max(1, int(math.ceil(dist / self.p['step_size'])))
                for i in range(1, nseg + 1):
                    t = i / nseg
                    path.append((ex + t * (goal[0] - ex), ey + t * (goal[1] - ey), goal[2] if i == nseg else et, ed))
            return path, True
        if fallback == 'best_h':
            tr = self.trace_h
        elif fallback == 'min_dist':
            tr = self.trace_d
        elif fallback == 'last':
            return self.chain(self.expanded[min(max_iter, len(self.expanded)) - 1]), False
        else:
            return None, False
        node = None
        for i, n in tr:
            if i <= max_iter:
                node = n
            else:
                break
        return (self.chain(node) if node else None), False
