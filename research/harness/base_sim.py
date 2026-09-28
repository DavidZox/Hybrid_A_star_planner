"""Research harness: configurable Hybrid-A* + pure pursuit, used to reverse-engineer the slides."""
import heapq, math
import numpy as np

def norm_ang(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Costmap:
    def __init__(self, x_min=-0.5, y_min=-1.5, width=6.0, height=4.0, resolution=0.05):
        self.x_min, self.y_min, self.res = x_min, y_min, resolution
        self.nx = int(round(width / resolution)); self.ny = int(round(height / resolution))
        self.grid = np.zeros((self.ny, self.nx))
        self.obstacles = []

    def add_obstacle(self, x, y, radius, inflation_radius):
        self.obstacles.append((x, y, radius, inflation_radius))
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        wx = self.x_min + ii * self.res; wy = self.y_min + jj * self.res
        d = np.hypot(wx - x, wy - y)
        c = np.where(d <= radius, 200.0,
                     np.where(d <= radius + inflation_radius,
                              100.0 * (1 - (d - radius) / inflation_radius), 0.0))
        self.grid = np.maximum(self.grid, c)

    def get_cost(self, x, y, mode='floor'):
        if mode == 'floor':
            i = int(math.floor((x - self.x_min) / self.res)); j = int(math.floor((y - self.y_min) / self.res))
        else:
            i = int(round((x - self.x_min) / self.res)); j = int(round((y - self.y_min) / self.res))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return self.grid[j, i]
        return 200.0

    def collision(self, x, y, robot_radius):
        for ox, oy, r, _ in self.obstacles:
            if math.hypot(x - ox, y - oy) <= r + robot_radius:
                return True
        return False


DEFAULT = dict(
    step_size=0.06, reverse_penalty=0.3, change_dir_penalty=0.8, steering_penalty=0.1,
    steer_deg=12.0, angle_bin_deg=8.0, xy_res=0.05, heading_w=0.8,
    goal_dist=0.1, goal_ang_deg=45.0, goal_ang2_deg=45.0,
    goal_ang2_mode='bearing_vs_heading',  # or 'bearing_vs_goal', 'none', 'same'
    rev_mode='mult',       # 'add' : +rp per reverse step ; 'mult': step*(1+rp) ; 'mult_only': step*rp
    steer_mode='add',      # 'add' : +sp ; 'mult': step*sp ; 'abs': sp*|steer(rad)|
    map_w=0.01,            # g += map_w * cost
    map_mode='add',        # 'add' or 'mult_step' (step*map_w*cost)
    start_dir=0,           # direction attribute of start node (0 = none, 1 = fwd)
    robot_radius=0.22, roi_margin=1.5, max_iter=20000, fallback='best_h',
    analytic='line',       # straight line interpolation to goal
    cost_lookup='floor', check_lethal=False, key_mode='floor',
    closed_on='pop',       # 'pop' or 'push'
)


class Node:
    __slots__ = ('x', 'y', 't', 'g', 'd', 'parent', 'steer')
    def __init__(self, x, y, t, g, d, parent, steer=0.0):
        self.x, self.y, self.t, self.g, self.d, self.parent, self.steer = x, y, t, g, d, parent, steer


class Planner:
    def __init__(self, costmap, **kw):
        self.cm = costmap
        self.p = dict(DEFAULT); self.p.update(kw)
        self.iterations = 0

    def key(self, x, y, t):
        p = self.p
        ab = math.radians(p['angle_bin_deg'])
        if p['key_mode'] == 'floor':
            return (int(math.floor(x / p['xy_res'])), int(math.floor(y / p['xy_res'])),
                    int(math.floor(norm_ang(t) / ab)))
        elif p['key_mode'] == 'round':
            return (int(round(x / p['xy_res'])), int(round(y / p['xy_res'])), int(round(norm_ang(t) / ab)))
        elif p['key_mode'] == 'int':
            return (int(x / p['xy_res']), int(y / p['xy_res']), int(norm_ang(t) / ab))
        elif p['key_mode'] == 'int2pi':
            return (int(x / p['xy_res']), int(y / p['xy_res']), int((t % (2 * math.pi)) / ab))

    def h(self, x, y, t, goal):
        return math.hypot(goal[0] - x, goal[1] - y) + self.p['heading_w'] * abs(norm_ang(goal[2] - t))

    def is_goal(self, n, goal):
        p = self.p
        dist = math.hypot(n.x - goal[0], n.y - goal[1])
        angle_diff = abs(norm_ang(n.t - goal[2]))
        m = p['goal_ang2_mode']
        if m == 'bearing_vs_heading':
            b = math.atan2(goal[1] - n.y, goal[0] - n.x)
            g2 = abs(norm_ang(b - n.t))
        elif m == 'bearing_vs_goal':
            b = math.atan2(goal[1] - n.y, goal[0] - n.x)
            g2 = abs(norm_ang(b - goal[2]))
        elif m == 'bearing_dir':  # direction-aware bearing
            b = math.atan2(goal[1] - n.y, goal[0] - n.x)
            hd = n.t if n.d >= 0 else n.t + math.pi
            g2 = abs(norm_ang(b - hd))
        else:
            g2 = 0.0
        return dist < p['goal_dist'] and angle_diff < math.radians(p['goal_ang_deg']) and g2 < math.radians(p['goal_ang2_deg'])

    def plan(self, start, goal):
        p = self.p; cm = self.cm
        m = p['roi_margin']
        self.roi = (min(start[0], goal[0]) - m, max(start[0], goal[0]) + m,
                    min(start[1], goal[1]) - m, max(start[1], goal[1]) + m)
        xmn, xmx, ymn, ymx = self.roi
        st = Node(start[0], start[1], start[2], 0.0, p['start_dir'], None)
        openl = []; cnt = 0
        heapq.heappush(openl, (self.h(*start, goal), cnt, st))
        closed = set()
        if p['closed_on'] == 'push':
            closed.add(self.key(st.x, st.y, st.t))
        steers = [math.radians(-p['steer_deg']), 0.0, math.radians(p['steer_deg'])]
        dirs = [1, -1]
        best = st; best_h = self.h(*start, goal)
        it = 0
        found = None
        while openl:
            f, _, cur = heapq.heappop(openl)
            if p['closed_on'] == 'pop':
                k = self.key(cur.x, cur.y, cur.t)
                if k in closed:
                    continue
                closed.add(k)
            it += 1
            if it > p['max_iter']:
                break
            if self.is_goal(cur, goal):
                found = cur; break
            hc = self.h(cur.x, cur.y, cur.t, goal)
            if hc < best_h:
                best_h = hc; best = cur
            for d in dirs:
                for s in steers:
                    nt = norm_ang(cur.t + s * d)
                    nx = cur.x + d * p['step_size'] * math.cos(nt)
                    ny = cur.y + d * p['step_size'] * math.sin(nt)
                    if not (xmn <= nx <= xmx and ymn <= ny <= ymx):
                        continue
                    if cm.collision(nx, ny, p['robot_radius']):
                        continue
                    c = cm.get_cost(nx, ny, p['cost_lookup'])
                    if p['check_lethal'] and c >= 200:
                        continue
                    if p['closed_on'] == 'push':
                        k = self.key(nx, ny, nt)
                        if k in closed:
                            continue
                    step = p['step_size']
                    if d == -1:
                        if p['rev_mode'] == 'add':
                            g = step + p['reverse_penalty']
                        elif p['rev_mode'] == 'mult':
                            g = step * (1 + p['reverse_penalty'])
                        elif p['rev_mode'] == 'mult_only':
                            g = step * p['reverse_penalty']
                        elif p['rev_mode'] == 'add_step':
                            g = step + p['reverse_penalty'] * step
                    else:
                        g = step
                    if s != 0:
                        if p['steer_mode'] == 'add':
                            g += p['steering_penalty']
                        elif p['steer_mode'] == 'mult':
                            g += step * p['steering_penalty']
                        elif p['steer_mode'] == 'abs':
                            g += p['steering_penalty'] * abs(s)
                    if cur.d != 0 and d != cur.d:
                        g += p['change_dir_penalty']
                    if p['map_mode'] == 'add':
                        g += p['map_w'] * c
                    else:
                        g += step * p['map_w'] * c
                    n = Node(nx, ny, nt, cur.g + g, d, cur, s)
                    if p['closed_on'] == 'push':
                        closed.add(k)
                    cnt += 1
                    heapq.heappush(openl, (n.g + self.h(nx, ny, nt, goal), cnt, n))
        self.iterations = it
        self.found = found is not None
        end = found if found is not None else (best if p['fallback'] == 'best_h' else None)
        if end is None:
            return None
        path = []
        n = end
        while n is not None:
            path.append((n.x, n.y, n.t, n.d if n.d != 0 else 1))
            n = n.parent
        path.reverse()
        # fix direction of first point to that of second
        if len(path) > 1:
            path[0] = (path[0][0], path[0][1], path[0][2], path[1][3])
        if found is not None and p['analytic'] == 'line':
            ex, ey, et, ed = path[-1]
            dist = math.hypot(goal[0] - ex, goal[1] - ey)
            nseg = max(1, int(math.ceil(dist / p['step_size'])))
            for i in range(1, nseg + 1):
                t = i / nseg
                path.append((ex + t * (goal[0] - ex), ey + t * (goal[1] - ey), goal[2] if i == nseg else et, ed))
        return path


def split_segments(path):
    segs = []; cur = [path[0]]
    for pt in path[1:]:
        if pt[3] != cur[-1][3]:
            segs.append(cur); cur = [cur[-1][:3] + (pt[3],), pt]
        else:
            cur.append(pt)
    segs.append(cur)
    return segs


TRACK = dict(v=0.2, dt=0.1, lookahead=0.15, k=2.5, wmax=1.0, seg_tol=0.1, end_tol=0.1,
             nearest='global', max_steps=4000)


def simulate(path, main_path, start, tk=None):
    tp = dict(TRACK); tp.update(tk or {})
    x, y, th = start
    traj = [(x, y, th)]
    segs = split_segments(path)
    phase_list = [(np.array([[p[0], p[1]] for p in s]), s[-1][3] if len(s) > 1 else s[0][3]) for s in segs]
    # direction of each segment = direction of its points (take last)
    phase_list.append((np.array(main_path), 1))
    for si, (pts, mdir) in enumerate(phase_list):
        is_main = si == len(phase_list) - 1
        tol = tp['end_tol'] if is_main else tp['seg_tol']
        last_idx = 0
        for _ in range(tp['max_steps']):
            if math.hypot(pts[-1, 0] - x, pts[-1, 1] - y) < tol:
                break
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            if tp['nearest'] == 'global':
                ni = int(np.argmin(d))
            else:
                ni = last_idx + int(np.argmin(d[last_idx:])); last_idx = ni
            ti = len(pts) - 1
            for j in range(ni, len(pts)):
                if d[j] >= tp['lookahead']:
                    ti = j; break
            tx, ty = pts[ti]
            heading = th if mdir > 0 else th + math.pi
            alpha = norm_ang(math.atan2(ty - y, tx - x) - heading)
            v = tp['v'] * mdir
            w = float(np.clip(tp['k'] * alpha, -tp['wmax'], tp['wmax']))
            x += v * math.cos(th) * tp['dt']
            y += v * math.sin(th) * tp['dt']
            th = norm_ang(th + w * tp['dt'])
            traj.append((x, y, th))
    return np.array(traj)
