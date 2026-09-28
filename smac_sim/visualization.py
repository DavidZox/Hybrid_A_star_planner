"""繪圖：重現簡報第 3–13 頁的模擬結果圖、搜尋樹圖與 GIF 動畫。

樣式由原圖量測而來：figsize 8x6、costmap 以 YlOrRd（alpha 0.3）顯示、ROI 橘色虛線（最上層）、
障礙物 darkred（alpha 0.9）、紅色虛線為碰撞邊界（障礙物半徑 + 機器人半徑）、主路徑紅線（alpha 0.6）、
Smac 路徑綠色虛線、實際軌跡藍線、終點機器人藍圓（alpha 0.6）與黑色朝向箭頭。
"""
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TITLE = 'Nav2 Smac Planner with ROI Trajectory Tracking Simulation'


def _draw_static(ax, res):
    cm = res['costmap']
    ax.imshow(cm.grid, cmap='YlOrRd', origin='lower', extent=cm.extent, alpha=0.3)
    xmn, xmx, ymn, ymx = res['roi']
    ax.add_patch(patches.Rectangle((xmn, ymn), xmx - xmn, ymx - ymn, linewidth=1.5, edgecolor='orange',
                                   facecolor='none', linestyle='--', label='Planner ROI Region', zorder=5))
    for k, (ox, oy, r, _) in enumerate(cm.obstacles):
        ax.add_patch(plt.Circle((ox, oy), r, color='darkred', alpha=0.9, label='Obstacle' if k == 0 else None))
        ax.add_patch(plt.Circle((ox, oy), r + res['robot_radius'], color='red', fill=False, linestyle='--',
                                linewidth=1.5))
    mp = np.asarray(res['main_path'])
    ax.plot(mp[:, 0], mp[:, 1], 'r-', linewidth=3, alpha=0.6, label='Target Main Path')
    ax.plot(res['entry'][0], res['entry'][1], 'go', markersize=10, label='Main Path Entry', zorder=4)
    path = np.asarray(res['smac_path'])
    ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2.5, label='Smac Generated SE2 Path', zorder=3)


def _draw_robot(ax, pose, robot_radius):
    x, y, th = pose
    body = plt.Circle((x, y), robot_radius, color='blue', alpha=0.6, zorder=6)
    ax.add_patch(body)
    arrow = ax.arrow(x, y, 0.3 * math.cos(th), 0.3 * math.sin(th), width=0.02, head_width=0.08,
                     head_length=0.06, fc='k', ec='k', zorder=7)
    return body, arrow


def _finish(ax, title=TITLE):
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')


def plot_result(res, out_path, title=TITLE, dpi=100):
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_static(ax, res)
    traj = np.asarray(res['trajectory'])
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Actual AMR Trajectory', zorder=2.5)
    _draw_robot(ax, traj[-1], res['robot_radius'])
    _finish(ax, title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_search_tree(res, out_path, dpi=100):
    """展開過的節點（顏色 = 展開順序）與最終路徑，放大在規劃區域。"""
    plan = res['plan']
    E = np.asarray(plan.expanded)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    cm = res['costmap']
    ax.imshow(cm.grid, cmap='Greys', origin='lower', extent=cm.extent, alpha=0.25, vmin=0, vmax=400)
    for ox, oy, r, _ in cm.obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, color='#52514e', alpha=0.9))
        ax.add_patch(plt.Circle((ox, oy), r + res['robot_radius'], color='#52514e', fill=False, linewidth=1))
    blues = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blues_trunc', plt.get_cmap('Blues')(np.linspace(0.25, 1.0, 256)))
    sc = ax.scatter(E[:, 0], E[:, 1], c=np.arange(len(E)), s=2, cmap=blues, linewidths=0)
    path = np.asarray(res['smac_path'])
    fwd = path[:, 3] > 0
    ax.plot(path[:, 0], path[:, 1], '-', color='#0b0b0b', linewidth=1.2)
    ax.plot(path[fwd, 0], path[fwd, 1], 'o', color='#2a78d6', markersize=3, label='forward step')
    ax.plot(path[~fwd, 0], path[~fwd, 1], 'o', color='#eb6834', markersize=3, label='reverse step')
    ax.plot(res['start'][0], res['start'][1], 's', color='#0b0b0b', markersize=6, label='start')
    ax.plot(res['entry'][0], res['entry'][1], '*', color='#0b0b0b', markersize=11, label='goal (main path entry)')
    xmn, xmx, ymn, ymx = res['roi']
    ax.set_xlim(xmn, xmx)
    ax.set_ylim(ymn, ymx)
    ax.set_aspect('equal')
    ax.grid(color='#e5e4df', linewidth=0.6)
    ax.set_axisbelow(True)
    status = 'goal reached' if plan.found else 'best-effort (max_iterations reached)'
    ax.set_title(f'Hybrid-A* search tree: {plan.iterations} expansions, {status}', fontsize=11)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('expansion order')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def animate_result(res, out_path, every=4, fps=20, dpi=70):
    """GIF 動畫（Pillow），every 為每幾個模擬步取一幀。"""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_static(ax, res)
    traj = np.asarray(res['trajectory'])
    line, = ax.plot([], [], 'b-', linewidth=2, label='Actual AMR Trajectory', zorder=2.5)
    _finish(ax)
    state = {'artists': _draw_robot(ax, traj[0], res['robot_radius'])}
    frames = list(range(0, len(traj), every))
    if frames[-1] != len(traj) - 1:
        frames.append(len(traj) - 1)
    labels = res['states']

    def update(k):
        for a in state['artists']:
            a.remove()
        line.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
        state['artists'] = _draw_robot(ax, traj[k], res['robot_radius'])
        st = 'COMPLETED' if k == len(traj) - 1 else labels[k]
        ax.set_title(f'{TITLE}\nstate: {st}   t = {k * res.get("dt", 0.1):.1f} s', fontsize=11)
        return [line, *state['artists']]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
