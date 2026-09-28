"""把重現結果與簡報原圖逐頁比對。

需先執行 scripts/extract_slide_reference.py 與 scripts/run_experiments.py。
輸出：
  results/comparison/<實驗>.png        左：簡報原圖／中：重現結果／右：疊圖（淺色點 = 原圖數位化曲線）
  results/comparison/overlay_all.png   11 組疊圖總覽
  results/reproduction_check.csv       路徑與軌跡偏差、換向次數與抵達方式對照
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smac_sim import load_config, run_scenario  # noqa: E402
from smac_sim.config import experiment_files  # noqa: E402
from smac_sim.plotstyle import use_report_style  # noqa: E402
from smac_sim.reference import MANEUVER_X_MAX, deviation, load_annotations, load_slide  # noqa: E402

OUT = ROOT / 'results' / 'comparison'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e5e4df'
REF_PATH, REF_TRAJ = '#8fc79a', '#a9c3e8'          # 原圖數位化點（淺色）
REP_PATH, REP_TRAJ = '#008300', '#2a78d6'          # 重現曲線


def overlay_axes(ax, cfg, res, ref, title):
    ax.plot(ref['blue'][:, 0], ref['blue'][:, 1], '.', ms=1.6, color=REF_TRAJ, label='slide: actual trajectory')
    ax.plot(ref['green'][:, 0], ref['green'][:, 1], '.', ms=1.6, color=REF_PATH, label='slide: Smac path')
    traj = res['trajectory']
    path = res['smac_path']
    ax.plot(traj[:, 0], traj[:, 1], '-', lw=1.2, color=REP_TRAJ, label='reproduced trajectory')
    ax.plot(path[:, 0], path[:, 1], '-', lw=1.6, color=REP_PATH, label='reproduced Smac path')
    ob = cfg['obstacle']
    ax.add_patch(plt.Circle((ob['x'], ob['y']), ob['radius'], color='#52514e', alpha=0.85))
    ax.add_patch(plt.Circle((ob['x'], ob['y']), ob['radius'] + cfg['planner']['robot_radius'], color=MUTED,
                            fill=False, lw=0.8))
    ax.plot(*res['entry'], 'o', color=INK, ms=4)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-1.0, 1.4)
    ax.set_aspect('equal')
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#c9c8c2')
    ax.set_title(title, fontsize=9, color=INK)


def main():
    use_report_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ann = load_annotations()
    rows, panels = [], []
    for f in experiment_files():
        cfg = load_config(f)
        slide = cfg['slide']
        res = run_scenario(cfg)
        ref = load_slide(slide)
        m = res['metrics']
        dp = deviation(ref['green'], res['smac_path'][:, :2])
        t = res['trajectory'][:, :2]
        dt = deviation(ref['blue'], t[t[:, 0] < MANEUVER_X_MAX])
        a = ann[slide]
        arrival = 'forward' if res['smac_path'][-2, 3] > 0 else 'reverse'
        rows.append(dict(
            name=cfg['name'], slide=slide, label=cfg['label'], change=cfg['description'],
            path_dev_mean_cm=round(dp['mean'] * 100, 1), path_dev_p95_cm=round(dp['p95'] * 100, 1),
            traj_dev_mean_cm=round(dt['mean'] * 100, 1), traj_dev_p95_cm=round(dt['p95'] * 100, 1),
            cusps_slide=a['cusps'], cusps_repro=m['cusps'],
            arrival_slide=a['arrival'], arrival_repro=arrival,
            goal_slide=a['goal'], goal_repro=m['found'], route_slide=a['route']))

        fig, axs = plt.subplots(1, 3, figsize=(17, 4.9), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
        axs[0].imshow(Image.open(ref['image']))
        axs[0].set_title(f'slide p.{slide} (original)', fontsize=10)
        axs[1].imshow(Image.open(ROOT / 'results' / cfg['name'] / 'figure.png'))
        axs[1].set_title('reproduction', fontsize=10)
        for ax in axs[:2]:
            ax.axis('off')
        overlay_axes(axs[2], cfg, res, ref,
                     f'overlay: path dev {dp["mean"] * 100:.1f} cm, trajectory dev {dt["mean"] * 100:.1f} cm')
        axs[2].legend(loc='upper right', fontsize=7, frameon=False, markerscale=4)
        fig.suptitle(f'{cfg["name"]}  -  {cfg["title"]}', fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f'{cfg["name"]}.png', dpi=90)
        plt.close(fig)
        panels.append((cfg, res, ref, dp, dt))
        print(f'{cfg["name"]:<36} path {dp["mean"] * 100:5.1f} cm  traj {dt["mean"] * 100:5.1f} cm  '
              f'cusps {a["cusps"]}/{m["cusps"]}  arrival {a["arrival"]}/{arrival}')

    fig, axs = plt.subplots(3, 4, figsize=(18, 14))
    for ax, (cfg, res, ref, dp, dt) in zip(axs.flat, panels):
        overlay_axes(ax, cfg, res, ref, f'p.{cfg["slide"]} {cfg["description"]}\n'
                                        f'path {dp["mean"] * 100:.1f} cm / trajectory {dt["mean"] * 100:.1f} cm')
    ax = axs.flat[-1]
    handles, labels = axs.flat[0].get_legend_handles_labels()
    ax.axis('off')
    ax.legend(handles, labels, loc='center', fontsize=11, frameon=False, markerscale=6,
              title='light dots = digitized slide curves\nlines = reproduction')
    fig.tight_layout()
    fig.savefig(OUT / 'overlay_all.png', dpi=70)
    plt.close(fig)

    with open(ROOT / 'results' / 'reproduction_check.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('->', OUT, 'and results/reproduction_check.csv')


if __name__ == '__main__':
    main()
