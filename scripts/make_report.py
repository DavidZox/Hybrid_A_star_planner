"""彙整指標圖與 results/summary.md。

需先執行 run_experiments.py 與 compare_with_slides.py（param_sweep.py 可選）。
輸出：
  results/figures/metrics_path.png         路徑長度（前進／倒車）、換向次數、累積轉角
  results/figures/metrics_search.png       展開節點數、規劃時間
  results/figures/metrics_tracking.png     追蹤誤差、最小間距、模擬時間
  results/figures/reproduction_error.png   與簡報原圖的偏差
  results/figures/overview_reproduced.png  11 組重現結果圖總覽
  results/summary.md                       指標與重現對照表
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smac_sim.plotstyle import INK, INK_2, SERIES, use_report_style  # noqa: E402

RES = ROOT / 'results'
FIG = RES / 'figures'


def read_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v in ('True', 'False'):
                    d[k] = v == 'True'
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows


def hbars(ax, labels, series, names=None, colors=None, fmt='{:.2f}', stacked=False, total_fmt=None,
          legend_loc='lower right'):
    """水平長條（由上到下 T01..T11）。series: list of arrays。"""
    n = len(labels)
    y = np.arange(n)[::-1]
    colors = colors or SERIES
    if stacked:
        left = np.zeros(n)
        for k, vals in enumerate(series):
            ax.barh(y, vals, left=left, height=0.52, color=colors[k], edgecolor='white', linewidth=1.5,
                    label=names[k] if names else None)
            left += vals
        for yy, tot in zip(y, left):
            ax.text(tot, yy, '  ' + (total_fmt or fmt).format(tot), va='center', ha='left', fontsize=8, color=INK_2)
    else:
        m = len(series)
        h = 0.52 if m == 1 else 0.36
        for k, vals in enumerate(series):
            off = (k - (m - 1) / 2) * h
            ax.barh(y - off, vals, height=h, color=colors[k], edgecolor='white', linewidth=1.5,
                    label=names[k] if names else None)
            for yy, v in zip(y - off, vals):
                ax.text(v, yy, '  ' + fmt.format(v), va='center', ha='left', fontsize=7.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.margins(x=0.18)
    if names:
        ax.legend(loc=legend_loc, fontsize=8)


def fig_path(rows):
    lab = [r['label'] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    hbars(axs[0], lab, [np.array([r['forward_length'] for r in rows]), np.array([r['reverse_length'] for r in rows])],
          names=['前進', '倒車'], stacked=True, fmt='{:.2f} m')
    axs[0].set_title('(a) Smac 路徑長度 (m)', loc='left', fontsize=11)
    hbars(axs[1], lab, [np.array([r['cusps'] for r in rows])], fmt='{:.0f}')
    axs[1].set_title('(b) 換向（尖點）次數', loc='left', fontsize=11)
    hbars(axs[2], lab, [np.array([r['heading_change_deg'] for r in rows])], fmt='{:.0f}°')
    axs[2].set_title('(c) 路徑累積轉角 (deg)', loc='left', fontsize=11)
    fig.suptitle('規劃路徑指標', x=0.01, ha='left', fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'metrics_path.png', dpi=100)
    plt.close(fig)


def fig_search(rows):
    lab = [r['label'] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    hbars(axs[0], lab, [np.array([r['iterations'] for r in rows])], fmt='{:.0f}')
    axs[0].set_title('(a) Hybrid-A* 展開節點數（上限 30000）', loc='left', fontsize=11)
    hbars(axs[1], lab, [np.array([r['plan_time_ms'] for r in rows])], fmt='{:.0f} ms')
    axs[1].set_title('(b) 規劃時間 (ms，Python 單執行緒)', loc='left', fontsize=11)
    fig.suptitle('搜尋成本', x=0.01, ha='left', fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'metrics_search.png', dpi=100)
    plt.close(fig)


def fig_tracking(rows):
    lab = [r['label'] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    hbars(axs[0], lab, [np.array([r['track_err_smac_mean'] * 100 for r in rows]),
                        np.array([r['track_err_smac_max'] * 100 for r in rows])],
          names=['平均', '最大'], fmt='{:.1f}')
    axs[0].set_title('(a) 追蹤 Smac 路徑的橫向誤差 (cm)', loc='left', fontsize=11)
    hbars(axs[1], lab, [np.array([r['min_clearance'] * 100 for r in rows]),
                        np.array([r['traj_min_clearance'] * 100 for r in rows])],
          names=['規劃路徑', '實際軌跡'], fmt='{:.0f}')
    axs[1].set_title('(b) 機器人外緣到障礙物的最小間距 (cm)', loc='left', fontsize=11)
    hbars(axs[2], lab, [np.array([r['sim_time'] for r in rows])], fmt='{:.1f} s')
    axs[2].set_title('(c) 從起點到主路徑終點的模擬時間 (s)', loc='left', fontsize=11)
    fig.suptitle('追蹤控制指標（Pure Pursuit，v = 0.15 m/s）', x=0.01, ha='left', fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'metrics_tracking.png', dpi=100)
    plt.close(fig)


def fig_repro(rows):
    lab = [r['label'] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    hbars(ax, lab, [np.array([r['path_dev_mean_cm'] for r in rows]), np.array([r['traj_dev_mean_cm'] for r in rows])],
          names=['Smac 規劃路徑', '實際軌跡'], fmt='{:.1f}', legend_loc='upper right')
    ax.set_title('與簡報原圖的平均偏差 (cm)', loc='left', fontsize=12)
    ax.set_xlabel('原圖數位化曲線 ↔ 重現曲線的雙向平均距離；完全重合時約 1 cm（線寬）', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'reproduction_error.png', dpi=100)
    plt.close(fig)


def fig_overview(rows):
    fig, axs = plt.subplots(3, 4, figsize=(20, 11.5))
    for ax, r in zip(axs.flat, rows):
        ax.imshow(Image.open(RES / r['name'] / 'figure.png').crop((80, 40, 740, 560)))
        ax.set_title(f'{r["label"]}（簡報 p.{int(r["slide"])}）', fontsize=11)
        ax.axis('off')
    axs.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(FIG / 'overview_reproduced.png', dpi=80)
    plt.close(fig)


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def write_summary(rows, rep):
    yes = lambda b: '✓' if b else '✗'
    t1 = md_table(
        ['實驗', '簡報', '參數變化', '達成終點', '展開節點', '規劃 ms', '路徑長 m (前進/倒車)', '換向', '累積轉角°',
         '路徑最小間距 cm', '追蹤誤差 平均/最大 cm', '模擬 s'],
        [[r['label'], f'p.{int(r["slide"])}', r['description'], yes(r['found']), int(r['iterations']),
          f'{r["plan_time_ms"]:.0f}', f'{r["path_length"]:.2f} ({r["forward_length"]:.2f}/{r["reverse_length"]:.2f})',
          int(r['cusps']), f'{r["heading_change_deg"]:.0f}', f'{r["min_clearance"] * 100:.0f}',
          f'{r["track_err_smac_mean"] * 100:.1f}/{r["track_err_smac_max"] * 100:.1f}', f'{r["sim_time"]:.1f}']
         for r in rows])
    t2 = md_table(
        ['實驗', '簡報', '路徑偏差 平均/95% cm', '軌跡偏差 平均/95% cm', '換向 簡報/重現', '抵達方式 簡報/重現',
         '達成終點 簡報/重現', '簡報路徑型態'],
        [[r['label'], f'p.{int(r["slide"])}', f'{r["path_dev_mean_cm"]:.1f} / {r["path_dev_p95_cm"]:.1f}',
          f'{r["traj_dev_mean_cm"]:.1f} / {r["traj_dev_p95_cm"]:.1f}', f'{int(r["cusps_slide"])} / {int(r["cusps_repro"])}',
          f'{r["arrival_slide"]} / {r["arrival_repro"]}', f'{yes(r["goal_slide"])} / {yes(r["goal_repro"])}',
          r['route_slide']] for r in rep])
    text = f"""# 結果摘要

由 `scripts/make_report.py` 自動產生。

## 1. 各實驗指標（results/metrics.csv）

{t1}

- 追蹤誤差：實際軌跡到 Smac 路徑的橫向距離，只計 TRACKING_SMAC_SEGMENT 階段。
- 最小間距：機器人外緣（半徑 0.22 m）到障礙物表面的距離。

![](figures/metrics_path.png)
![](figures/metrics_search.png)
![](figures/metrics_tracking.png)

## 2. 與簡報原圖對照（results/reproduction_check.csv）

{t2}

- 偏差：原圖數位化像素與重現曲線的雙向平均距離；線寬約 3 cm，完全重合時約 1 cm。
- 簡報欄位（換向、抵達方式、是否達成終點）由原圖人工判讀，定義見 `reference/slide_annotations.yaml`。

![](figures/reproduction_error.png)
![](comparison/overlay_all.png)

## 3. 參數敏感度（results/sensitivity.csv）

![](figures/sensitivity.png)
"""
    (RES / 'summary.md').write_text(text, encoding='utf-8')


def main():
    use_report_style()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RES / 'metrics.csv')
    rep = read_csv(RES / 'reproduction_check.csv')
    fig_path(rows)
    fig_search(rows)
    fig_tracking(rows)
    fig_repro(rep)
    fig_overview(rows)
    write_summary(rows, rep)
    print('-> results/figures/*.png, results/summary.md')


if __name__ == '__main__':
    main()
