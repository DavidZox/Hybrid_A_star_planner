"""參數敏感度掃描：在簡報測過的數值之間補點，觀察各懲罰與障礙物位置對路徑的影響。

以 configs/base.yaml 為基準（障礙物 x=0.9，除非掃描的就是障礙物位置），一次只改一個參數。
輸出：results/sensitivity.csv 與 results/figures/sensitivity.png
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smac_sim import load_config, run_scenario  # noqa: E402
from smac_sim.plotstyle import INK, INK_2, SERIES, use_report_style  # noqa: E402

SWEEPS = {
    'reverse_penalty': ('planner', [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0], [0.3, 2.0, 10.0]),
    'steering_penalty': ('planner', [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.0, 3.0], [0.1, 0.8, 2.0]),
    'change_dir_penalty': ('planner', [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.0], [0.1, 0.8]),
    'x': ('obstacle', [round(0.1 * k, 1) for k in range(0, 19)], [0.1, 0.5, 0.9, 1.2, 1.5]),
}


def run_sweeps():
    rows = []
    for param, (section, values, _) in SWEEPS.items():
        for v in values:
            ov = {section: {param: v}}
            if section != 'obstacle':
                ov['obstacle'] = {'x': 0.9}
            m = run_scenario(load_config(None, ov))['metrics']
            rows.append(dict(param=param, value=v, **m))
            print(f'{param:<20} {v:>5}  L={m["path_length"]:.2f} (fwd {m["forward_length"]:.2f} / rev '
                  f'{m["reverse_length"]:.2f})  cusps={m["cusps"]}  it={m["iterations"]}  '
                  f'{"found" if m["found"] else "best-effort"}')
    return rows


def plot(rows, out):
    use_report_style()
    by = lambda p: [r for r in rows if r['param'] == p]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))

    def panel(ax, p, key, label, title, color=SERIES[0], series_label=None, logx=False):
        rs = by(p)
        xs = [r['value'] for r in rs]
        ys = [r[key] for r in rs]
        ax.plot(xs, ys, '-', color=color, lw=2, label=series_label)
        tested = set(SWEEPS[p][2])
        for r, x, y in zip(rs, xs, ys):
            big = x in tested
            ok = r['found']
            ax.plot(x, y, 'o', ms=8 if big else 5, color=color if ok else 'white', mec=color if not ok else 'white',
                    mew=1.6 if not ok else (2 if big else 1), zorder=3)
        if logx:
            ax.set_xscale('symlog', linthresh=0.1)
            ticks = [t for t in (0, 0.1, 0.3, 1, 3, 10) if t <= max(xs) * 1.01]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=11, loc='left')

    panel(axs[0, 0], 'reverse_penalty', 'forward_length', '長度 (m)', '(a) reverse_penalty → 前進／倒車長度',
          SERIES[0], '前進', logx=True)
    panel(axs[0, 0], 'reverse_penalty', 'reverse_length', '長度 (m)', '(a) reverse_penalty → 前進／倒車長度',
          SERIES[1], '倒車', logx=True)
    axs[0, 0].legend(loc='center right')
    axs[0, 0].set_xlabel('reverse_penalty')
    panel(axs[0, 1], 'steering_penalty', 'heading_change_deg', '累積轉角 (deg)',
          '(b) steering_penalty → 路徑累積轉角', logx=True)
    axs[0, 1].set_xlabel('steering_penalty')
    panel(axs[0, 2], 'change_dir_penalty', 'cusps', '換向次數', '(c) change_dir_penalty → 換向（尖點）次數',
          logx=True)
    axs[0, 2].set_xlabel('change_dir_penalty')
    axs[0, 2].yaxis.get_major_locator().set_params(integer=True)
    panel(axs[1, 0], 'x', 'path_length', '路徑長度 (m)', '(d) 障礙物 x → 路徑長度')
    panel(axs[1, 1], 'x', 'min_clearance', '最小間距 (m)', '(e) 障礙物 x → 路徑與障礙物最小間距')
    panel(axs[1, 2], 'x', 'iterations', '展開節點數', '(f) 障礙物 x → Hybrid-A* 展開節點數')
    for ax in axs[1]:
        ax.set_xlabel('障礙物中心 x (m)')
    fig.text(0.01, 0.005, '大圓點 = 簡報中實際測試的數值；小圓點 = 補點；空心 = 30000 次迭代內未滿足終點條件（best-effort 路徑）。'
             '其餘參數同 configs/base.yaml（障礙物 x=0.9）。',
             color=INK_2, fontsize=9)
    fig.suptitle('參數敏感度（一次只改一個參數）', color=INK, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    rows = run_sweeps()
    (ROOT / 'results' / 'figures').mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(ROOT / 'results' / 'sensitivity.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    plot(rows, ROOT / 'results' / 'figures' / 'sensitivity.png')
    print('-> results/sensitivity.csv, results/figures/sensitivity.png')


if __name__ == '__main__':
    main()
