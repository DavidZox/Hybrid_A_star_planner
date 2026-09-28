"""執行簡報中的 11 組 Hybrid A* 測試（或自訂參數），輸出結果圖、路徑、軌跡與指標。

用法：
  python scripts/run_experiments.py                     # 全部 configs/experiments/*.yaml
  python scripts/run_experiments.py --exp configs/experiments/test03_obstacle_x0.9.yaml --gif
  python scripts/run_experiments.py --name my_test --set obstacle.x=0.7 planner.reverse_penalty=1.0

輸出（results/<實驗名稱>/）：
  figure.png          簡報樣式的模擬結果圖
  search_tree.png     Hybrid-A* 展開節點（依展開順序著色）與最終路徑
  smac_path.csv       規劃路徑 (x, y, theta, direction)
  trajectory.csv      實際軌跡 (t, x, y, theta, state)
  metrics.json        規劃與追蹤指標；config.yaml 為實際使用的完整參數
  animation.gif       --gif 時輸出
results/metrics.csv   所有實驗的指標彙整
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smac_sim import load_config, run_scenario  # noqa: E402
from smac_sim.config import experiment_files, parse_set  # noqa: E402
from smac_sim.visualization import animate_result, plot_result, plot_search_tree  # noqa: E402

RESULTS = ROOT / 'results'


def save_outputs(cfg, res, out_dir, gif=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_result(res, out_dir / 'figure.png')
    plot_search_tree(res, out_dir / 'search_tree.png')
    with open(out_dir / 'smac_path.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'theta_rad', 'direction'])
        for x, y, th, d in res['smac_path']:
            w.writerow([f'{x:.5f}', f'{y:.5f}', f'{th:.5f}', int(d)])
    dt = cfg['tracking']['dt']
    with open(out_dir / 'trajectory.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'x', 'y', 'theta_rad', 'state'])
        for k, ((x, y, th), st) in enumerate(zip(res['trajectory'], res['states'])):
            w.writerow([f'{k * dt:.2f}', f'{x:.5f}', f'{y:.5f}', f'{th:.5f}', st])
    (out_dir / 'metrics.json').write_text(json.dumps(res['metrics'], indent=2, ensure_ascii=False))
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    if gif:
        animate_result(res, out_dir / 'animation.gif')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', nargs='*', help='實驗設定檔；省略時執行全部')
    ap.add_argument('--name', help='自訂輸出名稱（搭配 --set 使用）')
    ap.add_argument('--set', nargs='*', default=[], help='覆寫參數，例如 planner.reverse_penalty=2.0')
    ap.add_argument('--gif', action='store_true', help='同時輸出 GIF 動畫（較慢）')
    args = ap.parse_args()

    overrides = parse_set(args.set)
    if args.exp:
        files = [Path(p) for p in args.exp]
    elif args.name or overrides:
        files = [None]
    else:
        files = experiment_files()

    rows = []
    for f in files:
        cfg = load_config(f, overrides)
        if args.name:
            cfg['name'] = args.name if len(files) == 1 else f'{args.name}_{cfg["name"]}'
        res = run_scenario(cfg)
        save_outputs(cfg, res, RESULTS / cfg['name'], gif=args.gif)
        m = res['metrics']
        line = (f'{cfg["name"]:<36} {"found" if m["found"] else "best-effort":<11} it={m["iterations"]:>6} '
                f'{m["plan_time_ms"]:7.1f} ms  L={m["path_length"]:.2f} m (fwd {m["forward_length"]:.2f} / '
                f'rev {m["reverse_length"]:.2f})  cusps={m["cusps"]}  track_err={m["track_err_smac_mean"] * 100:.1f} cm  '
                f'final=({res["trajectory"][-1, 0]:.3f}, {res["trajectory"][-1, 1]:.3f})')
        print(line)
        rows.append(dict(name=cfg['name'], slide=cfg.get('slide', ''), label=cfg.get('label', cfg['name']),
                         description=cfg.get('description', ''), **m))

    if files and files[0] is not None and not args.name and not overrides and len(files) == len(experiment_files()):
        keys = list(rows[0].keys())
        with open(RESULTS / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f'{v:.4f}' if isinstance(v, float) and math.isfinite(v) else v) for k, v in r.items()})
        print(f'-> {RESULTS / "metrics.csv"}')


if __name__ == '__main__':
    main()
