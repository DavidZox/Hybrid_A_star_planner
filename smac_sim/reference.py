"""讀取簡報數位化曲線（reference/digitized/*.json），計算重現結果與原圖的偏差。"""
import json
from pathlib import Path

import numpy as np
import yaml

from .metrics import point_to_polyline

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'reference'
LEGEND_BOX = (2.9, 1.2)          # 圖例在 x > 2.9 且 y > 1.2 的區域，不列入比對
MANEUVER_X_MAX = 2.6             # 軌跡只比對 x < 2.6（之後就是沿主路徑直行）


def load_slide(slide):
    d = json.loads((REF / 'digitized' / f'slide{int(slide):02d}.json').read_text())
    g = np.asarray(d['green'], dtype=float)
    b = np.asarray(d['blue'], dtype=float)
    keep = lambda p: p[~((p[:, 0] > LEGEND_BOX[0]) & (p[:, 1] > LEGEND_BOX[1]))]
    g, b = keep(g), keep(b)
    return dict(green=g, blue=b[b[:, 0] < MANEUVER_X_MAX], image=REF / 'slides' / d['image'], raw=d)


def load_annotations():
    return yaml.safe_load((REF / 'slide_annotations.yaml').read_text(encoding='utf-8'))


def _densify(xy, ds=0.01):
    xy = np.asarray(xy, dtype=float)
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.hypot(*(b - a)) / ds))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def _nearest(a, b):
    """a 中每點到點集 b 的最近距離。"""
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a)[0]


def deviation(ref_pts, curve_xy):
    """原圖像素點 ↔ 重現曲線的雙向偏差（公尺）。

    to_curve：原圖像素到重現折線的距離（原圖畫出的線有沒有被重現）
    to_ref：  重現曲線取樣點到原圖像素的距離（重現結果有沒有多出原圖沒有的部分）
    線寬約 0.03 m，完全重合時平均值約 0.01 m。
    """
    curve = _densify(curve_xy)
    to_curve = point_to_polyline(ref_pts, curve_xy) if len(curve_xy) > 1 else _nearest(ref_pts, curve)
    to_ref = _nearest(curve, ref_pts)
    both = np.concatenate([to_curve, to_ref])
    return dict(mean=float(both.mean()), p95=float(np.percentile(both, 95)),
                to_curve_mean=float(to_curve.mean()), to_ref_mean=float(to_ref.mean()))
