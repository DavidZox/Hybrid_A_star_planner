"""讀取 configs/base.yaml 並套用各實驗 yaml 的覆寫欄位。"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs' / 'base.yaml'
EXP_DIR = ROOT / 'configs' / 'experiments'


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(exp_path=None, overrides=None):
    cfg = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    if exp_path is not None:
        _merge(cfg, yaml.safe_load(Path(exp_path).read_text(encoding='utf-8')))
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    cfg.setdefault('name', 'custom')
    return cfg


def experiment_files():
    return sorted(EXP_DIR.glob('*.yaml'))


def parse_set(items):
    """--set planner.reverse_penalty=2.0 obstacle.x=0.5 → 巢狀 dict。"""
    out = {}
    for it in items or []:
        key, val = it.split('=', 1)
        node = out
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out
