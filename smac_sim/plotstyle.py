"""報表圖共用樣式：中性色字、細格線、固定分類色序；中文字自動改用可用的 CJK 字型。"""
import logging

import matplotlib
from matplotlib import font_manager

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

INK = '#0b0b0b'          # 主要文字
INK_2 = '#52514e'        # 次要文字
GRID = '#e5e4df'         # 格線（實線細線）
SURFACE = '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # 分類色序：藍、橘、綠松、黃（依序使用，不循環）

_CJK_CANDIDATES = ['Noto Sans CJK TC', 'Noto Sans TC', 'Microsoft JhengHei', 'WenQuanYi Zen Hei',
                   'PingFang TC', 'Droid Sans Fallback']


def use_report_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in _CJK_CANDIDATES if n in installed]
    matplotlib.rcParams.update({
        'font.family': ['DejaVu Sans'] + cjk,
        'axes.edgecolor': '#c9c8c2',
        'axes.labelcolor': INK_2,
        'axes.titlecolor': INK,
        'xtick.color': INK_2,
        'ytick.color': INK_2,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'legend.frameon': False,
    })
    return cjk
