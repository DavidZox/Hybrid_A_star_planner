"""從 SMAC_planner.pptx 取出各頁圖片與參數文字，並把第 3–13 頁的結果圖數位化。

輸出（全部在 reference/）：
  slides/slideNN_*.png        各頁內嵌的原始圖片
  slides_text.json            各頁標題與參數文字
  digitized/slideNN.json      座標校正結果、綠色規劃路徑與藍色實際軌跡的世界座標點

數位化方式：以 x/y 軸刻度線校正像素與公尺的對應（兩軸等比例，約 97.7 px/m），
再依顏色取出綠色虛線（'g--'）與藍色實線（'b-'）的像素並換算成公尺座標。
"""
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / 'SMAC_planner.pptx'
OUT = ROOT / 'reference'


def slide_media(z):
    """回傳 {頁碼: [圖片路徑,...]}，依投影片的關聯檔（rels）對應。"""
    res = {}
    for name in z.namelist():
        m = re.match(r'ppt/slides/_rels/slide(\d+)\.xml\.rels$', name)
        if not m:
            continue
        rels = z.read(name).decode('utf-8')
        res[int(m.group(1))] = ['ppt/' + t.replace('../', '') for t in re.findall(r'Target="(\.\./media/[^"]+)"', rels)]
    return dict(sorted(res.items()))


def slide_text(z, n):
    xml = z.read(f'ppt/slides/slide{n}.xml').decode('utf-8')
    paras = []
    for p in re.findall(r'<a:p>.*?</a:p>', xml, flags=re.S):
        t = ''.join(re.findall(r'<a:t>([^<]*)</a:t>', p)).strip()
        if t:
            paras.append(t.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&'))
    return paras


def _groups(idx):
    groups, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 1:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def calibrate(a):
    """由座標軸外框與刻度線求出像素 ↔ 公尺的線性關係。"""
    dark = a.sum(axis=2) < 200
    rows = np.where(dark.sum(axis=1) > 300)[0]
    top, bot = int(rows[0]), int(rows[-1])
    right = int(np.where(dark[top + 5:bot - 5].sum(axis=0) > 300)[0][0])
    xt = _groups(np.where(dark[bot + 2:bot + 5].sum(axis=0) >= 2)[0])      # x = 0..5 的刻度
    sx, x0 = np.polyfit(np.arange(len(xt)), xt, 1)
    left = int(round(x0 - 0.5 * sx))                                       # x 軸下限為 -0.5
    yt = _groups(np.where(dark[:, left - 5:left - 1].sum(axis=1) >= 3)[0])  # y = 2,1,0,-1,-2 的刻度
    k, y0 = np.polyfit([2, 1, 0, -1, -2], yt, 1)
    return dict(top=top, bottom=bot, left=left, right=right, sx=float(sx), x0=float(x0), sy=float(-k), y0=float(y0))


def digitize(a, c):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    H, W = R.shape
    rr, cc = np.mgrid[0:H, 0:W]
    inside = (rr > c['top'] + 1) & (rr < c['bottom'] - 1) & (cc > c['left'] + 1) & (cc < c['right'] - 1)
    wx = (cc - c['x0']) / c['sx']
    wy = (c['y0'] - rr) / c['sy']
    legend = (wx > 2.9) & (wy > 1.2)                                        # 右上角圖例
    green = (G > 90) & (R < 90) & (B < 90) & (G - R > 50) & (G - B > 50) & inside & ~legend
    blue = (B > 200) & (R < 80) & (G < 80) & inside & ~legend
    # 綠色虛線的每一段（中心、方向），用來判斷起點朝向與尖點
    lab, n = ndimage.label(green)
    dashes = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 4:
            continue
        P = np.stack([wx[m], wy[m]], 1)
        mu = P.mean(0)
        _, _, vt = np.linalg.svd(P - mu)
        dashes.append([round(float(mu[0]), 4), round(float(mu[1]), 4), int(m.sum()),
                       round(float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0]))), 1)])
    r4 = lambda v: np.round(v, 4).tolist()
    return dict(green=r4(np.stack([wx[green], wy[green]], 1)), blue=r4(np.stack([wx[blue], wy[blue]], 1)),
                dashes=dashes)


def main():
    if not PPTX.exists():
        sys.exit(f'找不到 {PPTX}')
    (OUT / 'slides').mkdir(parents=True, exist_ok=True)
    (OUT / 'digitized').mkdir(parents=True, exist_ok=True)
    texts = {}
    with zipfile.ZipFile(PPTX) as z:
        media = slide_media(z)
        for n, files in media.items():
            texts[n] = slide_text(z, n)
            for f in files:
                dst = OUT / 'slides' / f'slide{n:02d}_{Path(f).name}'
                with z.open(f) as src, open(dst, 'wb') as out:
                    shutil.copyfileobj(src, out)
                if n >= 3:
                    a = np.array(Image.open(dst).convert('RGB')).astype(int)
                    c = calibrate(a)
                    d = digitize(a, c)
                    d.update(slide=n, image=dst.name, calibration=c)
                    (OUT / 'digitized' / f'slide{n:02d}.json').write_text(json.dumps(d))
                    print(f'slide {n:2d}: {dst.name}  {c["sx"]:.2f}/{c["sy"]:.2f} px/m  '
                          f'green {len(d["green"])} px, blue {len(d["blue"])} px, dashes {len(d["dashes"])}')
                else:
                    print(f'slide {n:2d}: {dst.name}')
    (OUT / 'slides_text.json').write_text(json.dumps(texts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
