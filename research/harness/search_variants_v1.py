"""研究用：第一輪變體搜尋（base_sim 的規劃器，固定迭代上限）。用法：python search_variants_v1.py ../grids/grid1.json 3,4,5,8,10,13 out.json"""
import itertools, json, math, sys, time
from multiprocessing import Pool
import evalsim

SLIDES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3', '4', '5', '8', '10', '13']


def job(args):
    th, kw = args
    tot = 0; det = {}
    for s in SLIDES:
        try:
            r = evalsim.run(s, math.radians(th), kw, None)
            sc = evalsim.score(s, r)
            e = sc['g'] + sc['gcov']
            det[s] = (round(e, 3), r['found'], r['iters'])
        except Exception as ex:
            e = 9; det[s] = (9, False, str(ex))
        tot += e
    return dict(th=th, kw=kw, tot=round(tot, 3), det=det)


if __name__ == '__main__':
    grid = json.loads(open(sys.argv[1]).read())
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, vals)); th = d.pop('theta0')
        combos.append((th, d))
    print('combos', len(combos), flush=True)
    t0 = time.time()
    out = []
    with Pool(10) as p:
        for i, r in enumerate(p.imap_unordered(job, combos, chunksize=2)):
            out.append(r)
            if i % 50 == 0:
                print(i, '%.0fs' % (time.time() - t0), flush=True)
    out.sort(key=lambda r: r['tot'])
    json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else 'batch_out.json', 'w'), indent=0)
    for r in out[:25]:
        print(r['tot'], r['th'], r['kw'], r['det'])
