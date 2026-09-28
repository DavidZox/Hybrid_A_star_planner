# SMAC Planner（Hybrid-A* + ROI + Pure Pursuit）：簡報結果重現

依據 `SMAC_planner.pptx` 重建「Nav2 Smac Planner with ROI Trajectory Tracking Simulation」的 Python 模擬，重現簡報 p.3–p.13 的 11 組 Hybrid A* 測試：障礙物位置、steering / reverse / change_dir 三種懲罰，以及終點距離容許值。
原始程式已遺失，簡報上沒有寫出的細節（起始朝向、代價組合方式、終點角度條件、控制器參數等）是從 11 張結果圖量測與反推的，推導過程見 §6 與 [research/NOTES.md](research/NOTES.md)。

測試環境：WSL2、Python 3.12.14（conda `common_env`）、numpy 2.5.3、matplotlib 3.11.2、scipy 1.18.1。只用 CPU，全部重跑約 2 分鐘。

---

## 1. 重現結果摘要

| 實驗（簡報頁） | 參數變化 | 簡報中的路徑 | 重現結果 | 路徑偏差 | 軌跡偏差 | 判定 |
|---|---|---|---|---|---|---|
| T01（p.3） | 障礙物 x=0.1（基準） | 倒車直下 → 左下換向 → 前進接入 | 同樣一次換向，但下降段偏右（x≈1.3，簡報 x≈0.85） | 19.9 cm | 12.9 cm | △ |
| T02（p.4） | 障礙物 x=0.5 | 貼著障礙物倒車直下 → 換向 → 前進 | 結構相同，下降段偏右 | 10.1 | 6.1 | △ |
| T03（p.5） | 障礙物 x=0.9 | 先右轉繞過障礙物 → 倒車下降 → 換向 → 前進 | 相同 | 7.7 | 5.8 | ✓ |
| T04（p.6） | 障礙物 x=1.2 | 由障礙物左側倒車下降 → 換向 → 前進 | 相同 | 5.0 | 3.7 | ✓ |
| T05（p.7） | 障礙物 x=1.5 | 與 p.3 相同 | 結構相同，換向點偏右 | 8.2 | 6.2 | △ |
| T06（p.8） | steering 0.1→0.8 | 繞右側一路倒車進入 | 相同（終點多 0.14 m 前進接回） | 3.9 | 3.3 | ✓ |
| T07（p.9） | steering 0.1→2.0 | 同 p.8 但更平滑，略越過入口 | 相同 | **1.8** | **1.4** | ✓ |
| T08（p.10） | reverse 0.3→2.0 | 全程前進，上方右迴轉後往下接入 | 相同 | 3.2 | 2.1 | ✓ |
| T09（p.11） | reverse 0.3→10.0 | 全程前進，迴轉幅度較大，未滿足終點角度 | 全程前進，迴轉較小、正常抵達 | 10.8 | 10.4 | △ |
| T10（p.12） | change_dir 0.8→0.1 | 起點短倒車 → 前進斜下 → 倒車（停在入口前 0.23 m） | 前三段相同，之後再換向接回入口（3 次換向） | 6.2 | 4.6 | ✓ |
| T11（p.13） | change_dir 0.1＋終點容許 1 m | 倒車繞障礙物左側 → 前進 → 約 1 m 直線接入 | 在右側 1 m 內就以直線接入 | 48.6 | 32.6 | ✗ |

偏差是簡報原圖數位化曲線與重現曲線的雙向平均距離；線寬約 3 cm，完全重合時約 1 cm。✓ < 8 cm 且路線相同；△ 路線結構相同但位置或細節不同；✗ 路線不同。

![原圖曲線（淺色點）與重現曲線疊圖](results/comparison/overlay_all.png)

- **6 組吻合、4 組結構相同、1 組不同**。參數效果的趨勢都與簡報一致：steering_penalty 變大 → 改成倒車直接進入、路徑更平滑；reverse_penalty 變大 → 全程前進繞圈；change_dir_penalty 變小 → 換向次數增加；終點容許 1 m → 最後一段變成長直線。
- **重現的圖面樣式與原圖相同**：配色、線型、圖層順序、座標範圍（x −0.5–5.5、y −2.225–2.5，軸框 587×462 px 與原圖一致）、機器人停止位置（x = 4.90–4.92，原圖約 4.91）。每組的並排比較在 `results/comparison/<實驗>.png`。
- **差異來源**：原始程式遺失，搜尋細節只能反推。以 p.3 為例，左右兩條下降路線在 A* 的 f 值上估算只差約 0.07，路線選擇對實作細節非常敏感；約 9,300 個實作變體中，沒有一個能同時重現全部 11 頁（見 §6）。
- **依圖判斷，p.11、p.12 是「未達成終點」的 best-effort 路徑**（p.11 抵達時車頭朝下、p.12 停在入口前 0.23 m）；重現模型在 30000 次迭代內找到了合法終點，所以尾段不同。程式保留同樣的退回機制：超過 `max_iterations` 時回傳 h 最小節點的路徑（例如 §5.4 中 reverse_penalty = 0.8–1.5 的情況）。

---

## 2. 資料夾結構

```
Hybrid_A_star_planner/
├── README.md
├── SMAC_planner.pptx              原始簡報
├── run_all.sh                     一鍵重現全部流程
├── requirements.txt
├── configs/
│   ├── base.yaml                  基準參數（含反推值與來源註記）
│   └── experiments/               11 組測試，只寫與 base 不同的欄位（test01…test11，對應 p.3–p.13）
├── smac_sim/                      模擬程式（Python 套件）
│   ├── costmap.py                 代價地圖：膨脹代價、圓形碰撞檢查
│   ├── planner.py                 Smac Hybrid-A*：ROI、6 種運動原語、代價、解析擴展、best-effort
│   ├── controller.py              路徑分段、Pure Pursuit、狀態機
│   ├── simulation.py              串起規劃與追蹤、計算指標
│   ├── metrics.py / reference.py  指標計算；讀取原圖數位化曲線並計算偏差
│   ├── visualization.py           簡報樣式結果圖、搜尋樹圖、GIF
│   └── config.py / plotstyle.py   設定檔讀取；報表圖樣式
├── scripts/
│   ├── extract_slide_reference.py 取出簡報圖片與參數文字，數位化 p.3–p.13 的曲線
│   ├── run_experiments.py         執行實驗（全部、單一或自訂參數）
│   ├── compare_with_slides.py     與原圖並排比較、疊圖、偏差表
│   ├── param_sweep.py             參數敏感度掃描
│   └── make_report.py             指標圖與 results/summary.md
├── reference/                     slides/（各頁原圖）、digitized/（數位化曲線）、slide_annotations.yaml（人工判讀的路徑特徵）
├── results/
│   ├── test01…test11/             figure.png、search_tree.png、animation.gif、smac_path.csv、trajectory.csv、metrics.json、config.yaml
│   ├── comparison/                原圖 vs 重現（逐頁與總覽）
│   ├── figures/                   各指標比較圖、敏感度圖、結果總覽
│   ├── metrics.csv、reproduction_check.csv、sensitivity.csv
│   └── summary.md                 所有表格與圖的彙整
├── research/                      反推過程：harness/（變體規劃器與搜尋腳本）、grids/、runs/（約 9,300 個變體的評分）、figures/（診斷圖）、NOTES.md
└── logs/                          run_all.sh 各步驟的輸出
```

---

## 3. 部署方式

**本機**（套件已在 `common_env`）：

```bash
conda activate common_env
cd /home/david/Hybrid_A_star_planner
./run_all.sh
```

**新機器**：

```bash
conda create -n smac_sim python=3.12 -y && conda activate smac_sim
pip install -r requirements.txt
./run_all.sh
```

- 不需要 GPU、ROS 或網路；Windows 可直接執行 `run_all.sh` 裡的五個 python 指令。
- `run_all.sh` 把 matplotlib 的設定與字型快取放在專案內的 `.mplconfig/`。
- 報表圖的中文會自動使用系統上的 CJK 字型（Noto Sans CJK、微軟正黑體、文泉驛正黑等）；簡報樣式的結果圖只有英文，不受字型影響。

---

## 4. 使用方式

### 4.1 分步執行

| 步驟 | 指令 | 產出 |
|---|---|---|
| 1 參考資料 | `python scripts/extract_slide_reference.py` | `reference/slides/`、`reference/digitized/` |
| 2 實驗 | `python scripts/run_experiments.py [--gif]` | `results/test*/`、`results/metrics.csv` |
| 3 比對 | `python scripts/compare_with_slides.py` | `results/comparison/`、`results/reproduction_check.csv` |
| 4 敏感度 | `python scripts/param_sweep.py` | `results/sensitivity.csv`、`results/figures/sensitivity.png` |
| 5 報表 | `python scripts/make_report.py` | `results/figures/`、`results/summary.md` |

### 4.2 單一實驗與自訂參數

```bash
# 只跑 p.5 的設定並輸出動畫
python scripts/run_experiments.py --exp configs/experiments/test03_obstacle_x0.9.yaml --gif

# 以 base.yaml 為基礎臨時改參數，結果在 results/my_case/
python scripts/run_experiments.py --name my_case --set obstacle.x=0.7 planner.reverse_penalty=1.0 tracking.default_speed=0.2
```

`--set` 可改 `configs/base.yaml` 中的任何欄位，例如 `start.theta_deg`、`planner.max_iterations`、`planner.goal_dist_tolerance`、`tracking.lookahead`。要保存新的測試，複製 `configs/experiments/` 裡的任一檔修改即可。

### 4.3 在 Python 中使用

```python
from smac_sim import load_config, run_scenario
from smac_sim.visualization import plot_result

cfg = load_config('configs/experiments/test07_steering_penalty2.0.yaml',
                  overrides={'planner': {'steering_penalty': 1.2}})
res = run_scenario(cfg)
print(res['metrics']['found'], res['metrics']['path_length'], res['metrics']['cusps'])
plot_result(res, 'my_result.png')
# res['smac_path']：(N, 4) x, y, theta, direction；res['trajectory']：(M, 3) x, y, theta
```

也可以單獨使用規劃器：`SmacHybridAStar(costmap, PlannerParams(...)).plan(start, goal)`，回傳路徑、是否達成終點、迭代數與展開節點。

---

## 5. 各指標結果圖

### 5.1 每組實驗的輸出

| 檔案 | 內容 |
|---|---|
| `figure.png` | 與簡報同樣式的結果圖（總覽：[overview_reproduced.png](results/figures/overview_reproduced.png)） |
| `search_tree.png` | Hybrid-A* 展開過的節點（顏色 = 展開順序）與最終路徑的前進/倒車步 |
| `animation.gif` | 追蹤過程動畫，標題顯示狀態機目前狀態 |
| `smac_path.csv`、`trajectory.csv`、`metrics.json` | 規劃路徑、實際軌跡（含狀態）、所有指標 |

### 5.2 規劃路徑

![規劃路徑指標](results/figures/metrics_path.png)

- **障礙物位置（T01–T05）**：都是「倒車 → 一次換向 → 前進進入」。障礙物在 x = 0.9 時必須先右轉繞過（累積轉角 243°，最大）；x = 1.2 時從左側繞，路徑最長（2.54 m）。
- **steering_penalty（T06、T07）**：由「底部換向再前進」改成「倒車直接進入入口」，前進段只剩 0.09–0.14 m，累積轉角由 243° 降到 219°。
- **reverse_penalty（T08、T09）**：倒車代價高到不值得，改為全程前進、先往上迴轉，路徑 2.41 m、累積轉角 333°。2.0 與 10.0 的結果相同。
- **change_dir_penalty 0.1（T10）**：換向變便宜，出現 3 次換向（簡報為 2 次後停止）。
- **終點容許 1 m（T11）**：搜尋在第 1086 次展開就結束，最後約 1 m 是不考慮車頭角的直線。

### 5.3 搜尋成本與追蹤

![搜尋成本](results/figures/metrics_search.png)
![追蹤控制指標](results/figures/metrics_tracking.png)

- 展開節點 1,086–27,109 個，Python 規劃時間約 10–500 ms（每次執行略有差異）；T07（steering 2.0）最多，因為打角代價高時要展開更多狀態才找到平滑解。
- 追蹤 Smac 路徑的平均橫向誤差 1.0–5.0 cm。最大誤差出現在短子段：T06 終點前 0.14 m 的前進段、T10 的多次換向，Pure Pursuit 在長度小於前視距離的子段會繞一個小圈（見動畫）。
- 機器人外緣與障礙物的最小間距在 T04、T05 只有 9–10 cm（實際軌跡 7–9 cm）：代價地圖權重 0.01 對路徑的推力很小，簡報原圖同樣緊貼碰撞邊界。
- 所有實驗都停在主路徑終點前 0.085–0.098 m（完成條件 < 0.1 m），全程約 34–43 秒（v = 0.15 m/s）。

### 5.4 與原圖的偏差、參數敏感度

![與原圖的偏差](results/figures/reproduction_error.png)

![參數敏感度](results/figures/sensitivity.png)

- 在簡報測過的數值之間補點：reverse_penalty 在 0.5 與 0.8 之間從「以倒車為主」跳到「以前進為主」；0.8–1.5 時 30000 次迭代內找不到合法終點（空心點，回傳 best-effort 路徑）。
- change_dir_penalty ≤ 0.1 時 3 次換向、0.4–1.2 時 1 次、≥ 1.6 時改為全程前進。
- 障礙物 x = 1.1 時搜尋也會失敗：障礙物幾乎在起點正下方，往左或往右都要大幅繞行，展開數衝到上限。

完整數字（含各項最大值、進入主路徑時的位置/角度誤差）見 [results/summary.md](results/summary.md) 與 `results/*.csv`。

---

## 6. 反推說明（原始程式遺失）

| 來源 | 參數 |
|---|---|
| 簡報文字（p.3–13） | step_size 0.06、reverse_penalty 0.3、change_dir_penalty 0.8、steering_penalty 0.1、終點條件 `dist < 0.1 and angle_diff < 45° and goal_angle_diff < 45°`、障礙物半徑 0.2／膨脹 0.35 |
| 概念圖（p.2） | 解析度 0.05、θ 以 8° 離散、±12° 打角、運動方程式、h = 距離 + 0.8·\|Δθ\|、膨脹代價公式、ROI 公式、Pure Pursuit `w = clip(2.5α, −1, 1)`、狀態機 |
| 由圖量測 | 起點 (1, 1)、入口 (1.5, −0.5)、主路徑到 (5.0, −0.5)、costmap 範圍、ROI margin 1.5、機器人半徑 0.22、圖面樣式 |
| 反推 | **起始朝向 75°**（5 種路線的第一段方向誤差都 < 1°）；**直線解析擴展需無碰撞**（p.13）；**goal_angle_diff = 行進方向與終點朝向的夾角**（只接受前進抵達，p.9 的越過行為）；代價：倒車 `step×1.3`、打角 `+0.1×step`、換向 `+0.8`、地圖 `+0.01×cost`；`max_iterations` 30000、失敗時回傳 h 最小節點 |
| 擬合 | Pure Pursuit：v 0.15 m/s、前視 0.2 m、dt 0.1 s、子段切換 0.15 m、終點 0.1 m、主路徑每 0.1 m 取樣 |

反推方法：先把 11 張圖的座標校正（97.7 px/m）並數位化綠色與藍色曲線，再把每個不確定的實作細節做成開關（代價形式、封閉集離散方式、終點條件、啟發函數、迭代上限等），逐一與 11 頁比對。細節與批次結果見 [research/NOTES.md](research/NOTES.md)。

---

## 7. 建議

1. **找到原始程式時優先核對三點**：goal_angle_diff 的定義、倒車／打角懲罰是加法還是乘以步長、`max_iterations` 與搜尋失敗時的回傳方式。`research/harness/quick_eval.py` 可以在幾秒內評估任一組合與 11 頁的偏差。
2. **改善啟發函數**：0.8·|Δθ| 不是可採納的啟發值，搜尋容易在起點附近打轉，敏感度圖中有多處在 30000 次內失敗。可改用 Reeds-Shepp 距離與 2D 障礙物距離兩者取大（Nav2 的做法），或降低角度權重。
3. **解析擴展改用 Reeds-Shepp／Dubins 曲線**：直線不考慮車頭角與轉彎半徑，T11 的 1 m 直線抵達入口時角度偏差大，主路徑還要再修正。
4. **短子段處理**：換向後的子段若短於前視距離（0.2 m），Pure Pursuit 會繞圈。可合併或略過過短子段、在尖點附近降速，或改用原地對位控制。
5. **安全距離**：地圖代價權重 0.01 時路徑只離障礙物 9–10 cm，實車建議提高 `cost_penalty`（例如 0.05–0.1）或加大膨脹半徑，並用 `param_sweep.py` 檢查路徑是否仍能在迭代上限內找到。
6. **效能**：純 Python 單次規劃 0.01–0.5 s，適合離線測試；上車請改用 Nav2 `SmacPlannerHybrid`（C++）或以 numba／C++ 改寫展開迴圈。

---

## 8. 常見問題

- **為什麼 T08 與 T09 結果一樣？** reverse_penalty ≥ 2 時倒車已經不划算，兩者都是全程前進。簡報 p.11 的迴轉較大，且抵達入口時車頭朝下、不滿足終點條件，推測原程式在該設定下是 best-effort 結果。
- **路徑終點附近有一小段往回的線？** 那是解析擴展：節點在入口 0.1 m 內滿足條件後，以直線接到入口；若節點已越過入口，直線就會往回。
- **`best-effort` 是什麼？** 超過 `planner.max_iterations` 仍找不到合法終點時，回傳目前 h 最小節點的路徑，`metrics.json` 的 `found` 為 false，搜尋樹圖標題也會註明。
- **只想看圖不想跑程式？** 所有結果都已存在 `results/`，從 [results/summary.md](results/summary.md) 開始看。
