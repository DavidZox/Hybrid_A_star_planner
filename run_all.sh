#!/usr/bin/env bash
# 一鍵重現：抽取簡報參考資料 → 11 組實驗（含 GIF）→ 與原圖比對 → 參數敏感度 → 報表
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.mplconfig"      # matplotlib 設定與字型快取放在專案內
mkdir -p logs
python scripts/extract_slide_reference.py 2>&1 | tee logs/01_extract_reference.log
python scripts/run_experiments.py --gif     2>&1 | tee logs/02_run_experiments.log
python scripts/compare_with_slides.py       2>&1 | tee logs/03_compare_with_slides.log
python scripts/param_sweep.py               2>&1 | tee logs/04_param_sweep.log
python scripts/make_report.py               2>&1 | tee logs/05_make_report.log
echo "完成：結果在 results/，摘要見 results/summary.md"
