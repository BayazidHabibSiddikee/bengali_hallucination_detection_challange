#!/usr/bin/env bash
# Push the notebook to Kaggle, wait for the run, download output, submit.
#
# Prerequisite (one-time, in a browser): accept the competition rules —
#   open the invite link https://www.kaggle.com/t/5c9503557c4c404f846028899ea02ce7
#   (or https://www.kaggle.com/competitions/bengali-hallucination/rules) and click
#   "I Understand and Accept". Without this every API call returns 403.
#
# Usage: ./push_and_submit.sh ["submission message"]
set -euo pipefail
cd "$(dirname "$0")"

KERNEL="bayazidhabibsiddikee/bengali-hallucination-pipeline"
COMP="bengali-hallucination"
MSG="${1:-hybrid regime pipeline: NLI + RAG + LLM judge + LGBM meta}"

echo "== rebuilding notebook =="
python3 build_notebook.py

echo "== pushing kernel =="
kaggle kernels push -p .

echo "== waiting for kernel run (GPU queue + ~30-60 min runtime) =="
while true; do
    STATUS=$(kaggle kernels status "$KERNEL" 2>&1 | grep -v Warning || true)
    echo "$(date +%H:%M:%S)  $STATUS"
    case "$STATUS" in
        *COMPLETE*) break ;;
        *ERROR*|*CANCEL*)
            echo "Kernel failed — fetching log:"
            mkdir -p output && kaggle kernels output "$KERNEL" -p output || true
            exit 1 ;;
    esac
    sleep 90
done

echo "== downloading output =="
mkdir -p output
kaggle kernels output "$KERNEL" -p output

test -s output/submission.csv || { echo "submission.csv missing in kernel output"; exit 1; }
head -3 output/submission.csv
echo "rows: $(wc -l < output/submission.csv)"

echo "== submitting =="
kaggle competitions submit -c "$COMP" -f output/submission.csv -m "$MSG"

echo "== recent submissions =="
sleep 10
kaggle competitions submissions -c "$COMP" | head -8
