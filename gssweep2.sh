#!/bin/zsh
cd /Users/stefandulgheru/projects/ROGII
for g in 45 80 100 150 250 400; do
  echo "=== gs fixed $g ==="
  python3 pfcv.py 200 "{\"GS_LO\":$g,\"GS_HI\":$g}" 96 2>/dev/null | grep -E "pooled CV|leg RMSE"
done
