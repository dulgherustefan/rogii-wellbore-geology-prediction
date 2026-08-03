#!/bin/zsh
cd /Users/stefandulgheru/projects/ROGII
for g in 0 35 40 45 50 55; do
  if [ "$g" = "0" ]; then cfg='{}'; lbl='baseline'; else cfg="{\"GS_LO\":$g,\"GS_HI\":$g}"; lbl="gs=$g"; fi
  echo "=== $lbl (HOLDOUT wells 250-450) ==="
  python3 pfcv.py 200 "$cfg" 96 250 2>/dev/null | grep -E "pooled CV|leg RMSE"
done
