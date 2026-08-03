#!/bin/zsh
cd /Users/stefandulgheru/projects/ROGII
for cfg in '{}' '{"GS_HI":40}' '{"GS_HI":90}' '{"GS_HI":120}' '{"GS_HI":200}' '{"GS_LO":30,"GS_HI":60}' '{"GS_LO":60,"GS_HI":60}'; do
  echo "=== $cfg ==="
  python3 pfcv.py 200 "$cfg" 96 2>/dev/null | grep -E "pooled CV|leg RMSE"
done
