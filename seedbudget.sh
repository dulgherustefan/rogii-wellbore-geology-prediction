#!/bin/zsh
cd /Users/stefandulgheru/projects/ROGII
for off in 0 250; do
for sd in 24 32 48; do
  echo "=== offset=$off gs=45 seeds=$sd ==="
  python3 pfcv.py 200 '{"GS_LO":45,"GS_HI":45}' $sd $off 2>/dev/null | grep -E "pooled CV"
done; done
