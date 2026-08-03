#!/bin/zsh
cd /Users/stefandulgheru/projects/ROGII
for off in 0 250; do
for cfg in '{}' '{"GS_LO":45,"GS_HI":45}' '{"GS_LIST":[35,45,55]}' '{"GS_LIST":[30,40,50,60]}' '{"GS_LIST":[40,45,50]}'; do
  echo "=== offset=$off $cfg ==="
  python3 pfcv.py 200 "$cfg" 96 $off 2>/dev/null | grep -E "pooled CV"
done; done
