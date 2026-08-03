#!/bin/zsh
cd /Users/stefandulgheru/projects/ROGII
for g in 35 40 50 55 60; do
  LEG_GS=$g LEG_OUT=leg_$g.pkl python3 legbuild.py 2>/dev/null | tail -1
done
