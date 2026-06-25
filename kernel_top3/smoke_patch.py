"""Patch the top-3 notebook code for a LOCAL CPU smoke test on a subset of wells.
Verifies feature building (beam/PF/imputers) + a tiny train run end-to-end, no crash."""
import re
src = open('kernel_top3/full.py').read()
src = src.replace('from __future__ import annotations\n', '')

# 1) data path -> local 'data'
src = src.replace(
    'for p in [Path("/kaggle/input/rogii-wellbore-geology-prediction"),',
    'for p in [Path("data"), Path("/kaggle/input/rogii-wellbore-geology-prediction"),')
# 2) output to /tmp
src = src.replace('OUT=Path("/kaggle/working/submission.csv")', 'OUT=Path("/tmp/sub_smoke.csv")')
# 3) numba cache dir local + skip pip
src = src.replace('os.environ["NUMBA_CACHE_DIR"]="/kaggle/working/.numba"', 'os.environ["NUMBA_CACHE_DIR"]="/tmp/.numba"')
src = src.replace('os.makedirs("/kaggle/working/.numba",exist_ok=True)', 'os.makedirs("/tmp/.numba",exist_ok=True)')
src = re.sub(r'for pkg in \["numba"\]:.*?--quiet"\]\)', 'pass', src, flags=re.S)
# 4) GPU -> CPU
src = src.replace('device_type="gpu", gpu_use_dp=False, max_bin=255,', 'max_bin=255,')
src = src.replace('task_type="GPU",', 'task_type="CPU",')
src = src.replace('devices="0:1",           # both T4', '')
# 5) tiny iterations
src = src.replace('learning_rate=0.025, n_estimators=8000, seed=42', 'learning_rate=0.05, n_estimators=120, seed=42')
src = src.replace('learning_rate=0.020, n_estimators=8000, seed=7', 'learning_rate=0.05, n_estimators=120, seed=7')
src = src.replace('learning_rate=0.030, n_estimators=8000, seed=123', 'learning_rate=0.05, n_estimators=120, seed=123')
src = src.replace('iterations=8000,', 'iterations=120,')
# nvidia-smi will fail locally -> guard
src = src.replace('print("GPUs:",_s.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],\n      capture_output=True,text=True).stdout.strip())',
                  'print("GPUs: (local cpu smoke)")')
# 6) subset train wells to 25 for speed (keep all test = 3)
src = src.replace('train_df=build_dataset(hw_paths,is_train=True,label="train")',
                  'train_df=build_dataset(hw_paths[:25],is_train=True,label="train")')

open('/tmp/top3_smoke.py', 'w').write(src)
print("patched -> /tmp/top3_smoke.py")
