#!/usr/bin/env python3
"""
PEDG Parallel Experiment Launcher
Splits 10 families across 3 GPUs for ~3x speedup.
"""
import os, sys, subprocess, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV = str(PROJECT_ROOT / "../venv/bin/python")
LOG_DIR = Path("/tmp/pedg_parallel")
LOG_DIR.mkdir(exist_ok=True)

GPU_CONFIGS = [
    {"gpu": 1, "offset": 0,  "count": 4},   # families 0-3
    {"gpu": 2, "offset": 4,  "count": 3},   # families 4-6
    {"gpu": 3, "offset": 7,  "count": 3},   # families 7-9
]

def kill_all():
    subprocess.run("pkill -f 'pedg_gpu' 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'main_experiment' 2>/dev/null", shell=True)
    print("Killed all processes.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--kill":
        kill_all(); return

    print("Stopping existing processes...")
    subprocess.run("pkill -f 'pedg_gpu' 2>/dev/null; pkill -f 'main_experiment.*--gpu' 2>/dev/null", shell=True)
    time.sleep(2)

    print("=" * 60)
    print("PEDG PARALLEL LAUNCHER")
    print("=" * 60)
    for c in GPU_CONFIGS:
        print(f"  GPU {c['gpu']}: families {c['offset']}–{c['offset']+c['count']-1} ({c['count']} families)")
    print("=" * 60)

    for c in GPU_CONFIGS:
        log_path = LOG_DIR / f"gpu{c['gpu']}.log"
        script_path = LOG_DIR / f"pedg_gpu{c['gpu']}.py"

        script = f'''
import subprocess, sys
from pathlib import Path
sys.path.insert(0, "{PROJECT_ROOT}")
sys.exit(subprocess.call([
    "{VENV}",
    str(Path("{PROJECT_ROOT}") / "main_experiment.py"),
    "--mode", "full",
    "--gpu", "{c['gpu']}",
    "--family-offset", "{c['offset']}",
]))
'''
        with open(script_path, "w") as f:
            f.write(script)

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                [VENV, str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        print(f"  GPU {c['gpu']}: PID {proc.pid}, families {c['offset']}–{c['offset']+c['count']-1}, log: {log_path}")

    print()
    print("Speedup: ~3x (50h -> ~17h)")
    print("Watch:   tail -f /tmp/pedg_parallel/gpu1.log")
    print("Stop:    python parallel_launcher.py --kill")

if __name__ == "__main__":
    main()
