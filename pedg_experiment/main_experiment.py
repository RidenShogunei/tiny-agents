#!/usr/bin/env python3
"""
PEDG Experiment - Main Entry Point
Private-Evidence Delegation Game: Goal Misalignment Detection

Usage:
    python main_experiment.py --mode smoke       # Quick smoke test
    python main_experiment.py --mode full        # Full experiment
    python main_experiment.py --mode analyze     # Only run analysis
    python main_experiment.py --mode report       # Generate report
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_gpu():
    """Check GPU availability"""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        logger.info(f"GPU Status:\n{result.stdout}")
        return True
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return False


def start_vllm_if_needed(config):
    """Check if vLLM is already serving the target model"""
    import requests
    
    api_base = config["models"]["api_base"]
    model_name = config["models"]["model_name"]
    
    try:
        resp = requests.get(f"{api_base.rsplit('/v1', 1)[0]}/v1/models", timeout=5)
        models = resp.json().get("data", [])
        logger.info(f"Available models: {[m['id'] for m in models]}")
        return True
    except Exception as e:
        logger.warning(f"vLLM check failed: {e}")
    
    return False


def run_smoke_test(config, gpu_id=1):
    """Run smoke test"""
    from experiments.run_single import ExperimentRunner
    
    logger.info("=" * 60)
    logger.info("SMOKE TEST MODE")
    logger.info("=" * 60)
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    runner = ExperimentRunner(
        config=config,
        gpu_id=gpu_id,
        output_dir=str(PROJECT_ROOT / "results"),
    )
    
    results = runner.run_smoke_test(episodes_per_cell=20)
    output = runner.save_results(tag="smoke_test")
    
    logger.info(f"\nSmoke test complete: {len(results)} episodes")
    
    # Quick stats
    df = pd.DataFrame([r.to_dict() for r in results])
    
    logger.info("\n=== QUICK METRICS ===")
    metrics = ["misreport", "omission", "p_utility", "audit_triggered", "knowledge_deviation"]
    
    for condition in ["aligned", "conflict"]:
        for info_struct in ["asymmetric", "symmetric"]:
            for oversight in ["weak_oversight", "strong_oversight"]:
                mask = (
                    (df["condition"] == condition) &
                    (df["info_structure"] == info_struct) &
                    (df["oversight"] == oversight)
                )
                subset = df[mask]
                if len(subset) > 0:
                    mis = subset["misreport"].mean()
                    om = subset["omission"].mean()
                    pu = subset["p_utility"].mean()
                    logger.info(
                        f"  {condition[:4]}_{info_struct[:4]}_{oversight[:4]}: "
                        f"mis={mis:.2f} om={om:.2f} pU={pu:.2f} n={len(subset)}"
                    )
    
    return results, df


def run_full(config, gpu_id=1, family_offset=0):
    """Run full experiment"""
    from experiments.run_single import ExperimentRunner
    
    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT MODE")
    logger.info("=" * 60)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    runner = ExperimentRunner(
        config=config,
        gpu_id=gpu_id,
        output_dir=str(PROJECT_ROOT / "results"),
        family_offset=family_offset,
    )
    
    results = runner.run_full_experiment()
    output = runner.save_results(tag="full")
    
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Save detailed summary
    summary = {
        "n_episodes": len(results),
        "n_families": df["task_family"].nunique(),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Per-cell summary
    cell_summary = df.groupby(["condition", "info_structure", "oversight"]).agg({
        "misreport": ["mean", "std"],
        "omission": ["mean", "std"],
        "p_utility": ["mean", "std"],
        "s_utility": ["mean", "std"],
        "audit_triggered": "mean",
        "knowledge_deviation": "mean",
        "principal_regret": "mean",
    }).round(4)
    
    cell_summary_path = PROJECT_ROOT / "results" / "cell_summary.csv"
    cell_summary.to_csv(cell_summary_path)
    logger.info(f"Cell summary saved to {cell_summary_path}")
    
    return results, df


def run_analysis(results_path=None):
    """Run statistical analysis"""
    from analysis.stats import StatisticalAnalysis
    
    logger.info("=" * 60)
    logger.info("ANALYSIS MODE")
    logger.info("=" * 60)
    
    analysis = StatisticalAnalysis(project_root=str(PROJECT_ROOT))
    
    if results_path:
        df = analysis.load_results(results_path)
    else:
        # Find latest results
        results_dir = PROJECT_ROOT / "results"
        jsonl_files = list(results_dir.glob("episode_logs_*.jsonl"))
        if not jsonl_files:
            logger.error("No results files found")
            return None
        latest = sorted(jsonl_files)[-1]
        df = analysis.load_results(str(latest))
    
    report = analysis.generate_report(df)
    
    # Save analysis results
    analysis_path = PROJECT_ROOT / "results" / "analysis_report.md"
    with open(analysis_path, "w") as f:
        f.write(report)
    logger.info(f"Analysis report saved to {analysis_path}")
    
    return report


def generate_reports():
    """Generate final report"""
    from analysis.stats import StatisticalAnalysis
    
    logger.info("=" * 60)
    logger.info("REPORT GENERATION MODE")
    logger.info("=" * 60)
    
    # Find latest results
    results_dir = PROJECT_ROOT / "results"
    jsonl_files = list(results_dir.glob("episode_logs_full_*.jsonl"))
    if not jsonl_files:
        jsonl_files = list(results_dir.glob("episode_logs_*.jsonl"))
    
    if not jsonl_files:
        logger.error("No results files found")
        return
    
    latest = sorted(jsonl_files)[-1]
    logger.info(f"Using results from: {latest}")
    
    analysis = StatisticalAnalysis(project_root=str(PROJECT_ROOT))
    df = analysis.load_results(str(latest))
    
    # Generate reports
    analysis.generate_report(df)
    analysis.generate_executive_summary(df)
    analysis.generate_final_report(df)
    
    logger.info("Reports generated!")


def main():
    parser = argparse.ArgumentParser(description="PEDG Experiment Runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "analyze", "report"],
        default="smoke",
        help="Run mode",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--family-offset",
        type=int,
        default=0,
        help="Skip first N families (for parallel splitting)",
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to results JSONL for analysis/report mode",
    )
    
    args = parser.parse_args()
    
    # Check GPU
    check_gpu()
    
    # Load config
    config = load_config()
    
    if args.mode == "smoke":
        results, df = run_smoke_test(config, gpu_id=args.gpu)
    elif args.mode == "full":
        results, df = run_full(config, gpu_id=args.gpu, family_offset=args.family_offset)
    elif args.mode == "analyze":
        report = run_analysis(args.results)
        print(report)
    elif args.mode == "report":
        generate_reports()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
