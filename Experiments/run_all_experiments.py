"""
run_all_experiments.py
-----------------------
Master runner — executes all rebuttal experiments in priority order.

PRIORITY ORDER (most impact first):
  1. Ground-truth compilation (hours — immediate insight, no API cost)
  2. Ablation study          (days  — highest reviewer impact)
  3. Slither validation      (days  — external non-LLM evidence)
  4. Temperature sensitivity (days  — robustness evidence)
  5. Debiased metric         (days  — depends on ablation results)
  6. Generate tables         (minutes — final step)

Usage:
  # Run everything (full week plan):
  python run_all_experiments.py --dataset /path/to/requirement_code.jsonl

  # Run only the fast, high-priority experiments first:
  python run_all_experiments.py --dataset /path/to/requirement_code.jsonl \
      --experiments gt_compilation ablation

  # Run with smaller sample sizes for a quick test:
  python run_all_experiments.py --dataset /path/to/requirement_code.jsonl \
      --ablation_n 50 --slither_n 50 --temperature_n 20 --debiased_n 50

Environment:
  OPENAI_API_KEY must be set.

Estimated API cost (at gpt-4o-mini pricing ~$0.15/1M input tokens):
  Ablation (300 contracts × 4 conditions) ≈ $8–15
  Slither  (200 contracts × 2 conditions) ≈ $3–6
  Temperature (100 × 4 temps)             ≈ $2–4
  Debiased (200 pairs + LLM M4)           ≈ $2–4
  TOTAL                                   ≈ $15–30
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Load .env from workspace root before anything else so OPENAI_API_KEY is available
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent


def run_experiment(
    script: str,
    args_list: list,
    label: str,
) -> bool:
    """
    Run a single experiment script as a subprocess.
    Returns True if it completed without error, False otherwise.
    """
    cmd = [sys.executable, str(EXPERIMENTS_DIR / script)] + args_list
    print(f"\n{'#'*70}")
    print(f"# STARTING: {label}")
    print(f"# Command:  {' '.join(cmd)}")
    print(f"# Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, text=True)
    except KeyboardInterrupt:
        elapsed = round(time.time() - t0, 1)
        print(f"\n⚠ Interrupted during: {label} (after {elapsed}s)")
        print("  Results written up to the last completed contract are saved to disk.")
        raise
    elapsed = round(time.time() - t0, 1)

    if result.returncode == 0:
        print(f"\n✅ {label} completed in {elapsed}s")
        return True
    else:
        print(f"\n❌ {label} FAILED (exit code {result.returncode}) after {elapsed}s")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all KDD rebuttal experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to FSM-SCG requirement_code.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Root output directory (default: ./results)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "gt_compilation",
            "ablation",
            "slither",
            "temperature",
            "debiased",
            "contract_type",
            "tables",
        ],
        choices=[
            "gt_compilation",
            "ablation",
            "slither",
            "temperature",
            "debiased",
            "contract_type",
            "tables",
        ],
        help="Which experiments to run (default: all)",
    )
    # Sample size overrides for quick testing
    parser.add_argument("--ablation_n", type=int, default=300)
    parser.add_argument("--slither_n", type=int, default=200)
    parser.add_argument("--temperature_n", type=int, default=100)
    parser.add_argument("--debiased_n", type=int, default=200)
    parser.add_argument(
        "--gt_n", type=int, default=-1, help="-1 = all ground-truth contracts"
    )
    parser.add_argument(
        "--ct_n",
        type=int,
        default=30,
        help="Contracts per category for experiment 6 (default: 30)",
    )
    # Ablation conditions
    parser.add_argument(
        "--ablation_conditions",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        choices=["A", "B", "C", "D", "E"],
    )
    parser.add_argument(
        "--ablation_seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for ablation sampling (default: 42). Use '42 123 999' for 3-seed run.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.7, 1.0],
    )
    parser.add_argument(
        "--skip_llm_m4",
        action="store_true",
        help="Skip LLM-based M4 re-evaluation in debiased experiment (faster)",
    )
    args = parser.parse_args()

    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found at {args.dataset}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  KDD 2026 REBUTTAL EXPERIMENT SUITE")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Output:      {output_dir.resolve()}")
    print(f"  Model:       {args.model}")
    print(f"  Experiments: {', '.join(args.experiments)}")
    print("=" * 70)

    results = {}
    master_t0 = time.time()

    if "gt_compilation" in args.experiments:
        ok = run_experiment(
            "experiment_2_ground_truth_compilation.py",
            [
                "--dataset",
                args.dataset,
                "--n_samples",
                str(args.gt_n),
                "--output_dir",
                str(output_dir / "gt_compilation"),
            ],
            "Experiment 2: Ground-Truth Compilation Rate",
        )
        results["gt_compilation"] = "✅ OK" if ok else "❌ FAILED"

    if "ablation" in args.experiments:
        ok = run_experiment(
            "experiment_1_ablation.py",
            [
                "--dataset",
                args.dataset,
                "--n_samples",
                str(args.ablation_n),
                "--output_dir",
                str(output_dir / "ablation"),
                "--conditions",
                *args.ablation_conditions,
                "--seeds",
                *[str(s) for s in args.ablation_seeds],
                "--model",
                args.model,
            ],
            "Experiment 1: Ablation Study",
        )
        results["ablation"] = "✅ OK" if ok else "❌ FAILED"

    if "slither" in args.experiments:
        ok = run_experiment(
            "experiment_4_slither.py",
            [
                "--dataset",
                args.dataset,
                "--n_samples",
                str(args.slither_n),
                "--output_dir",
                str(output_dir / "slither"),
                "--model",
                args.model,
            ],
            "Experiment 4: Slither External Validation",
        )
        results["slither"] = "✅ OK" if ok else "❌ FAILED"

    if "temperature" in args.experiments:
        ok = run_experiment(
            "experiment_3_temperature.py",
            [
                "--dataset",
                args.dataset,
                "--n_samples",
                str(args.temperature_n),
                "--temperatures",
                *[str(t) for t in args.temperatures],
                "--output_dir",
                str(output_dir / "temperature"),
                "--model",
                args.model,
            ],
            "Experiment 3: Temperature Sensitivity",
        )
        results["temperature"] = "✅ OK" if ok else "❌ FAILED"

    if "debiased" in args.experiments:
        debiased_args = [
            "--dataset",
            args.dataset,
            "--n_samples",
            str(args.debiased_n),
            "--output_dir",
            str(output_dir / "debiased"),
            "--model",
            args.model,
        ]
        # Point to ablation condition C results if they exist
        cond_c = output_dir / "ablation" / "condition_C_iter1.jsonl"
        if cond_c.exists():
            debiased_args += ["--generated_results", str(cond_c)]
        if args.skip_llm_m4:
            debiased_args.append("--skip_llm_m4")

        ok = run_experiment(
            "experiment_5_debiased_metric.py",
            debiased_args,
            "Experiment 5: Debiased Metric Analysis",
        )
        results["debiased"] = "✅ OK" if ok else "❌ FAILED"

    if "contract_type" in args.experiments:
        ok = run_experiment(
            "experiment_6_contract_type.py",
            [
                "--dataset",
                args.dataset,
                "--n_per_category",
                str(args.ct_n),
                "--output_dir",
                str(output_dir / "contract_type"),
                "--model",
                args.model,
            ],
            "Experiment 6: Cross-Contract-Type Generalization",
        )
        results["contract_type"] = "✅ OK" if ok else "❌ FAILED"

    if "tables" in args.experiments:
        ok = run_experiment(
            "generate_rebuttal_tables.py",
            [
                "--results_dir",
                str(output_dir),
                "--output_dir",
                str(output_dir),
            ],
            "Table Generation",
        )
        results["tables"] = "✅ OK" if ok else "❌ FAILED"

    total_elapsed = round(time.time() - master_t0, 1)
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUITE COMPLETE")
    print(f"  Total time: {total_elapsed}s  ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*70}")
    for exp, status in results.items():
        print(f"  {exp:<25} {status}")
    print(f"{'='*70}")
    print(f"\n  Output files in: {output_dir.resolve()}")
    print(f"  Rebuttal tables: {output_dir / 'rebuttal_tables.txt'}")
    print(f"  Rebuttal tables: {output_dir / 'rebuttal_tables.md'}")

    if any("FAILED" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Run cancelled by user (Ctrl+C).")
        print("  Any fully completed contracts have been saved incrementally to disk.")
        print(
            "  Re-run with --ablation_conditions <remaining> to continue from where you left off."
        )
        sys.exit(0)
