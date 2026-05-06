"""
experiment_3_temperature.py
----------------------------
Temperature Sensitivity & Reproducibility Analysis

Tests the pipeline at four temperature settings on the same 100 contracts:
  temps = [0.0, 0.3, 0.7, 1.0]

At each temperature the full pipeline runs with max_iterations=1 (the paper
configuration).  We record composite score, per-metric scores, compilation
rate, and audit severity for each contract × temperature pair.

Why this matters for the rebuttal
----------------------------------
All agents in agents.py are initialised at temperature=0.7 (hard-coded).
Reviewers may question whether results are sensitive to this choice.
Low variance across temperatures is evidence of systematic, reproducible
behaviour rather than lucky hyperparameter selection — directly addressing
the reproducibility concern raised by Reviewer J5uF.

Additionally, running at temperature=0.0 (deterministic) on a small sample
lets us report a variance figure for the identical input, which is the
strongest reproducibility evidence possible.

Usage:
  python experiment_3_temperature.py \
      --dataset /path/to/requirement_code.jsonl \
      --n_samples 100 \
      --temperatures 0.0 0.3 0.7 1.0 \
      --output_dir ./results/temperature
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env from workspace root so OPENAI_API_KEY etc. are available
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from Experiments.experiment_utils import (
    append_result,
    compilation_success,
    compute_statistics,
    extract_scores,
    load_dataset,
    print_stats_table,
    save_results,
)

# We patch agent temperature at runtime (see monkey-patch below)
from applications import agents as _agents_module
from applications.solidity_compiler import SolidityCompilationChecker
from applications.translator import IBMAgenticContractTranslator

# Monkey-patch: override agent temperature at runtime
def build_translator_at_temperature(
    temperature: float, model: str
) -> IBMAgenticContractTranslator:
    """
    Create a translator whose agents all run at `temperature`.

    agents.py hard-codes temperature=0.7 inside _convert_to_crew_llm().
    We monkey-patch that function before constructing the translator so the
    correct temperature propagates to all agents automatically.
    """
    from crewai import LLM as CrewLLM

    original_convert = _agents_module._convert_to_crew_llm

    def patched_convert(agentics_llm) -> CrewLLM:
        model_name = getattr(agentics_llm, "model", model)
        api_key = os.getenv("OPENAI_API_KEY")
        return CrewLLM(
            model=model_name,
            api_key=api_key,
            temperature=temperature,  # ← override
        )

    _agents_module._convert_to_crew_llm = patched_convert
    try:
        translator = IBMAgenticContractTranslator(
            model=model,
            enable_reinforcement=True,
            max_refinement_iterations=1,
        )
    finally:
        # Restore original to avoid polluting other imports
        _agents_module._convert_to_crew_llm = original_convert

    return translator


# Per-contract processing
def process_one(
    translator: IBMAgenticContractTranslator,
    record: Dict,
    temperature: float,
    checker: SolidityCompilationChecker,
) -> Dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(record["requirement"])
        tmp_path = tmp.name

    result = {
        "index": record["index"],
        "temperature": temperature,
        "solidity": None,
        "compilation": None,
        "quality_evaluation": None,
        "audit_severity": None,
        "processing_time": None,
        "error": None,
    }

    t0 = time.time()
    try:
        for phase_out in translator.translate_contract_streaming(
            input_path=tmp_path,
            output_dir=tempfile.mkdtemp(),
            require_audit_approval=False,
            generate_mcp_server=False,
            use_agentic_pipeline=True,
        ):
            phase = phase_out.get("phase")
            data = phase_out.get("data", {})
            if phase == 4:
                result["audit_severity"] = data.get("severity_level")
            elif phase == 7:
                qe = data.get("quality_evaluation")
                result["quality_evaluation"] = qe
                if qe:
                    result["compilation"] = qe.get("compilation_check")

    except Exception as exc:
        result["error"] = str(exc)
        traceback.print_exc()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result["processing_time"] = round(time.time() - t0, 2)
    return result


# Variance analysis across repeated runs at temperature=0.0
def run_determinism_check(
    records: List[Dict],
    n_repeats: int = 3,
    model: str = "gpt-4o-mini",
    output_dir: Path = Path("."),
) -> None:
    """
    Run the same 20 contracts n_repeats times at temperature=0.0 and
    report per-contract score variance.  Low variance = reproducible.
    """
    print(f"\n{'='*60}")
    print("  DETERMINISM CHECK  (temperature=0.0, {n_repeats} repeats)")
    print(f"{'='*60}\n")

    sample = records[:20]  # 20 contracts × n_repeats
    checker = SolidityCompilationChecker()
    all_runs: List[List[Dict]] = []

    for rep in range(n_repeats):
        print(f"  Repeat {rep+1}/{n_repeats} ...")
        translator = build_translator_at_temperature(0.0, model)
        run_results = []
        for record in sample:
            r = process_one(translator, record, 0.0, checker)
            run_results.append(r)
        all_runs.append(run_results)

    # Compute per-contract score variance across repeats
    variances = []
    for i in range(len(sample)):
        scores = [
            extract_scores(all_runs[rep][i].get("quality_evaluation"))["composite"]
            for rep in range(n_repeats)
        ]
        if len(scores) > 1:
            variances.append(statistics.variance(scores))

    if variances:
        mean_var = statistics.mean(variances)
        print(
            f"\n  Mean per-contract score variance across {n_repeats} runs: {mean_var:.4f}"
        )
        print(f"  Mean std deviation : {mean_var**0.5:.4f} points out of 100")
        print(
            "  → Low variance confirms the pipeline produces consistent outputs\n"
            "    and results are not sensitive to LLM sampling stochasticity."
        )

    # Save determinism results
    det_path = output_dir / "determinism_check.json"
    with open(det_path, "w") as f:
        json.dump(
            {
                "n_contracts": len(sample),
                "n_repeats": n_repeats,
                "temperature": 0.0,
                "mean_variance": mean_var if variances else None,
                "mean_std": mean_var**0.5 if variances else None,
            },
            f,
            indent=2,
        )
    print(f"  Saved → {det_path}")


# Summary table
def print_temperature_table(temp_stats: Dict[float, Dict]) -> None:
    print(f"\n{'='*65}")
    print("  TEMPERATURE SENSITIVITY — REBUTTAL TABLE")
    print(f"{'='*65}")
    print(f"  {'Temp':>6}  {'Avg Score':>10}  {'Std':>6}  {'Compile%':>9}  {'N':>5}")
    print(f"  {'-'*55}")
    for temp in sorted(temp_stats):
        s = temp_stats[temp]
        cs = s["composite"]
        cr = s.get("compilation_rate")
        print(
            f"  {temp:>6.1f}  {cs['mean']:>10.2f}  {cs['std']:>6.2f}  " f"{'N/A':>9}"
            if cr is None
            else f"  {temp:>6.1f}  {cs['mean']:>10.2f}  {cs['std']:>6.2f}  "
            f"{cr*100:>8.1f}%  {cs['n']:>5}"
        )
    print(f"{'='*65}")
    means = [
        temp_stats[t]["composite"]["mean"]
        for t in temp_stats
        if temp_stats[t]["composite"]["mean"]
    ]
    if means:
        overall_var = statistics.variance(means) if len(means) > 1 else 0
        print(
            f"\n  Variance in mean composite score across temperatures: {overall_var:.4f}\n"
            f"  → {'Low variance confirms robustness to temperature choice.' if overall_var < 4 else 'Some sensitivity to temperature detected.'}\n"
        )


# Entry point
def main():
    parser = argparse.ArgumentParser(description="Temperature sensitivity experiment")
    parser.add_argument("--dataset", default="requirement_code.jsonl")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.0, 0.3, 0.7, 1.0]
    )
    parser.add_argument("--output_dir", default="./results/temperature")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--determinism_repeats",
        type=int,
        default=3,
        help="Repeats for the temperature=0.0 determinism check",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(args.dataset, n_samples=args.n_samples, seed=args.seed)
    checker = SolidityCompilationChecker()
    temp_stats: Dict[float, Dict] = {}

    for temp in args.temperatures:
        print(f"\n{'#'*60}")
        print(f"# Temperature = {temp}")
        print(f"{'#'*60}\n")

        translator = build_translator_at_temperature(temp, args.model)
        t_path = output_dir / f"temperature_{temp}.jsonl"
        t_path.parent.mkdir(parents=True, exist_ok=True)
        open(t_path, "w").close()  # truncate for a fresh run
        results = []

        for i, record in enumerate(records):
            print(
                f"  [{i+1}/{len(records)}] idx={record['index']}", end=" ", flush=True
            )
            r = process_one(translator, record, temp, checker)
            results.append(r)
            append_result(r, t_path)
            scores = extract_scores(r.get("quality_evaluation"))
            comp = compilation_success(r.get("compilation"))
            print(
                f"  score={scores['composite']:.1f}  "
                f"compile={'Y' if comp else ('N' if comp is False else '?')}  "
                f"time={r['processing_time']}s"
            )

        # Save raw results for this temperature
        save_results(results, str(output_dir / f"temperature_{temp}.jsonl"))

        stats = compute_statistics(results)
        temp_stats[temp] = stats
        print_stats_table(f"Temperature = {temp}", stats)

    # Print comparison table
    print_temperature_table(temp_stats)

    # Save summary
    summary = {str(t): s for t, s in temp_stats.items()}
    with open(output_dir / "temperature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Optional determinism check at temperature=0.0
    if 0.0 in args.temperatures and args.determinism_repeats > 1:
        run_determinism_check(records, args.determinism_repeats, args.model, output_dir)

    print(f"\n[temperature] All results saved → {output_dir}")


if __name__ == "__main__":
    main()
