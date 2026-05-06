"""
experiment_1_ablation.py
------------------------
Ablation Study — the highest-priority new experiment for the rebuttal.

Tests four conditions on the SAME randomly-sampled subset of contracts
so results are directly comparable:

  Condition A  Zero-shot baseline
               A single GPT-4o-mini call with no schema extraction,
               no FSM parsing, no audit, no reinforcement.
               This is the "strong baseline" both 4-reviewers demanded.

  Condition B  Full pipeline, reinforcement DISABLED
               enable_reinforcement=False

  Condition C  Full pipeline, max_iterations=1
               The configuration that was ACTUALLY run in the paper
               (DEFAULT_MAX_REFINEMENT_ITERATIONS = 1 in agents.py).

  Condition D  Full pipeline, max_iterations=2
               What the paper TEXT claims was used.

  Condition E  Schema-structured prompt, single LLM (no agent orchestration)
               Two sequential GPT calls (schema extraction then code generation)
               without any CrewAI agents, audit loop, or reinforcement.
               Isolates the value of multi-agent architecture vs. structured
               prompting alone — directly addresses reviewer inVW.

Addresses reviewer concerns:
  - J5uF:  "no strong baselines or ablations"
  - inVW:  "standard actor-critic implementation" / multi-agent value unclear
  - ybKn:  "how does each component contribute?"

Multi-seed support (--seeds 42 123 999) provides reproducibility evidence:
  Cross-seed variance in condition means confirms sampling stability.

Usage:
  python experiment_1_ablation.py \
      --dataset /path/to/requirement_code.jsonl \
      --n_samples 300 \
      --output_dir ./results/ablation \
      --conditions A B C D E \
      --seeds 42 123 999
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# parents[1] is the workspace root so 'from applications.*' imports resolve.
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
    classify_error_modes,
    compilation_success,
    compute_pairwise_significance,
    compute_statistics,
    evaluate_ground_truth_quality,
    extract_audit_info,
    extract_scores,
    get_raw_scores,
    load_dataset,
    print_stats_table,
    save_results,
)

from applications.solidity_compiler import SolidityCompilationChecker
from applications.translator import IBMAgenticContractTranslator

# Condition A: Zero-shot baseline (direct OpenAI call, no agents)
ZERO_SHOT_PROMPT = """You are a Solidity smart contract developer.
Generate a complete, production-ready Solidity smart contract (pragma ^0.8.0)
that implements the following natural-language specification exactly.

SPECIFICATION:
{requirement}

Return ONLY the complete Solidity code. No explanations, no markdown fences."""


def run_zero_shot_baseline(requirement: str) -> str:
    """
    Call GPT-4o-mini directly with a single prompt — no agents, no schema,
    no audit, no reinforcement.  Returns raw Solidity code.
    """
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": ZERO_SHOT_PROMPT.format(requirement=requirement),
            }
        ],
    )
    raw = response.choices[0].message.content or ""
    # Strip markdown fences if present
    if "```solidity" in raw:
        raw = raw.split("```solidity")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return raw.strip()


def process_zero_shot(record: Dict, checker: SolidityCompilationChecker) -> Dict:
    """Run one contract through the zero-shot baseline and return a result dict."""
    import openai

    result = {
        "index": record["index"],
        "condition": "A_zero_shot",
        "requirement": record["requirement"][:200],
        "solidity": None,
        "compilation": None,
        "ground_truth_compilation": None,
        "quality_evaluation": None,
        "ground_truth_quality_evaluation": None,
        "audit": None,
        "processing_time": None,
        "error": None,
    }

    t0 = time.time()
    try:
        solidity = run_zero_shot_baseline(record["requirement"])
        result["solidity"] = solidity

        # Compilation check on generated code
        result["compilation"] = checker.check_compilation(solidity)

        # Quality evaluation: run the LLM evaluator in a standalone crew so
        # Condition A has comparable quality scores alongside B/C/D.
        # We build a minimal schema from the requirement text since Phase 2
        # (contract parsing) is not run in the zero-shot baseline.
        try:
            from crewai import LLM as CrewLLM
            from crewai import Agent, Crew, Task

            from applications.task_builders import (
                create_quality_evaluation_task_description,
            )
            from applications.translator import IBMAgenticContractTranslator

            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            crew_llm = CrewLLM(
                model=model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
            )
            eval_agent = Agent(
                role="Smart Contract Quality Analyst",
                goal="Comprehensively evaluate generated smart contracts against natural language specifications",
                backstory=(
                    "You are an expert smart contract quality analyst evaluating how well "
                    "generated Solidity code implements natural language contract specifications. "
                    "You assess functional completeness, variable fidelity, state machine correctness, "
                    "business logic implementation, and code quality with objective scoring."
                ),
                llm=crew_llm,
                verbose=False,
                allow_delegation=False,
            )
            minimal_schema = {
                "contract_type": "unknown",
                "conditions": {},
                "raw_requirement": record["requirement"][:500],
            }
            task_desc = create_quality_evaluation_task_description(
                solidity_code=solidity,
                schema=minimal_schema,
                contract_name="ZeroShot",
            )
            eval_task = Task(
                description=task_desc,
                agent=eval_agent,
                expected_output="JSON quality evaluation",
            )
            eval_crew = Crew(agents=[eval_agent], tasks=[eval_task], verbose=False)
            raw = str(eval_crew.kickoff())

            import json as _json
            import re as _re

            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if m:
                result["quality_evaluation"] = _json.loads(m.group())
        except Exception as eval_exc:
            print(f"  [zero-shot eval] quality evaluation failed: {eval_exc}")

    except openai.OpenAIError as exc:
        result["error"] = f"OpenAI error: {exc}"
    except Exception as exc:
        result["error"] = str(exc)
        traceback.print_exc()

    result["processing_time"] = round(time.time() - t0, 2)

    # Ground-truth compilation
    if record.get("code"):
        result["ground_truth_compilation"] = checker.check_compilation(record["code"])
    # GT quality eval is skipped for Condition A (no translator instance).
    # Debiased-metric experiment (experiment_5) handles the GT quality comparison.
    return result


# Condition E: schema-structured prompt, single LLM (no agent orchestration)
SCHEMA_EXTRACTION_PROMPT = """\
You are a smart contract analyst.
Extract a structured schema from this natural-language specification.

SPECIFICATION:
{requirement}

Return a JSON object with these keys (include as many as are relevant):
{{
  "contract_type": "<type, e.g. escrow, token, auction>",
  "parties": ["<list of roles/parties>"],
  "state_variables": ["<list of state variables needed>"],
  "functions": ["<list of function signatures needed>"],
  "events": ["<list of events>"],
  "conditions": ["<key invariants and conditions>"]
}}
Return ONLY valid JSON, no explanation."""

SINGLE_LLM_GENERATE_PROMPT = """\
You are a senior Solidity developer.
Generate a complete, production-ready Solidity smart contract (pragma ^0.8.0).

NATURAL LANGUAGE SPECIFICATION:
{requirement}

EXTRACTED SCHEMA:
{schema}

Requirements:
- Implement ALL functions listed in the schema
- Use proper access control (owner or role-based as appropriate)
- Add require() statements for all validation conditions
- Emit events for all state changes
- Every function must be fully implemented — no placeholders

Return ONLY the complete Solidity code, no markdown fences, no explanation."""


def run_single_llm_e(
    requirement: str,
    model: str = "gpt-4o-mini",
) -> Tuple[str, Dict]:
    """
    Condition E: two sequential GPT calls — schema extraction then code generation.
    No agents, no audit, no reinforcement.  Returns (solidity_code, schema_dict).
    """
    import json as _json
    import re as _re

    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Call 1: schema extraction (lower temperature for structured JSON output)
    schema_raw = (
        client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": SCHEMA_EXTRACTION_PROMPT.format(requirement=requirement),
                }
            ],
        )
        .choices[0]
        .message.content
        or "{}"
    )

    schema: Dict = {}
    try:
        m = _re.search(r"\{.*\}", schema_raw, _re.DOTALL)
        if m:
            schema = _json.loads(m.group())
    except Exception:
        pass  # fall through with empty schema

    # Call 2: code generation using extracted schema
    code_raw = (
        client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": SINGLE_LLM_GENERATE_PROMPT.format(
                        requirement=requirement,
                        schema=_json.dumps(schema, indent=2),
                    ),
                }
            ],
        )
        .choices[0]
        .message.content
        or ""
    )

    # Strip markdown fences if present
    if "```solidity" in code_raw:
        code_raw = code_raw.split("```solidity")[1].split("```")[0]
    elif "```" in code_raw:
        code_raw = code_raw.split("```")[1].split("```")[0]

    return code_raw.strip(), schema


def process_single_llm_e(
    record: Dict,
    checker: SolidityCompilationChecker,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Run one contract through Condition E and return a result dict."""
    import openai

    result = {
        "index": record["index"],
        "condition": "E_single_llm",
        "requirement": record["requirement"][:200],
        "solidity": None,
        "extracted_schema": None,
        "compilation": None,
        "ground_truth_compilation": None,
        "quality_evaluation": None,
        "ground_truth_quality_evaluation": None,
        "processing_time": None,
        "error": None,
    }

    t0 = time.time()
    try:
        solidity, schema = run_single_llm_e(record["requirement"], model=model)
        result["solidity"] = solidity
        result["extracted_schema"] = schema

        result["compilation"] = checker.check_compilation(solidity)

        # Quality evaluation: same standalone crew as condition A
        try:
            from crewai import LLM as CrewLLM
            from crewai import Agent, Crew, Task

            from applications.task_builders import (
                create_quality_evaluation_task_description,
            )

            crew_llm = CrewLLM(
                model=model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
            )
            eval_agent = Agent(
                role="Smart Contract Quality Analyst",
                goal="Comprehensively evaluate generated smart contracts against natural language specifications",
                backstory=(
                    "You are an expert smart contract quality analyst evaluating how well "
                    "generated Solidity code implements natural language contract specifications."
                ),
                llm=crew_llm,
                verbose=False,
                allow_delegation=False,
            )
            eval_schema = (
                schema
                if schema
                else {
                    "contract_type": "unknown",
                    "conditions": {},
                    "raw_requirement": record["requirement"][:500],
                }
            )
            task_desc = create_quality_evaluation_task_description(
                solidity_code=solidity,
                schema=eval_schema,
                contract_name="SingleLLM_E",
            )
            eval_task = Task(
                description=task_desc,
                agent=eval_agent,
                expected_output="JSON quality evaluation",
            )
            eval_crew = Crew(agents=[eval_agent], tasks=[eval_task], verbose=False)
            raw = str(eval_crew.kickoff())

            import json as _json2
            import re as _re2

            m = _re2.search(r"\{.*\}", raw, _re2.DOTALL)
            if m:
                result["quality_evaluation"] = _json2.loads(m.group())
        except Exception as eval_exc:
            print(f"  [cond-E eval] quality evaluation failed: {eval_exc}")

    except openai.OpenAIError as exc:
        result["error"] = f"OpenAI error: {exc}"
    except Exception as exc:
        result["error"] = str(exc)
        traceback.print_exc()

    result["processing_time"] = round(time.time() - t0, 2)

    if record.get("code"):
        result["ground_truth_compilation"] = checker.check_compilation(record["code"])

    return result


# Conditions B / C / D: full pipeline variants
def process_pipeline_contract(
    translator: IBMAgenticContractTranslator,
    record: Dict,
    condition_label: str,
    checker: SolidityCompilationChecker,
) -> Dict:
    """
    Run one contract through the translator streaming pipeline
    and collect every useful signal into a flat result dict.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(record["requirement"])
        tmp_path = tmp.name

    result = {
        "index": record["index"],
        "condition": condition_label,
        "requirement": record["requirement"][:200],
        "solidity": None,
        "compilation": None,
        "ground_truth_compilation": None,
        "quality_evaluation": None,
        "ground_truth_quality_evaluation": None,
        "audit": None,
        "refinement_iterations": 0,
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

            if phase == 3:
                result["solidity"] = data.get("solidity")

            elif phase == 4:
                # May fire multiple times (status: refining / needs_approval)
                result["audit"] = data
                result["refinement_iterations"] = data.get(
                    "refinement_iterations", result["refinement_iterations"]
                )

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

    # Ground-truth compilation check
    if record.get("code"):
        result["ground_truth_compilation"] = checker.check_compilation(record["code"])

    # Ground-truth quality evaluation using the translator's quality evaluator agent
    if record.get("code") and hasattr(translator, "quality_evaluator_agent"):
        try:
            result["ground_truth_quality_evaluation"] = evaluate_ground_truth_quality(
                ground_truth_code=record["code"],
                schema=None,
                requirement=record["requirement"],
                quality_evaluator_agent=translator.quality_evaluator_agent,
            )
        except Exception as gt_exc:
            print(f"  [GT quality eval] {gt_exc}")

    return result


# Runner helpers
CONDITION_CONFIGS = {
    "A": {
        "label": "A_zero_shot",
        "description": "Zero-shot baseline (single GPT-4o-mini call, no agents)",
        "enable_reinforcement": None,  # N/A, handled separately
        "max_iterations": None,
    },
    "B": {
        "label": "B_no_reinforce",
        "description": "Full pipeline, reinforcement DISABLED",
        "enable_reinforcement": False,
        "max_iterations": 1,
    },
    "C": {
        "label": "C_iter1",
        "description": "Full pipeline, max_iterations=1 (actual paper config)",
        "enable_reinforcement": True,
        "max_iterations": 1,
    },
    "D": {
        "label": "D_iter2",
        "description": "Full pipeline, max_iterations=2 (paper text claims this)",
        "enable_reinforcement": True,
        "max_iterations": 2,
    },
    "E": {
        "label": "E_single_llm",
        "description": "Schema-structured prompt, single LLM (no agent orchestration)",
        "enable_reinforcement": None,  # N/A, no pipeline
        "max_iterations": None,
    },
}


def run_condition(
    condition_key: str,
    records: List[Dict],
    output_dir: Path,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    Execute all records for one experimental condition and save results.
    """
    cfg = CONDITION_CONFIGS[condition_key]
    label = cfg["label"]
    desc = cfg["description"]

    print(f"\n{'#'*70}")
    print(f"# Condition {condition_key}: {desc}")
    print(f"# N = {len(records)} contracts")
    print(f"{'#'*70}\n")

    checker = SolidityCompilationChecker()
    out_path = output_dir / f"condition_{label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w").close()  # truncate for a fresh run
    results = []

    if condition_key == "A":
        # Zero-shot: no translator needed
        for i, record in enumerate(records):
            print(
                f"  [{i+1}/{len(records)}] idx={record['index']} ...",
                end=" ",
                flush=True,
            )
            r = process_zero_shot(record, checker)
            results.append(r)
            append_result(r, out_path)
            status = "✓" if not r["error"] else "✗"
            comp = compilation_success(r["compilation"])
            print(
                f"{status}  compile={'Y' if comp else ('N' if comp is False else '?')}  "
                f"time={r['processing_time']}s"
            )
    elif condition_key == "E":
        # Single-LLM structured prompt: no translator needed
        for i, record in enumerate(records):
            print(
                f"  [{i+1}/{len(records)}] idx={record['index']} ...",
                end=" ",
                flush=True,
            )
            r = process_single_llm_e(record, checker, model=model)
            results.append(r)
            append_result(r, out_path)
            status = "✓" if not r["error"] else "✗"
            comp = compilation_success(r["compilation"])
            scores = extract_scores(r.get("quality_evaluation"))
            print(
                f"{status}  score={scores['composite']:.1f}  "
                f"compile={'Y' if comp else ('N' if comp is False else '?')}  "
                f"time={r['processing_time']}s"
            )
    else:
        # Full pipeline conditions B / C / D
        translator = IBMAgenticContractTranslator(
            model=model,
            enable_reinforcement=cfg["enable_reinforcement"],
            max_refinement_iterations=cfg["max_iterations"],
        )

        for i, record in enumerate(records):
            print(
                f"  [{i+1}/{len(records)}] idx={record['index']} ...",
                end=" ",
                flush=True,
            )
            r = process_pipeline_contract(translator, record, label, checker)
            results.append(r)
            append_result(r, out_path)

            scores = extract_scores(r.get("quality_evaluation"))
            status = "✓" if not r["error"] else "✗"
            comp = compilation_success(r.get("compilation"))
            print(
                f"{status}  score={scores['composite']:.1f}  "
                f"compile={'Y' if comp else ('N' if comp is False else '?')}  "
                f"refine_iters={r.get('refinement_iterations', 0)}  "
                f"time={r['processing_time']}s"
            )

    # Save raw results (re-write complete set for consistency)
    save_results(results, str(out_path))

    return results


# Summary table
def print_ablation_summary(
    all_condition_results: Dict[str, List[Dict]],
    seed_stability: Optional[Dict] = None,
) -> None:
    """
    Print a publication-ready comparison table with pairwise significance vs. C.

    Args:
        all_condition_results: {condition_key: [results]}
        seed_stability: optional cross-seed stats dict from multi-seed run
    """

    def _fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    ref_results = all_condition_results.get("C")

    print(f"\n{'='*100}")
    print("  ABLATION STUDY — REBUTTAL TABLE")
    print(f"{'='*100}")
    print(
        f"  {'Condition':<34} {'Score':>8} {'±Std':>6} {'Compile%':>9} "
        f"{'M1':>6} {'M3':>6} {'M4':>6} {'N':>5}  {'p vs C':>15}"
    )
    print("-" * 100)

    for key, results in all_condition_results.items():
        cfg = CONDITION_CONFIGS[key]
        stats = compute_statistics(results)

        comp_rate = stats.get("compilation_rate")
        comp_str = f"{comp_rate*100:.1f}%" if comp_rate is not None else "N/A"
        score_mean = stats["composite"]["mean"]
        score_std = stats["composite"]["std"]
        m1_mean = stats["m1_functional"]["mean"]
        m3_mean = stats["m3_state_machine"]["mean"]
        m4_mean = stats["m4_business_logic"]["mean"]
        label = cfg["label"]
        n = stats["n_total"] - stats["n_errors"]

        # Pairwise significance vs reference condition C
        if key == "C":
            sig_str = "[reference]"
        elif ref_results:
            sig = compute_pairwise_significance(results, ref_results)
            comp_info = sig.get("composite", {})
            p_t = comp_info.get("p_value_t")
            stars = comp_info.get("stars", "")
            d = comp_info.get("cohens_d", 0.0)
            diff = comp_info.get("mean_diff", 0.0)
            if p_t is not None:
                sig_str = f"Δ={diff:+.3f} d={d:.2f} p={p_t:.4f}{stars}"
            else:
                sig_str = f"Δ={diff:+.3f} (no scipy)"
        else:
            sig_str = "N/A"

        print(
            f"  {label:<34} {_fmt(score_mean):>8} {_fmt(score_std):>6} {comp_str:>9}  "
            f"{_fmt(m1_mean):>6}  {_fmt(m3_mean):>6}  {_fmt(m4_mean):>6}  {n:>5}  {sig_str}"
        )

    print(f"{'='*100}")
    print(
        "\n  Interpretation for rebuttal:"
        "\n  - Cond A: zero-shot (no agents, no schema, no audit) — strongest baseline"
        "\n  - Cond B: full pipeline without reinforcement loop"
        "\n  - Cond C: paper config — full multi-agent pipeline, max_iter=1 [REFERENCE]"
        "\n  - Cond D: full pipeline, max_iter=2 (paper text claim)"
        "\n  - Cond E: 2-call structured LLM (no agent orchestration) [vs inVW reviewer]"
        "\n  - Δ(C vs A): value of complete multi-agent pipeline over zero-shot"
        "\n  - Δ(C vs E): value of agent orchestration over structured prompting alone"
        "\n  - Δ(C vs B): isolated contribution of the reinforcement loop"
        "\n  - p-values: paired t-test (same contracts, matched by index); *** p<0.001"
    )

    if seed_stability:
        print(f"\n  Cross-seed stability (mean ± std across seeds):")
        for cond_key, ss in seed_stability.items():
            print(
                f"    {ss.get('label', cond_key):<30} "
                f"{ss.get('cross_seed_mean_composite', 'N/A'):.3f} "
                f"± {ss.get('cross_seed_std_composite', 0):.3f}"
            )


# Entry point
def main():
    parser = argparse.ArgumentParser(description="Ablation study for rebuttal")
    parser.add_argument(
        "--dataset",
        default="requirement_code.jsonl",
        help="Path to FSM-SCG requirement_code.jsonl",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=300,
        help="Number of contracts to sample per seed (default: 300)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for sampling (default: [42]). "
        "Use '--seeds 42 123 999' for 3-seed reproducibility check.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/ablation",
        help="Directory to write result JSONL files",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        choices=["A", "B", "C", "D", "E"],
        help="Which conditions to run (default: all five)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multi_seed = len(args.seeds) > 1
    if multi_seed:
        print(f"[ablation] Multi-seed mode: seeds={args.seeds}")

    # all_seed_results[seed][condition_key] = [results]
    all_seed_results: Dict[int, Dict[str, List[Dict]]] = {}
    seed_stability: Dict[str, Any] = {}

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        seed_dir = output_dir / f"seed_{seed}" if multi_seed else output_dir
        seed_dir.mkdir(parents=True, exist_ok=True)

        records = load_dataset(args.dataset, n_samples=args.n_samples, seed=seed)
        seed_results: Dict[str, List[Dict]] = {}

        for cond_key in args.conditions:
            cond_results = run_condition(cond_key, records, seed_dir, model=args.model)
            seed_results[cond_key] = cond_results
            stats = compute_statistics(cond_results)
            lbl = CONDITION_CONFIGS[cond_key]["label"]
            print_stats_table(f"{lbl} (seed={seed})" if multi_seed else lbl, stats)

        if len(seed_results) > 1:
            print_ablation_summary(seed_results)

        all_seed_results[seed] = seed_results

        # Per-seed summary JSON
        seed_summary = {k: compute_statistics(v) for k, v in seed_results.items()}
        with open(seed_dir / "ablation_summary.json", "w") as f:
            json.dump(seed_summary, f, indent=2)
        print(f"[ablation] Seed {seed} summary → {seed_dir / 'ablation_summary.json'}")

    # Single-seed backward-compat: write summary at output_dir root
    if not multi_seed:
        final = all_seed_results[args.seeds[0]]
        summary = {k: compute_statistics(v) for k, v in final.items()}
        with open(output_dir / "ablation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ablation] Summary written → {output_dir / 'ablation_summary.json'}")

    # Multi-seed: cross-seed aggregation
    if multi_seed:
        print(f"\n{'='*70}")
        print("  CROSS-SEED AGGREGATED RESULTS")
        print(f"{'='*70}")
        import statistics as _stat

        for cond_key in args.conditions:
            cfg = CONDITION_CONFIGS[cond_key]
            per_seed_means: List[float] = []
            per_seed_compiles: List[float] = []
            for seed, seed_results in all_seed_results.items():
                if cond_key not in seed_results:
                    continue
                s = compute_statistics(seed_results[cond_key])
                m = s["composite"]["mean"]
                cr = s.get("compilation_rate")
                if m is not None:
                    per_seed_means.append(m)
                if cr is not None:
                    per_seed_compiles.append(cr)

            if per_seed_means:
                cross_mean = _stat.mean(per_seed_means)
                cross_std = (
                    _stat.stdev(per_seed_means) if len(per_seed_means) > 1 else 0.0
                )
                comp_mean = _stat.mean(per_seed_compiles) if per_seed_compiles else None
                comp_str = f"{comp_mean*100:.1f}%" if comp_mean is not None else "N/A"
                print(
                    f"  {cfg['label']:<30}: {cross_mean:.3f} ± {cross_std:.3f}  "
                    f"(compile: {comp_str})"
                )
                seed_stability[cond_key] = {
                    "label": cfg["label"],
                    "cross_seed_mean_composite": round(cross_mean, 4),
                    "cross_seed_std_composite": round(cross_std, 4),
                    "per_seed_means": per_seed_means,
                    "seeds": args.seeds,
                }

        with open(output_dir / "seed_stability.json", "w") as f:
            json.dump(seed_stability, f, indent=2)
        print(f"[ablation] Seed stability → {output_dir / 'seed_stability.json'}")

        # Print final aggregated table
        final = all_seed_results[args.seeds[-1]]
        if len(final) > 1:
            print_ablation_summary(final, seed_stability=seed_stability)

    # Pairwise significance tests (computed from last/only seed results)
    final = all_seed_results[args.seeds[-1]]
    if "C" in final and len(final) > 1:
        pairwise: Dict[str, Any] = {}
        print(f"\n  Pairwise significance vs Condition C:")
        for cond_key, cond_results in final.items():
            if cond_key == "C":
                continue
            sig = compute_pairwise_significance(cond_results, final["C"])
            pairwise[cond_key] = sig
            comp = sig.get("composite", {})
            p_t = comp.get("p_value_t")
            diff = comp.get("mean_diff", 0.0)
            d = comp.get("cohens_d", 0.0)
            stars = comp.get("stars", "")
            print(
                f"  {cond_key} vs C: Δ={diff:+.3f}  d={d:.2f}  "
                f"p={'N/A' if p_t is None else f'{p_t:.4f}'}{stars}  "
                f"n_pairs={sig.get('n_pairs', 0)}"
            )

        with open(output_dir / "pairwise_significance.json", "w") as f:
            json.dump(pairwise, f, indent=2)
        print(
            f"[ablation] Pairwise significance → {output_dir / 'pairwise_significance.json'}"
        )


if __name__ == "__main__":
    main()
