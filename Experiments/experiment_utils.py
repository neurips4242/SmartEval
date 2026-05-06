"""
experiment_utils.py
Shared utilities for all experiments: dataset loading, pipeline execution,
score extraction, result persistence, and statistics.
"""

import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy.stats import ttest_rel
    from scipy.stats import wilcoxon as _wilcoxon  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

# Experiments/ is directly under the workspace root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env from workspace root
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from applications.solidity_compiler import SolidityCompilationChecker
from applications.task_builders import create_quality_evaluation_task_description

# Dataset loading
def load_dataset(
    jsonl_path: str,
    n_samples: int = 300,
    seed: int = 42,
    requirement_key: str = "requirement",
    code_key: str = "code",
) -> List[Dict]:
    """
    Load and randomly sample contracts from the FSM-SCG JSONL dataset.

    Args:
        jsonl_path:      Path to requirement_code.jsonl
        n_samples:       How many contracts to sample (-1 = all)
        seed:            Random seed for reproducibility
        requirement_key: JSON key for the natural-language spec
        code_key:        JSON key for the ground-truth Solidity code

    Returns:
        List of dicts, each with keys 'requirement', 'code', and 'index'.
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                req = (
                    obj.get(requirement_key)
                    or obj.get("user_requirement")
                    or obj.get("nl")
                    or obj.get("description", "")
                )
                code = (
                    obj.get(code_key) or obj.get("solidity") or obj.get("contract", "")
                )
                if req and code:
                    records.append({"index": idx, "requirement": req, "code": code})
            except json.JSONDecodeError:
                continue

    print(f"[utils] Loaded {len(records)} records from {jsonl_path}")

    if n_samples > 0 and n_samples < len(records):
        random.seed(seed)
        records = random.sample(records, n_samples)
        print(f"[utils] Sampled {n_samples} contracts (seed={seed})")

    return records


# Pipeline execution helper
def process_single_contract(
    translator,
    requirement: str,
    ground_truth_code: str,
    contract_index: int,
    generate_mcp: bool = False,
) -> Dict:
    """
    Run one contract through the full streaming translator pipeline and
    collect all phase outputs into a single result dict.

    Args:
        translator:         An initialised IBMAgenticContractTranslator instance.
        requirement:        Natural-language contract specification.
        ground_truth_code:  Expert-written Solidity (for paired evaluation).
        contract_index:     Position in the dataset (for bookkeeping).
        generate_mcp:       Whether to run the MCP-server generation phase.

    Returns:
        Dict with keys: index, requirement, solidity, audit, quality_evaluation,
        compilation, ground_truth_compilation, processing_time, error (if any).
    """
    # Write requirement to a temp file so translate_contract_streaming can read it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(requirement)
        tmp_path = tmp.name

    result = {
        "index": contract_index,
        "requirement": requirement[:200],  # truncate for storage
        "ground_truth_code": ground_truth_code,
        "solidity": None,
        "audit": None,
        "quality_evaluation": None,
        "ground_truth_quality_evaluation": None,
        "compilation": None,
        "ground_truth_compilation": None,
        "processing_time": None,
        "error": None,
    }

    t0 = time.time()
    try:
        for phase_result in translator.translate_contract_streaming(
            input_path=tmp_path,
            output_dir=tempfile.mkdtemp(),  # throwaway output dir
            require_audit_approval=False,  # never prompt in batch mode
            generate_mcp_server=generate_mcp,
            use_agentic_pipeline=True,
        ):
            phase = phase_result.get("phase")
            data = phase_result.get("data", {})

            if phase == 2:  # Contract parsing / schema extraction
                result["schema"] = data.get("schema")
            elif phase == 3:  # Solidity generation
                result["solidity"] = data.get("solidity")
            elif phase == 4:  # Audit (may fire multiple times during refinement)
                result["audit"] = data
            elif phase == 7:  # Quality evaluation
                result["quality_evaluation"] = data.get("quality_evaluation")
                result["compilation"] = data.get("quality_evaluation", {}).get(
                    "compilation_check"
                )

    except Exception as exc:
        result["error"] = str(exc)
        traceback.print_exc()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result["processing_time"] = round(time.time() - t0, 2)

    # Compile ground-truth code to compare against generated
    if ground_truth_code:
        checker = SolidityCompilationChecker()
        result["ground_truth_compilation"] = checker.check_compilation(
            ground_truth_code
        )

    # Evaluate ground-truth quality with the same rubric used for generated code
    if ground_truth_code and hasattr(translator, "quality_evaluator_agent"):
        result["ground_truth_quality_evaluation"] = evaluate_ground_truth_quality(
            ground_truth_code=ground_truth_code,
            schema=result.get("schema"),  # best-effort; may be None
            requirement=requirement,
            quality_evaluator_agent=translator.quality_evaluator_agent,
        )

    return result


def evaluate_ground_truth_quality(
    ground_truth_code: str,
    schema,
    requirement: str,
    quality_evaluator_agent,
) -> Optional[Dict]:
    """
    Run the same LLM-based quality evaluator on the GROUND-TRUTH Solidity code
    so that generated and expert contracts are scored on an equal footing.

    This directly addresses the +8.29 gap validity concern: averaging quality
    scores over generated contracts and comparing to compilation-only GT metrics
    is an apples-to-oranges comparison.  This function gives us true GT quality
    baselines under the same 5-metric rubric.

    Args:
        ground_truth_code:      Expert-written Solidity code from the dataset.
        schema:                 Parsed UniversalContractSchema (may be None if
                                Phase 2 result was not saved).
        requirement:            Original natural-language specification; used to
                                build a minimal schema dict when schema is None.
        quality_evaluator_agent: CrewAI Agent instance from the translator.

    Returns:
        Quality evaluation dict (same structure as generated-code evaluation),
        or None on failure.
    """
    try:
        from crewai import Crew, Task

        # Build a minimal placeholder schema when it wasn't passed
        if schema is None:
            schema_for_eval = {
                "contract_type": "unknown",
                "conditions": {},
                "raw_requirement": requirement[:500],
            }
        else:
            schema_for_eval = schema

        task_desc = create_quality_evaluation_task_description(
            solidity_code=ground_truth_code,
            schema=schema_for_eval,
            contract_name="GroundTruth",
        )

        eval_task = Task(
            description=task_desc,
            agent=quality_evaluator_agent,
            expected_output="JSON quality evaluation report for ground-truth contract",
        )
        crew = Crew(agents=[quality_evaluator_agent], tasks=[eval_task], verbose=False)
        raw_output = crew.kickoff()
        raw_text = str(raw_output)

        # Extract JSON from the agent output
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None

    except Exception as exc:
        print(f"[utils] GT quality evaluation failed: {exc}")
        return None


# Score / audit extraction helpers
def extract_scores(quality_evaluation: Optional[Dict]) -> Dict[str, float]:
    """
    Pull m1-m5 and the composite score out of a quality_evaluation dict.
    Returns a flat dict with keys m1, m2, m3, m4, m5, composite.
    All values default to 0.0 if missing.
    """
    if not quality_evaluation:
        return dict(m1=0.0, m2=0.0, m3=0.0, m4=0.0, m5=0.0, composite=0.0)

    m1 = float(
        quality_evaluation.get("metric_1_functional_completeness", {}).get("score", 0)
    )
    m2 = float(quality_evaluation.get("metric_2_variable_fidelity", {}).get("score", 0))
    m3 = float(quality_evaluation.get("metric_3_state_machine", {}).get("score", 0))
    m4 = float(quality_evaluation.get("metric_4_business_logic", {}).get("score", 0))
    m5 = float(quality_evaluation.get("metric_5_code_quality", {}).get("score", 0))

    # Recompute composite deterministically
    composite = round(m1 * 0.25 + m2 * 0.15 + m3 * 0.15 + m4 * 0.35 + m5 * 0.10, 2)

    return dict(m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, composite=composite)


def extract_audit_info(audit_data: Optional[Dict]) -> Dict:
    """
    Extract severity, vuln count, and approval flag from an audit dict.
    Handles both raw audit reports and the phase-4 streaming payload.
    """
    if not audit_data:
        return dict(severity="unknown", approved=False, vuln_count=0, critical=0)

    severity = (audit_data.get("severity_level") or "unknown").lower()
    approved = bool(audit_data.get("approved", False))
    vuln_count = int(
        audit_data.get("vulnerability_count") or len(audit_data.get("issues", []))
    )
    issues = audit_data.get("issues", [])
    critical = sum(1 for i in issues if isinstance(i, str) and "critical" in i.lower())
    return dict(
        severity=severity,
        approved=approved,
        vuln_count=vuln_count,
        critical=critical,
    )


def compilation_success(compilation_result: Optional[Dict]) -> Optional[bool]:
    """Return True/False/None from a compilation result dict."""
    if not compilation_result:
        return None
    return compilation_result.get("compiles")


# Result persistence
def save_results(results: List[Dict], output_path: str) -> None:
    """Write results to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"[utils] Saved {len(results)} results → {output_path}")


def append_result(result: Dict, output_path: str) -> None:
    """Write a single result immediately to disk (crash-safe incremental save)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, default=str) + "\n")


def load_results(path: str) -> List[Dict]:
    """Load previously saved experiment results."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Pairwise Statistical Significance
def get_raw_scores(results: List[Dict]) -> Dict[str, Any]:
    """
    Extract per-contract score lists for use in paired statistical tests.
    Returns a dict with keys: indices, composite, m1, m2, m3, m4, m5.
    Only includes records without errors and with a valid quality_evaluation.
    """
    indices, composites, m1s, m2s, m3s, m4s, m5s = [], [], [], [], [], [], []
    for r in results:
        if r.get("error"):
            continue
        s = extract_scores(r.get("quality_evaluation"))
        if s["composite"] == 0.0 and s["m1"] == 0.0:
            continue  # skip null evaluations
        indices.append(r.get("index"))
        composites.append(s["composite"])
        m1s.append(s["m1"])
        m2s.append(s["m2"])
        m3s.append(s["m3"])
        m4s.append(s["m4"])
        m5s.append(s["m5"])
    return dict(
        indices=indices, composite=composites, m1=m1s, m2=m2s, m3=m3s, m4=m4s, m5=m5s
    )


def _effect_size_label(d: float) -> str:
    """Cohen's d magnitude label (Cohen 1988 thresholds)."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def _sig_stars(p: Optional[float]) -> str:
    """Return significance stars for a p-value."""
    if p is None:
        return "   "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "   "


def compute_pairwise_significance(
    results_a: List[Dict],
    results_b: List[Dict],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute paired statistical tests (t-test, Wilcoxon, Cohen's d) between two
    experimental conditions. Results are aligned by contract index.
    Returns a dict keyed by metric name, or {"error": ..., "n_pairs": 0} if insufficient data.
    """
    if metrics is None:
        metrics = ["composite", "m1", "m2", "m3", "m4", "m5"]

    raw_a = get_raw_scores(results_a)
    raw_b = get_raw_scores(results_b)

    # Build index-matched pairs
    idx_a = {idx: i for i, idx in enumerate(raw_a["indices"])}
    idx_b = {idx: i for i, idx in enumerate(raw_b["indices"])}
    common = sorted(set(idx_a) & set(idx_b))
    n = len(common)

    if n < 2:
        return {"error": "insufficient paired samples", "n_pairs": n}

    output: Dict[str, Any] = {"n_pairs": n}

    for metric in metrics:
        a_vals = [raw_a[metric][idx_a[idx]] for idx in common]
        b_vals = [raw_b[metric][idx_b[idx]] for idx in common]
        diffs = [b - a for a, b in zip(a_vals, b_vals)]

        mean_a = statistics.mean(a_vals)
        mean_b = statistics.mean(b_vals)
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if n > 1 else 0.0
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        entry: Dict[str, Any] = {
            "mean_a": round(mean_a, 3),
            "mean_b": round(mean_b, 3),
            "mean_diff": round(mean_diff, 3),
            "cohens_d": round(cohens_d, 3),
            "effect_size": _effect_size_label(cohens_d),
        }

        if _SCIPY_AVAILABLE:
            t_stat, p_t = ttest_rel(a_vals, b_vals)
            entry["t_stat"] = round(float(t_stat), 3)
            entry["p_value_t"] = round(float(p_t), 4)
            entry["stars"] = _sig_stars(float(p_t))

            if any(d != 0 for d in diffs):
                w_stat, p_w = _wilcoxon(diffs)
                entry["w_stat"] = round(float(w_stat), 3)
                entry["p_value_w"] = round(float(p_w), 4)
            else:
                entry["w_stat"] = 0.0
                entry["p_value_w"] = 1.0
        else:
            # Fallback: manual t-statistic (scipy not available)
            if std_diff > 0:
                t_stat = mean_diff / (std_diff / math.sqrt(n))
                entry["t_stat"] = round(t_stat, 3)
                entry["p_value_t"] = None
                entry["stars"] = "(?)"
                entry["note"] = "install scipy for exact p-values"
            else:
                entry["t_stat"] = 0.0
                entry["p_value_t"] = 1.0
                entry["stars"] = "   "

        output[metric] = entry

    return output


# Statistics
def classify_error_modes(quality_evaluation: Optional[Dict]) -> Dict[str, bool]:
    """
    Map a quality evaluation to structured error-mode flags.
    Categories are non-exclusive (a contract can fail in multiple ways at once).
    """
    if not quality_evaluation:
        return {
            "logic_omission": True,
            "state_transition_err": True,
            "access_control_fail": True,
            "economic_logic_err": True,
            "naming_deviation": True,
            "temporal_logic_miss": True,
            "code_quality_low": True,
            "fully_correct": False,
        }

    scores = extract_scores(quality_evaluation)
    m1, m2, m3, m4, m5 = (
        scores["m1"],
        scores["m2"],
        scores["m3"],
        scores["m4"],
        scores["m5"],
    )

    # Check if spec had dates but M4 temporal subscore is 0
    temporal_subscore = (
        quality_evaluation.get("metric_4_business_logic", {})
        .get("temporal_logic", {})
        .get("points", None)
    )
    total_dates = (
        quality_evaluation.get("metric_4_business_logic", {})
        .get("temporal_logic", {})
        .get("total_dates", 0)
    )

    # M1 missing access control flag
    m1_missing_access_ctrl = bool(
        quality_evaluation.get("metric_1_functional_completeness", {})
        .get("implementation_quality", {})
        .get("missing_access_control", [])
    )

    # M4 financial logic failures
    financial_missing = bool(
        quality_evaluation.get("metric_4_business_logic", {})
        .get("financial_logic", {})
        .get("missing_financial_logic", [])
    )

    composite = scores["composite"]

    return {
        "logic_omission": m1 < 60,
        "state_transition_err": m3 < 60,
        "access_control_fail": m1_missing_access_ctrl and m4 < 70,
        "economic_logic_err": financial_missing or (m4 < 60),
        "naming_deviation": m2 < 70,
        "temporal_logic_miss": (total_dates > 0 and temporal_subscore == 0),
        "code_quality_low": m5 < 50,
        "fully_correct": composite >= 80,
    }


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics over a list of experiment results."""
    composites, m1s, m2s, m3s, m4s, m5s = [], [], [], [], [], []
    gt_composites, gt_m1s, gt_m2s, gt_m3s, gt_m4s, gt_m5s = [], [], [], [], [], []
    compiled, gt_compiled, processing_times = [], [], []
    errors = 0
    error_mode_counts: Dict[str, int] = {}

    for r in results:
        if r.get("error"):
            errors += 1
            continue

        scores = extract_scores(r.get("quality_evaluation"))
        composites.append(scores["composite"])
        m1s.append(scores["m1"])
        m2s.append(scores["m2"])
        m3s.append(scores["m3"])
        m4s.append(scores["m4"])
        m5s.append(scores["m5"])

        # Ground-truth contract scores (when available)
        gt_qe = r.get("ground_truth_quality_evaluation")
        if gt_qe:
            gt_scores = extract_scores(gt_qe)
            gt_composites.append(gt_scores["composite"])
            gt_m1s.append(gt_scores["m1"])
            gt_m2s.append(gt_scores["m2"])
            gt_m3s.append(gt_scores["m3"])
            gt_m4s.append(gt_scores["m4"])
            gt_m5s.append(gt_scores["m5"])

        # Error mode classification
        error_modes = classify_error_modes(r.get("quality_evaluation"))
        for mode, active in error_modes.items():
            if active:
                error_mode_counts[mode] = error_mode_counts.get(mode, 0) + 1

        c = compilation_success(r.get("compilation"))
        if c is not None:
            compiled.append(int(c))

        gtc = compilation_success(r.get("ground_truth_compilation"))
        if gtc is not None:
            gt_compiled.append(int(gtc))

        if r.get("processing_time"):
            processing_times.append(r["processing_time"])

    def _stat(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    n_valid = len(results) - errors
    error_mode_rates = {
        mode: round(count / n_valid, 4) if n_valid else None
        for mode, count in error_mode_counts.items()
    }

    return {
        "n_total": len(results),
        "n_errors": errors,
        # Generated contract quality
        "composite": _stat(composites),
        "m1_functional": _stat(m1s),
        "m2_variable": _stat(m2s),
        "m3_state_machine": _stat(m3s),
        "m4_business_logic": _stat(m4s),
        "m5_code_quality": _stat(m5s),
        # Ground-truth quality (same rubric)
        "gt_composite": _stat(gt_composites),
        "gt_m1_functional": _stat(gt_m1s),
        "gt_m2_variable": _stat(gt_m2s),
        "gt_m3_state_machine": _stat(gt_m3s),
        "gt_m4_business_logic": _stat(gt_m4s),
        "gt_m5_code_quality": _stat(gt_m5s),
        # Compilation rates
        "compilation_rate": (
            round(sum(compiled) / len(compiled), 4) if compiled else None
        ),
        "gt_compilation_rate": (
            round(sum(gt_compiled) / len(gt_compiled), 4) if gt_compiled else None
        ),
        # Error mode breakdown
        "error_mode_rates": error_mode_rates,
        "avg_processing_s": (
            round(statistics.mean(processing_times), 2) if processing_times else None
        ),
    }


def print_stats_table(label: str, stats: Dict) -> None:
    """Pretty-print a statistics dict, including GT quality scores when available."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  N total / errors : {stats['n_total']} / {stats['n_errors']}")

    def _fmt(d, key="mean"):
        v = d.get(key) if d else None
        return f"{v:.2f}" if v is not None else "N/A"

    def _fmtpair(gen_key: str, gt_key: str) -> str:
        gen = _fmt(stats.get(gen_key))
        gt = _fmt(stats.get(gt_key))
        if gt == "N/A":
            return gen
        delta = None
        gmean = stats.get(gen_key, {}).get("mean")
        gtmean = stats.get(gt_key, {}).get("mean")
        if gmean is not None and gtmean is not None:
            delta = gmean - gtmean
        delta_str = f"  (Δ={delta:+.2f} gen vs GT)" if delta is not None else ""
        return f"{gen}  [GT: {gt}]{delta_str}"

    print(f"\n  {'Metric':<22} {'Generated':>10}   {'GT (same rubric)':>18}")
    print(f"  {'-'*55}")
    print(f"  {'Composite score':<22} {_fmtpair('composite', 'gt_composite')}")
    print(f"  {'M1 Functional':<22} {_fmtpair('m1_functional', 'gt_m1_functional')}")
    print(f"  {'M2 Variable':<22} {_fmtpair('m2_variable', 'gt_m2_variable')}")
    print(
        f"  {'M3 State Machine':<22} {_fmtpair('m3_state_machine', 'gt_m3_state_machine')}"
    )
    print(
        f"  {'M4 Business Logic':<22} {_fmtpair('m4_business_logic', 'gt_m4_business_logic')}"
    )
    print(
        f"  {'M5 Code Quality':<22} {_fmtpair('m5_code_quality', 'gt_m5_code_quality')}"
    )

    print()
    cr = stats.get("compilation_rate")
    print(
        f"  Compilation rate : {cr*100:.1f}%"
        if cr is not None
        else "  Compilation rate : N/A"
    )
    gtcr = stats.get("gt_compilation_rate")
    print(
        f"  GT compile rate  : {gtcr*100:.1f}%"
        if gtcr is not None
        else "  GT compile rate  : N/A"
    )
    t = stats.get("avg_processing_s")
    print(f"  Avg time/contract: {t:.1f}s" if t else "  Avg time/contract: N/A")

    # Error-mode breakdown
    em = stats.get("error_mode_rates", {})
    if em:
        print(f"\n  Error Mode Rates (fraction of contracts):")
        mode_labels = {
            "logic_omission": "Logic Omission (M1<60)",
            "state_transition_err": "State Transition Error (M3<60)",
            "access_control_fail": "Access Control Failure",
            "economic_logic_err": "Economic Logic Error (M4<60)",
            "naming_deviation": "Naming Deviation (M2<70)",
            "temporal_logic_miss": "Temporal Logic Missing",
            "code_quality_low": "Code Quality Low (M5<50)",
            "fully_correct": "Fully Correct (composite≥80)",
        }
        for mode, lbl in mode_labels.items():
            rate = em.get(mode)
            if rate is not None:
                print(f"    {lbl:<40} {rate*100:5.1f}%")

    print(f"{'='*70}\n")
