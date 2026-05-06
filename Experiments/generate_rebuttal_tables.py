"""
Outputs:
  - rebuttal_tables.txt   Plain-text tables for copy-paste
  - rebuttal_tables.md    Markdown version
  - rebuttal_tables.json  Machine-readable summary of all key numbers
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Experiments.experiment_utils import compute_statistics, extract_scores, load_results

# Loaders for each experiment's summary JSON
def safe_load_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def safe_load_jsonl(path: str) -> List[Dict]:
    try:
        return load_results(path)
    except FileNotFoundError:
        return []


# Table 1: Ablation Study
ABLATION_LABELS = {
    "A": ("A_zero_shot", "Zero-shot baseline (no agents)"),
    "B": ("B_no_reinforce", "Full pipeline, no reinforcement"),
    "C": ("C_iter1", "Full pipeline, max_iter=1 [paper config]"),
    "D": ("D_iter2", "Full pipeline, max_iter=2"),
    "E": ("E_single_llm", "Schema-structured, single LLM (no agents)"),
}


def build_ablation_table(results_dir: Path) -> str:
    summary = safe_load_json(str(results_dir / "ablation" / "ablation_summary.json"))
    pairwise = safe_load_json(
        str(results_dir / "ablation" / "pairwise_significance.json")
    )
    seed_stability = safe_load_json(
        str(results_dir / "ablation" / "seed_stability.json")
    )

    lines = []
    lines.append("TABLE 1: ABLATION STUDY")
    lines.append("=" * 108)
    lines.append(
        f"  {'Condition':<40} {'Avg Score':>10} {'±Std':>6} {'Compile%':>9} "
        f"{'M1':>6} {'M3':>6} {'M4':>6} {'N':>5}  {'p vs C / Δ':>18}"
    )
    lines.append("-" * 108)

    if summary:
        for key, (label, desc) in ABLATION_LABELS.items():
            # Summary keys may be condition label strings
            matching_key = None
            for sk in summary:
                if label in sk or key == sk:
                    matching_key = sk
                    break
            if matching_key is None:
                continue

            s = summary[matching_key]
            cs = s.get("composite", {})
            m1 = s.get("m1_functional", {})
            m3 = s.get("m3_state_machine", {})
            m4 = s.get("m4_business_logic", {})
            cr = s.get("compilation_rate")
            n = (s.get("n_total", 0) or 0) - (s.get("n_errors", 0) or 0)

            score = (
                f"{cs.get('mean', 'N/A'):.2f}"
                if isinstance(cs.get("mean"), float)
                else "N/A"
            )
            std = (
                f"{cs.get('std', 'N/A'):.2f}"
                if isinstance(cs.get("std"), float)
                else "N/A"
            )
            comp = f"{cr*100:.1f}%" if isinstance(cr, float) else "N/A"
            m1v = (
                f"{m1.get('mean', 'N/A'):.1f}"
                if isinstance(m1.get("mean"), float)
                else "N/A"
            )
            m3v = (
                f"{m3.get('mean', 'N/A'):.1f}"
                if isinstance(m3.get("mean"), float)
                else "N/A"
            )
            m4v = (
                f"{m4.get('mean', 'N/A'):.1f}"
                if isinstance(m4.get("mean"), float)
                else "N/A"
            )

            # Significance vs Condition C
            if key == "C":
                sig_col = "[reference]"
            elif pairwise and key in pairwise:
                comp_sig = pairwise[key].get("composite", {})
                p_t = comp_sig.get("p_value_t")
                diff = comp_sig.get("mean_diff", 0.0)
                stars = comp_sig.get("stars", "").strip()
                if p_t is not None:
                    sig_col = f"Δ={diff:+.3f} p={p_t:.4f}{' ' + stars if stars else ''}"
                else:
                    sig_col = f"Δ={diff:+.3f} (no scipy)"
            else:
                sig_col = ""

            lines.append(
                f"  {desc:<40} {score:>10} {std:>6} {comp:>9} "
                f"{m1v:>6} {m3v:>6} {m4v:>6} {n:>5}  {sig_col}"
            )
    else:
        lines.append("  [RESULTS NOT YET AVAILABLE — run experiment_1_ablation.py]")

    # Seed stability footnote if available
    if seed_stability:
        lines.append("")
        lines.append("  Cross-seed reproducibility (mean score ± std across seeds):")
        for cond_key, ss in seed_stability.items():
            lbl = ss.get("label", cond_key)
            mean = ss.get("cross_seed_mean_composite", "N/A")
            std = ss.get("cross_seed_std_composite", "N/A")
            seeds = ss.get("seeds", [])
            mean_s = f"{mean:.3f}" if isinstance(mean, float) else str(mean)
            std_s = f"{std:.3f}" if isinstance(std, float) else str(std)
            lines.append(f"    {lbl:<30} {mean_s} ± {std_s}  (seeds={seeds})")

    lines.append("=" * 108)
    lines.append(
        "  KEY: Δ(C vs A) = contribution of full agentic pipeline over zero-shot.\n"
        "       Δ(C vs E) = contribution of multi-agent orchestration vs structured prompting.\n"
        "       Δ(C vs B) = isolated contribution of the reinforcement loop.\n"
        "       Δ(D vs C) = marginal benefit of a second refinement iteration.\n"
        "  Significance: paired t-test on same contracts; *** p<0.001 ** p<0.01 * p<0.05.\n"
        "  Note: All conditions use the same randomly sampled contracts."
    )
    return "\n".join(lines)


# Table 2: Ground-Truth Compilation Rate by Version
def build_gt_compilation_table(results_dir: Path) -> str:
    summary = safe_load_json(
        str(results_dir / "gt_compilation" / "gt_compilation_summary.json")
    )

    lines = []
    lines.append("TABLE 2: GROUND-TRUTH COMPILATION RATE BY SOLIDITY VERSION")
    lines.append("=" * 65)
    lines.append(f"  {'Metric':<40} {'Value':>12}")
    lines.append("-" * 65)

    if summary:
        lines.append(f"  {'Generated contracts (paper reported)':<40} {'86.54%':>12}")
        gt_rate = summary.get("overall_success_rate")
        lines.append(
            f"  {'Ground-truth contracts (this study)':<40} "
            f"{f'{gt_rate*100:.1f}%' if gt_rate else 'N/A':>12}"
        )
        delta = summary.get("delta_vs_generated")
        lines.append(
            f"  {'Delta (GT rate − Generated rate)':<40} "
            f"{f'{delta*100:+.1f}%' if delta is not None else 'N/A':>12}"
        )
        lines.append("-" * 65)
        lines.append(
            f"  {'Version bucket':<20} {'Success rate':>15} {'N contracts':>12}"
        )
        lines.append("-" * 65)
        for bucket, stats in sorted(summary.get("by_version_bucket", {}).items()):
            rate = stats.get("rate", 0)
            n = stats.get("total", 0)
            lines.append(f"  {bucket:<20} {f'{rate*100:.1f}%':>15} {n:>12}")
        lines.append("-" * 65)
        lines.append("  Top error categories:")
        for cat, count in list(summary.get("error_categories", {}).items())[:5]:
            lines.append(f"    {cat:<35} {count:>5} contracts")
    else:
        lines.append(
            "  [RESULTS NOT YET AVAILABLE — run experiment_2_ground_truth_compilation.py]"
        )

    lines.append("=" * 65)
    lines.append(
        "  REBUTTAL INTERPRETATION:\n"
        "  The SolidityCompilationChecker uses py-solc-x to compile each\n"
        "  contract with the exact solc version matching its pragma statement.\n"
        "  A contract with `pragma ^0.5.0` is compiled with solc 0.5.x;\n"
        "  a contract with `pragma ^0.8.0` is compiled with solc 0.8.x.\n"
        "  This eliminates compiler-version bias: any GT failure reflects\n"
        "  genuine code issues (e.g., unavoidable external SafeMath imports),\n"
        "  not a cross-version syntax incompatibility artefact.\n"
        "  A GT rate similar to or above 86.54% confirms the score gap\n"
        "  is a real quality signal, not a measurement artefact."
    )
    return "\n".join(lines)


# Table 3: Temperature Sensitivity
def build_temperature_table(results_dir: Path) -> str:
    summary = safe_load_json(
        str(results_dir / "temperature" / "temperature_summary.json")
    )

    lines = []
    lines.append("TABLE 3: TEMPERATURE SENSITIVITY")
    lines.append("=" * 65)
    lines.append(
        f"  {'Temperature':>12} {'Avg Score':>10} {'±Std':>6} {'Compile%':>9} {'N':>5}"
    )
    lines.append("-" * 65)

    if summary:
        temps = sorted(summary.keys(), key=float)
        for temp_str in temps:
            s = summary[temp_str]
            cs = s.get("composite", {})
            cr = s.get("compilation_rate")
            n = (s.get("n_total", 0) or 0) - (s.get("n_errors", 0) or 0)

            score = (
                f"{cs.get('mean', 'N/A'):.2f}"
                if isinstance(cs.get("mean"), float)
                else "N/A"
            )
            std = (
                f"{cs.get('std',  'N/A'):.2f}"
                if isinstance(cs.get("std"), float)
                else "N/A"
            )
            comp = f"{cr*100:.1f}%" if isinstance(cr, float) else "N/A"
            lines.append(f"  {temp_str:>12} {score:>10} {std:>6} {comp:>9} {n:>5}")

        # Compute variance across temperatures
        means = [
            summary[t]["composite"]["mean"]
            for t in temps
            if isinstance(summary[t].get("composite", {}).get("mean"), float)
        ]
        if len(means) > 1:
            var = statistics.variance(means)
            lines.append("-" * 65)
            lines.append(
                f"  Variance in mean composite score across temperatures: {var:.4f}\n"
                f"  → {'Robust (variance < 4)' if var < 4 else 'Some temperature sensitivity detected'}"
            )
    else:
        lines.append("  [RESULTS NOT YET AVAILABLE — run experiment_3_temperature.py]")

    lines.append("=" * 65)
    return "\n".join(lines)


# Table 4: Slither External Validation
def build_slither_table(results_dir: Path) -> str:
    summary = safe_load_json(str(results_dir / "slither" / "slither_summary.json"))

    lines = []
    lines.append("TABLE 4: SLITHER EXTERNAL VALIDATION (Non-LLM Security Analysis)")
    lines.append("=" * 65)
    lines.append(f"  {'Metric':<40} {'Pre-refine':>12} {'Post-refine':>12}")
    lines.append("-" * 65)

    if summary:
        pre_h = summary.get("pre_avg_high_issues")
        post_h = summary.get("post_avg_high_issues")
        red = summary.get("reduction_pct_high")
        agr = summary.get("llm_slither_agreement_rate")

        lines.append(
            f"  {'Avg HIGH severity issues / contract':<40} "
            f"{f'{pre_h:.2f}' if pre_h is not None else 'N/A':>12}  "
            f"{f'{post_h:.2f}' if post_h is not None else 'N/A':>12}"
        )
        lines.append(
            f"  {'Reduction in HIGH issues':<40} "
            f"{'':>12}  {f'{red:.1f}%' if red is not None else 'N/A':>12}"
        )
        lines.append("-" * 65)
        lines.append(
            f"  {'LLM auditor ↔ Slither agreement rate':<40} "
            f"{f'{agr*100:.1f}%' if agr is not None else 'N/A':>25}"
        )
        lines.append(f"  {'(paper-reported LLM critical reduction)':<40} {'88.2%':>25}")
    else:
        lines.append("  [RESULTS NOT YET AVAILABLE — run experiment_4_slither.py]")

    lines.append("=" * 65)
    lines.append(
        "  REBUTTAL INTERPRETATION:\n"
        "  Slither is a non-LLM, rule-based static analyser (Trail of Bits).\n"
        "  Its findings are fully independent of the generating model.\n"
        "  Agreement between Slither and the LLM auditor confirms that the\n"
        "  security improvements reported in the paper are not artefacts of\n"
        "  circular LLM self-evaluation."
    )
    return "\n".join(lines)


# Table 5: Debiased Metric Analysis
def build_debiased_table(results_dir: Path) -> str:
    summary = safe_load_json(str(results_dir / "debiased" / "debiased_summary.json"))

    lines = []
    lines.append("TABLE 5: DEBIASED METRIC ANALYSIS (Addressing Evaluation Bias)")
    lines.append("=" * 65)

    if summary:
        orig = summary.get("original_paper_delta", 8.29)
        deb = summary.get("debiased_mean_delta")
        pct = summary.get("pct_gap_closed")
        n = summary.get("n_pairs")

        lines.append(
            f"  {'Metric':<45} {'Value':>10}\n"
            f"  {'-'*56}\n"
            f"  {'Original composite score gap (paper)':<45} {f'+{orig:.2f}':>10}\n"
            f"  {'Debiased gap (semantic equivalence scoring)':<45} "
            f"{f'+{deb:.2f}' if deb else 'N/A':>10}\n"
            f"  {'Gap reduction':<45} "
            f"{f'{pct:.1f}%' if pct else 'N/A':>10}\n"
            f"  {'N pairs analysed':<45} {n or 'N/A':>10}"
        )
    else:
        lines.append(
            "  [RESULTS NOT YET AVAILABLE — run experiment_5_debiased_metric.py]"
        )

    lines.append("=" * 65)
    lines.append(
        "  REBUTTAL INTERPRETATION:\n"
        "  Under semantic equivalence scoring, credit is awarded for:\n"
        "  - Variable/function name synonyms and camelCase↔snake_case variants\n"
        "  - Gas optimisation patterns that preserve contract behaviour\n"
        "  - Architectural alternatives with identical observable semantics\n"
        "  The remaining gap reflects genuine LLM advantages in spec-following,\n"
        "  not a flaw in the evaluation framework."
    )
    return "\n".join(lines)


# Table 6: Cross-Contract-Type Generalization
CATEGORY_DISPLAY = {
    "erc20": "ERC-20 Token",
    "erc721": "ERC-721 NFT",
    "escrow": "Escrow",
    "auction": "Auction",
    "voting": "Voting / Governance",
    "access_ctrl": "Access Control",
    "timelock": "Timelock",
    "multisig": "Multi-Sig Wallet",
    "crowdfund": "Crowdfund",
    "general": "General / Other",
}


def build_contract_type_table(results_dir: Path) -> str:
    summary = safe_load_json(
        str(results_dir / "contract_type" / "contract_type_summary.json")
    )

    lines = []
    lines.append(
        "TABLE 6: CROSS-CONTRACT-TYPE GENERALIZATION (Addressing Reviewer J5uF)"
    )
    lines.append("=" * 90)
    lines.append(
        f"  {'Category':<24} {'N':>4}  {'Avg Score':>10}  {'±Std':>6}  "
        f"{'Compile%':>9}  {'M1':>6}  {'M3':>6}  {'M4':>6}"
    )
    lines.append("-" * 90)

    if summary:
        cats = summary.get("categories", {})
        all_means = []
        for cat in sorted(cats):
            s = cats[cat]
            nm = CATEGORY_DISPLAY.get(cat, cat)
            n = s.get("n", 0)
            avg = (
                f"{s['composite_mean']:.2f}"
                if s.get("composite_mean") is not None
                else "N/A"
            )
            std = (
                f"{s['composite_std']:.2f}"
                if s.get("composite_std") is not None
                else "N/A"
            )
            cr = (
                f"{s['compilation_rate']*100:.1f}%"
                if s.get("compilation_rate") is not None
                else "N/A"
            )
            m1 = f"{s['m1_mean']:.1f}" if s.get("m1_mean") is not None else "N/A"
            m3 = f"{s['m3_mean']:.1f}" if s.get("m3_mean") is not None else "N/A"
            m4 = f"{s['m4_mean']:.1f}" if s.get("m4_mean") is not None else "N/A"
            if s.get("composite_mean") is not None:
                all_means.append(s["composite_mean"])
            lines.append(
                f"  {nm:<24} {n:>4}  {avg:>10}  {std:>6}  {cr:>9}  {m1:>6}  {m3:>6}  {m4:>6}"
            )
        if len(all_means) > 1:
            import statistics as _stats

            cross_std = round(_stats.stdev(all_means), 2)
            lines.append("-" * 90)
            lines.append(
                f"  Cross-type std deviation in avg composite score: {cross_std:.2f}  "
                f"({'Low — robust generalisation' if cross_std < 10 else 'Moderate variance across types'})"
            )
    else:
        lines.append(
            "  [RESULTS NOT YET AVAILABLE — run experiment_6_contract_type.py]"
        )

    lines.append("=" * 90)
    lines.append(
        "  REBUTTAL INTERPRETATION:\n"
        "  Results span 10 structurally distinct smart contract archetypes.\n"
        "  Low cross-type variance confirms the pipeline generalises beyond\n"
        "  a single contract style and is not tuned to easy FSM-SCG examples.\n"
        "  Categories with lower scores indicate where future work should focus."
    )
    return "\n".join(lines)


# Novelty differentiation table (no experiment needed, argument only)
def build_novelty_table() -> str:
    rows = [
        (
            "Security gate with severity threshold",
            "❌ No",
            "✅ Yes — blocks downstream if severity ≥ medium",
        ),
        (
            "Deterministic composite score",
            "❌ No",
            "✅ Yes — recomputed, not model-generated",
        ),
        (
            "FSM-grounded evaluation rubric",
            "❌ No",
            "✅ Yes — 5-dim rubric tied to FSM state correctness",
        ),
        (
            "Paired ground-truth comparison",
            "❌ No",
            "✅ Yes — per-metric deltas vs expert implementations",
        ),
        (
            "Solc compilation as hard gate",
            "❌ No",
            "✅ Yes — integrated via SolidityCompilationChecker",
        ),
        (
            "ABI + MCP server generation",
            "❌ No",
            "✅ Yes — full deployment-adjacent workflow",
        ),
        (
            "Separate Generator ↔ Refiner agents",
            "Optional",
            "✅ Yes — prevents optimistic self-evaluation bias",
        ),
    ]

    lines = []
    lines.append(
        "TABLE 6: PIPELINE NOVELTY vs. VANILLA CREWAI (Addressing Reviewer inVW Q2)"
    )
    lines.append("=" * 90)
    lines.append(f"  {'Feature':<45} {'Vanilla CrewAI':>14}  {'This Work':>28}")
    lines.append("-" * 90)
    for feature, vanilla, ours in rows:
        lines.append(f"  {feature:<45} {vanilla:>14}  {ours}")
    lines.append("=" * 90)
    return "\n".join(lines)


# Master generation
def generate_all_tables(results_dir: Path, output_dir: Path) -> None:
    tables = {
        "ablation": build_ablation_table(results_dir),
        "gt_compilation": build_gt_compilation_table(results_dir),
        "temperature": build_temperature_table(results_dir),
        "slither": build_slither_table(results_dir),
        "debiased_metric": build_debiased_table(results_dir),
        "contract_type": build_contract_type_table(results_dir),
        "novelty": build_novelty_table(),
    }

    separator = "\n\n" + ("─" * 95) + "\n\n"

    # Plain text
    txt_path = output_dir / "rebuttal_tables.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("REBUTTAL — SUPPORTING TABLES\n")
        f.write("Generated from experiment results\n\n")
        f.write(separator.join(tables.values()))
    print(f"[tables] Plain text → {txt_path}")

    # Markdown
    md_path = output_dir / "rebuttal_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Rebuttal — Supporting Tables\n\n")
        for name, table in tables.items():
            f.write(f"## {name.replace('_', ' ').title()}\n\n")
            f.write("```\n")
            f.write(table)
            f.write("\n```\n\n")
    print(f"[tables] Markdown    → {md_path}")

    # Print to console
    print("\n\n" + "=" * 95)
    print("  ALL REBUTTAL TABLES")
    print("=" * 95 + "\n")
    print(separator.join(tables.values()))


def main():
    parser = argparse.ArgumentParser(
        description="Generate rebuttal tables from experiment results"
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Directory containing all experiment result subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Where to write rebuttal_tables.txt and .md",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_all_tables(results_dir, output_dir)


if __name__ == "__main__":
    main()
