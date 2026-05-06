"""
experiment_6_contract_type.py
------------------------------
Cross-Contract-Type Generalization Analysis

Categorises the FSM-SCG dataset contracts by common smart contract archetype
(ERC-20, ERC-721, escrow, auction, governance/voting, access control, timelock,
multi-sig, crowdfund, general), then runs the full pipeline on a balanced sample
of each type and reports per-category quality scores and compilation rates.

Why this matters for the rebuttal
----------------------------------
Reviewer J5uF asks:
  "How does the pipeline generalise beyond the specific contract types shown?"
Reviewer inVW asks:
  "The evaluation seems biased towards simple, spec-following examples."

Showing that the pipeline achieves similar quality scores across structurally
diverse contract categories — spanning token standards, financial instruments,
governance mechanisms, and access-control patterns — directly addresses both
concerns.  It also establishes which contract categories are hardest for the
LLM pipeline, providing actionable guidance for future work.

Category Definitions (keyword-based; lightweight, no extra API calls)
----------------------------------------------------------------------
  erc20       — ERC-20 / fungible token / transfer / approve / allowance
  erc721      — ERC-721 / NFT / non-fungible / tokenURI / safeTransfer
  escrow      — escrow / hold / release / deposit / arbitrat
  auction     — auction / bid / reserve / highest bidder / lot
  voting      — vote / ballot / proposal / quorum / delegate / governance
  access_ctrl — role / permission / whitelist / blacklist / grant / revoke
  timelock    — timelock / time lock / delay / schedule / execute / cancel
  multisig    — multi-sig / multisig / multi-signature / threshold / approvals
  crowdfund   — crowdfund / fundraise / goal / pledge / contributor / refund
  general     — anything that doesn't match the above

Usage:
  python experiment_6_contract_type.py \\
      --dataset /path/to/requirement_code.jsonl \\
      --n_per_category 30 \\
      --output_dir ./results/contract_type \\
      --model gpt-4o-mini
"""

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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
    compute_statistics,
    evaluate_ground_truth_quality,
    extract_scores,
    load_dataset,
    print_stats_table,
    save_results,
)

from applications.solidity_compiler import SolidityCompilationChecker
from applications.translator import IBMAgenticContractTranslator

# Contract-type taxonomy
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "erc20": [
        "erc20",
        "erc-20",
        "fungible token",
        "token transfer",
        "transferfrom",
        "allowance",
        "approve",
        "mint",
        "burn",
        "total supply",
        "balanceof",
    ],
    "erc721": [
        "erc721",
        "erc-721",
        "nft",
        "non-fungible",
        "tokenuRI",
        "safetransfer",
        "ownerof",
        "tokenid",
        "metadata",
    ],
    "escrow": [
        "escrow",
        "third party hold",
        "release funds",
        "arbitrat",
        "deposit and release",
        "intermediary",
        "locked funds",
    ],
    "auction": [
        "auction",
        "bid",
        "highest bidder",
        "reserve price",
        "lot",
        "winner",
        "auction end",
        "place bid",
    ],
    "voting": [
        "vote",
        "ballot",
        "proposal",
        "quorum",
        "delegate",
        "governance",
        "governance token",
        "cast vote",
        "tally",
    ],
    "access_ctrl": [
        "role",
        "permission",
        "whitelist",
        "blacklist",
        "grant role",
        "revoke role",
        "access control",
        "onlyowner",
        "only owner",
        "only admin",
    ],
    "timelock": [
        "timelock",
        "time lock",
        "time-lock",
        "delay",
        "schedule",
        "execute after",
        "queued",
        "cancel pending",
    ],
    "multisig": [
        "multi-sig",
        "multisig",
        "multi-signature",
        "m-of-n",
        "threshold",
        "multiple approvals",
        "require confirmations",
    ],
    "crowdfund": [
        "crowdfund",
        "fundraise",
        "fundraising",
        "campaign goal",
        "pledge",
        "contributor",
        "refund if goal",
        "crowdsale",
    ],
}


def classify_contract_type(text: str) -> str:
    """
    Return the first matching category for a requirement-text string, or
    'general' if no category keywords are found.

    Uses case-insensitive substring matching — no model calls required.
    """
    lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "general"


def categorise_dataset(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Partition records into per-category lists."""
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        cat = classify_contract_type(record["requirement"])
        record["contract_type"] = cat
        by_category[cat].append(record)
    return dict(by_category)


# Per-contract processing (mirrors experiment_1 conditions B/C/D)
def process_contract(
    translator: IBMAgenticContractTranslator,
    record: Dict,
    checker: SolidityCompilationChecker,
) -> Dict:
    """Run one contract through the full pipeline and collect scores."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(record["requirement"])
        tmp_path = tmp.name

    result = {
        "index": record["index"],
        "contract_type": record.get("contract_type", "unknown"),
        "requirement": record["requirement"][:200],
        "solidity": None,
        "compilation": None,
        "ground_truth_compilation": None,
        "quality_evaluation": None,
        "ground_truth_quality_evaluation": None,
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

            if phase == 3:
                result["solidity"] = data.get("solidity")
            elif phase == 4:
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

    # Ground-truth compilation and quality evaluation
    if record.get("code"):
        result["ground_truth_compilation"] = checker.check_compilation(record["code"])

        if hasattr(translator, "quality_evaluator_agent"):
            result["ground_truth_quality_evaluation"] = evaluate_ground_truth_quality(
                ground_truth_code=record["code"],
                schema=None,
                requirement=record["requirement"],
                quality_evaluator_agent=translator.quality_evaluator_agent,
            )

    return result


# Per-category statistics
def stats_for(results: List[Dict]) -> Dict:
    """Lightweight stat dict for a list of results within one category."""
    composites, compile_flags = [], []
    m1s, m2s, m3s, m4s, m5s = [], [], [], [], []
    errors = 0

    for r in results:
        if r.get("error"):
            errors += 1
            continue
        s = extract_scores(r.get("quality_evaluation"))
        composites.append(s["composite"])
        m1s.append(s["m1"])
        m2s.append(s["m2"])
        m3s.append(s["m3"])
        m4s.append(s["m4"])
        m5s.append(s["m5"])
        c = compilation_success(r.get("compilation"))
        if c is not None:
            compile_flags.append(int(c))

    def _avg(lst):
        return round(statistics.mean(lst), 2) if lst else None

    def _std(lst):
        return round(statistics.stdev(lst), 2) if len(lst) > 1 else 0.0

    return {
        "n": len(results),
        "n_errors": errors,
        "composite_mean": _avg(composites),
        "composite_std": _std(composites),
        "m1_mean": _avg(m1s),
        "m2_mean": _avg(m2s),
        "m3_mean": _avg(m3s),
        "m4_mean": _avg(m4s),
        "m5_mean": _avg(m5s),
        "compilation_rate": (
            round(sum(compile_flags) / len(compile_flags), 3) if compile_flags else None
        ),
    }


# Reporting
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


def print_type_report(category_stats: Dict[str, Dict]) -> None:
    print(f"\n{'='*90}")
    print("  CROSS-CONTRACT-TYPE GENERALIZATION — REBUTTAL TABLE")
    print(f"{'='*90}")
    print(
        f"  {'Category':<24} {'N':>4}  {'Avg Score':>10}  {'±Std':>6}  "
        f"{'Compile%':>9}  {'M1':>6}  {'M3':>6}  {'M4':>6}"
    )
    print(f"  {'-'*80}")

    all_composites = []
    for cat in sorted(category_stats):
        s = category_stats[cat]
        nm = CATEGORY_DISPLAY.get(cat, cat)
        n = s["n"]
        avg = f"{s['composite_mean']:.2f}" if s["composite_mean"] is not None else "N/A"
        std = f"{s['composite_std']:.2f}" if s["composite_std"] is not None else "N/A"
        cr = (
            f"{s['compilation_rate']*100:.1f}%"
            if s["compilation_rate"] is not None
            else "N/A"
        )
        m1 = f"{s['m1_mean']:.1f}" if s["m1_mean"] is not None else "N/A"
        m3 = f"{s['m3_mean']:.1f}" if s["m3_mean"] is not None else "N/A"
        m4 = f"{s['m4_mean']:.1f}" if s["m4_mean"] is not None else "N/A"

        if s["composite_mean"] is not None:
            all_composites.append(s["composite_mean"])

        print(
            f"  {nm:<24} {n:>4}  {avg:>10}  {std:>6}  {cr:>9}  {m1:>6}  {m3:>6}  {m4:>6}"
        )

    print(f"  {'─'*80}")

    if len(all_composites) > 1:
        overall_mean = round(statistics.mean(all_composites), 2)
        cross_type_std = round(statistics.stdev(all_composites), 2)
        print(f"\n  Overall mean composite score (all types): {overall_mean:.2f}")
        print(f"  Std deviation across contract types:       {cross_type_std:.2f}")
        if cross_type_std < 10:
            print(
                f"\n  → Low cross-type variance ({cross_type_std:.2f} pts) confirms the pipeline\n"
                f"    generalises robustly across diverse smart contract archetypes.\n"
                f"    Results are not driven by a single easy contract category."
            )
        else:
            best = max(
                category_stats, key=lambda c: category_stats[c]["composite_mean"] or 0
            )
            worst = min(
                category_stats, key=lambda c: category_stats[c]["composite_mean"] or 999
            )
            print(
                f"\n  → Best category:  {CATEGORY_DISPLAY.get(best, best)} "
                f"({category_stats[best]['composite_mean']:.2f})\n"
                f"    Worst category: {CATEGORY_DISPLAY.get(worst, worst)} "
                f"({category_stats[worst]['composite_mean']:.2f})\n"
                f"    The gap indicates which contract types benefit most from\\n"
                f"    improved specification quality or additional training examples."
            )

    print(f"{'='*90}\n")


# Dataset composition table (no LLM needed)
def print_dataset_composition(by_category: Dict[str, List[Dict]]) -> None:
    total = sum(len(v) for v in by_category.values())
    print(f"\n  DATASET COMPOSITION BY CONTRACT TYPE  (full dataset)")
    print(f"  {'Category':<24} {'N':>6}  {'%':>7}")
    print(f"  {'─'*40}")
    for cat in sorted(by_category, key=lambda c: -len(by_category[c])):
        n = len(by_category[cat])
        pct = n / total * 100
        nm = CATEGORY_DISPLAY.get(cat, cat)
        print(f"  {nm:<24} {n:>6}  {pct:>6.1f}%")
    print(f"  {'─'*40}")
    print(f"  {'Total':<24} {total:>6}")
    print()


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="Cross-contract-type generalization analysis"
    )
    parser.add_argument("--dataset", default="requirement_code.jsonl")
    parser.add_argument(
        "--n_per_category",
        type=int,
        default=30,
        help="Contracts to sample per category (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./results/contract_type")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Restrict to these category names (e.g. erc20 auction escrow voting crowdfund)",
    )
    parser.add_argument(
        "--composition_only",
        action="store_true",
        help="Print dataset composition and exit (no LLM calls)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset for composition analysis, then sample per category
    all_records = load_dataset(args.dataset, n_samples=-1, seed=args.seed)
    by_category = categorise_dataset(all_records)

    print_dataset_composition(by_category)

    if args.composition_only:
        comp_path = output_dir / "dataset_composition.json"
        with open(comp_path, "w") as f:
            json.dump(
                {cat: len(recs) for cat, recs in by_category.items()},
                f,
                indent=2,
            )
        print(f"[contract_type] Composition saved → {comp_path}")
        return

    # Sample up to n_per_category from each category
    import random

    random.seed(args.seed)

    sampled: Dict[str, List[Dict]] = {}
    for cat, recs in by_category.items():
        if args.categories and cat not in args.categories:
            continue
        if len(recs) >= 5:  # skip categories with very few examples
            sampled[cat] = random.sample(recs, min(args.n_per_category, len(recs)))

    total_contracts = sum(len(v) for v in sampled.values())
    print(
        f"[contract_type] Processing {total_contracts} contracts "
        f"across {len(sampled)} categories\n"
    )

    checker = SolidityCompilationChecker()
    translator = IBMAgenticContractTranslator(
        model=args.model,
        enable_reinforcement=True,
        max_refinement_iterations=1,  # paper config (Condition C)
    )

    all_results: Dict[str, List[Dict]] = {}
    category_stats: Dict[str, Dict] = {}

    for cat, records in sampled.items():
        nm = CATEGORY_DISPLAY.get(cat, cat)
        print(f"\n{'#'*65}")
        print(f"# Category: {nm}  (N={len(records)})")
        print(f"{'#'*65}\n")

        cat_path = output_dir / f"results_{cat}.jsonl"
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        open(cat_path, "w").close()  # truncate for a fresh run
        cat_results = []
        for i, record in enumerate(records):
            print(
                f"  [{i+1}/{len(records)}] idx={record['index']} ...",
                end=" ",
                flush=True,
            )
            r = process_contract(translator, record, checker)
            cat_results.append(r)
            append_result(r, cat_path)

            scores = extract_scores(r.get("quality_evaluation"))
            comp = compilation_success(r.get("compilation"))
            status = "✓" if not r["error"] else "✗"
            print(
                f"{status}  score={scores['composite']:.1f}  "
                f"compile={'Y' if comp else ('N' if comp is False else '?')}  "
                f"time={r['processing_time']}s"
            )

        # Save raw results per category
        save_results(cat_results, str(output_dir / f"results_{cat}.jsonl"))

        all_results[cat] = cat_results
        category_stats[cat] = stats_for(cat_results)

        print_stats_table(f"Category: {nm}", compute_statistics(cat_results))

    # Print cross-type summary table
    print_type_report(category_stats)

    # Save summary JSON
    summary = {
        "model": args.model,
        "n_per_category": args.n_per_category,
        "seed": args.seed,
        "categories": category_stats,
        "dataset_composition": {
            cat: len(by_category.get(cat, [])) for cat in CATEGORY_KEYWORDS
        },
    }
    summary_path = output_dir / "contract_type_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[contract_type] Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
