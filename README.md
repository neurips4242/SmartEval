# SmartEval: A Benchmark for Evaluating LLM-Generated Smart Contracts from Natural Language Specifications

> Code repository for the NeurIPS 2026 Datasets and Benchmarks submission.

**Dataset:** https://www.kaggle.com/datasets/neurips4242/smarteval-llm-generated-smart-contract-benchmark
**DOI:** https://doi.org/10.5281/zenodo.20046036
**Paper:** *(link to be added upon publication)*

---

## Overview

SmartEval is a benchmark and agentic pipeline for evaluating how well large language models (LLMs) translate natural language smart contract specifications into correct, secure, and deployable Solidity code. The system generates contracts, audits them for security vulnerabilities, iteratively refines them via a severity-gated reinforcement loop, and scores the results across a five-dimensional quality rubric, all in a single reproducible pipeline.

The benchmark dataset contains 9,000 LLM-generated Solidity contracts paired with expert-written ground-truth implementations from the FSM-SCG dataset, each annotated with quality scores, security audit reports, ABI artifacts, and compilation results. The pipeline that produced them is fully contained in this repository and can be used to generate and evaluate new contracts from any natural language specification.

---

## Repository Structure

```
SmartEval/
├── contract-translator/       # Core pipeline package (seven-agent system)
│   ├── core/
│   │   ├── agents.py          # All seven agent instantiations + should_refine() gate
│   │   ├── translator.py      # IBMAgenticContractTranslator orchestrator
│   │   ├── task_builders.py   # Per-agent task prompt constructors
│   │   ├── schemas.py         # Pydantic data models (UniversalContractSchema)
│   │   ├── solidity_compiler.py # Compilation validation via solc/solcjs
│   │   └── programs.py        # Legacy IBM Agentics Program class wrappers
│   ├── demo.html              # Live translation demo interface
│   ├── sampler.html           # Dataset browser interface
│   └── output/                # Generated contract artifacts (per-batch subdirectories)
├── applications/              # Namespace shim: exposes contract-translator/core/ as applications.*
├── mcp/
│   └── chatbot_api.py         # Flask API backend serving demo.html and sampler.html
├── Experiments/               # Ablation study and validation experiment scripts
│   ├── experiment_1_ablation.py          # Five-condition ablation study (conditions A-E)
│   ├── experiment_2_ground_truth_compilation.py  # Ground-truth compilation rate analysis
│   ├── experiment_3_temperature.py       # Temperature sensitivity and reproducibility
│   ├── experiment_4_slither.py           # External Slither validation (§5.10 of paper)
│   ├── experiment_5_debiased_metric.py   # Semantic debiasing of the quality rubric
│   ├── experiment_6_contract_type.py     # Per-contract-type breakdown
│   └── experiment_utils.py    # Shared statistical utilities (Cohen's d, etc.)
├── src/agentics/              # IBM Agentics library (LLM connections, async executor, etc.)
├── launch_demo.py             # One-command demo launcher (starts all servers)
├── launch_demo.bat            # Windows equivalent of launch_demo.py
├── test_checkpoints.py        # Batch checkpoint verification utility
├── requirements.txt           # All Python dependencies
└── __init__.py
```

---

## How the Pipeline Maps to the Paper

The `IBMAgenticContractTranslator` class in `contract-translator/core/translator.py` orchestrates the seven-phase pipeline described in §3 of the paper. Each phase corresponds directly to a named agent:

| Phase | Paper Section | Agent / Module | What It Does |
|---|---|---|---|
| 1 | §3.1 | `agents.py`: Contract Analysis Expert | Parses natural language spec into `UniversalContractSchema` with exact terminology extraction |
| 2 | §3.2 | `agents.py`: Senior Solidity Smart Contract Engineer | Generates 150–400 line Solidity contract following 12 mandatory rules |
| 3 | §3.3 | `agents.py`: Blockchain Security Auditor | Audits across 8 vulnerability categories, returns structured JSON |
| 4 | §3.4 | `agents.py`: Smart Contract Security Refiner + `should_refine()` gate | Severity-gated refinement loop; blocks medium/high/critical severity contracts |
| 5 | §3.5 | `agents.py`: Ethereum ABI Specialist | Generates complete JSON ABI array |
| 6 | §3.5 | `agents.py`: MCP Server Developer | Generates FastMCP Python server exposing contract endpoints |
| 7 | §4.2 | `agents.py`: Smart Contract Quality Analyst | Scores against five-dimensional rubric; compares to ground-truth |

The `should_refine()` function in `agents.py` is the architectural centerpiece of the pipeline. It implements a hard blocking gate that re-enters contracts into the refinement loop when audit severity is medium or higher:

```python
def should_refine(audit_report, refinement_count, max_iterations=2):
    if refinement_count >= max_iterations:
        return False
    severity = audit_report.get('severity_level')
    approved = audit_report.get('approved', False)
    if (not approved and severity in ['medium', 'high', 'critical']):
        return True
    return False
```

Disabling this gate (Condition B in the ablation study) raises output standard deviation by 111% (std 10.31 vs. 4.89) and reduces compilation by 5.2 percentage points, as reported in Table 8 of the paper.

---

## Installation

**Requirements:** Python 3.9+, Node.js (for `solcjs` compilation fallback), an OpenAI API key.

```bash
# Clone the repository
git clone https://github.com/neurips4242/SmartEval.git
cd SmartEval

# Install Python dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes all core dependencies:

```
crewai                  # Multi-agent orchestration framework
openai                  # LLM API client
flask, flask-cors       # Demo API backend
fastmcp, mcp            # Model Context Protocol server generation
web3                    # Blockchain interaction for optional deployment
pydantic                # Schema validation (UniversalContractSchema)
pandas, numpy, scipy    # Data processing and statistical analysis
slither-analyzer        # External security validation (Experiments/experiment_4_slither.py)
sentence-transformers   # Semantic similarity for evaluation
python-dotenv           # Environment variable management
rich, click, tqdm       # CLI utilities
pyyaml                  # Configuration file support
```

**Configure your API key:**

```bash
cp .env.example .env
# Edit .env and set:
# OPENAI_API_KEY=your_openai_api_key_here
```

---

## Quick Start: Running the Pipeline on a Single Contract

```python
import os
import tempfile
from applications.translator import IBMAgenticContractTranslator

os.environ["OPENAI_API_KEY"] = "your_key_here"

translator = IBMAgenticContractTranslator(
    model="gpt-4o-mini",
    enable_reinforcement=True,       # Enables the severity-gated refinement loop
    max_refinement_iterations=1      # Main paper configuration (Condition C)
)

specification = """
A staking contract where users can stake tokens during a farming period.
The contract transitions through three states: Farming Not Started,
Farming Ongoing, and Farming Ended. Users can stake tokens after the
start time, withdraw staked tokens, and claim rewards based on their
staking period. Rewards cease when the end time arrives.
"""

# The pipeline reads from a file path, so write the spec to a temp file
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write(specification)
    tmp_path = f.name

# Stream phase-by-phase updates
for phase_update in translator.translate_contract_streaming(
    input_path=tmp_path,
    output_dir="./output",
    require_audit_approval=False,
    generate_mcp_server=False,
):
    phase = phase_update["phase"]
    status = phase_update["status"]
    print(f"Phase {phase}: {status}")
    if phase == 7:  # Quality evaluation (final phase)
        qe = phase_update["data"].get("quality_evaluation", {})
        score = qe.get("composite_score", {}).get("final_score", 0)
        grade = qe.get("composite_score", {}).get("grade", "?")
        audit = phase_update["data"].get("audit", {})
        print(f"Composite Score: {score}/100")
        print(f"Grade: {grade}")
        print(f"Security Severity: {audit.get('severity_level', 'unknown')}")
```

**Configuration options** (all correspond to ablation study conditions in §4.4):

```python
# Condition C: main paper configuration (recommended)
translator = IBMAgenticContractTranslator(
    enable_reinforcement=True,
    max_refinement_iterations=1
)

# Condition D: two refinement iterations (2.2x compute cost, +0.27 score)
translator = IBMAgenticContractTranslator(
    enable_reinforcement=True,
    max_refinement_iterations=2
)

# Condition B: full pipeline, reinforcement disabled (for ablation replication)
translator = IBMAgenticContractTranslator(
    enable_reinforcement=False
)
```

---

## Interactive Demo: demo.html and sampler.html

The repository includes a full browser-based demo environment with two interfaces that work together: `sampler.html` for browsing the FSM-SCG dataset and `demo.html` for running live contract generation and evaluation. Both are served by `launch_demo.py`, which starts all required backend services automatically.

### Starting the Demo

**Mac/Linux:**
```bash
python launch_demo.py
```

**Windows:**
```bat
launch_demo.bat
```

Or with the legacy implementation mode:
```bash
USE_MODULAR_CORE=false python launch_demo.py
```

`launch_demo.py` starts three services in parallel:
- An HTTP server on `http://localhost:8000` serving both HTML interfaces from the `contract-translator/` directory
- A Flask translation API on `http://localhost:5000` (`mcp/chatbot_api.py`) that handles generation requests
- Automatic browser tabs opening both `demo.html` and `sampler.html`

The launcher also checks and installs any missing dependencies before starting.

### sampler.html - Dataset Browser (`http://localhost:8000/sampler.html`)

`sampler.html` is a browser interface for exploring the FSM-SCG source dataset (`requirement_fsm_code.jsonl`). It lets you:

- Browse all 21,976 contract specifications filterable by contract type and complexity
- Read the full natural language requirement, FSM specification, and expert-written ground-truth Solidity code for any entry side by side
- Click "Open in Demo" on any entry to automatically pre-load that specification into `demo.html` for live generation

This is the recommended starting point for reviewers who want to understand the dataset before running the pipeline.

### demo.html - Live Translation Interface (`http://localhost:8000/demo.html`)

`demo.html` is the primary interactive interface for running the full seven-phase pipeline on any natural language specification. When a specification is submitted:

1. The frontend sends a POST request to the Flask API at `localhost:5000/translate`
2. The API instantiates `IBMAgenticContractTranslator` and runs the pipeline, streaming phase-by-phase status updates back to the browser via server-sent events
3. As each phase completes, the interface updates in real time showing: the extracted schema, the generated Solidity code, the security audit report, refinement status, and finally the five-dimensional quality scores
4. The completed contract, ABI, audit report, and quality evaluation are all downloadable from the interface

**What you can do in demo.html:**
- Paste any natural language contract description and run end-to-end generation
- Pre-load any FSM-SCG specification from `sampler.html` by clicking "Open in Demo"
- Inspect the `UniversalContractSchema` JSON extracted by Phase 1 to verify terminology accuracy
- View the security audit JSON with severity classification, exploit paths, and remediation recommendations
- Download the generated `.sol` file, ABI JSON, and quality evaluation report
- Toggle between the modular core implementation (`USE_MODULAR_CORE=true`, default) and the legacy monolithic implementation (`USE_MODULAR_CORE=false`) via environment variable

**Recommended reviewer workflow:**
1. Run `python launch_demo.py`
2. Go to `http://localhost:8000/sampler.html`, browse the dataset, and pick a contract type of interest
3. Click "Open in Demo" to send it to `demo.html`
4. Click "Translate" and watch the pipeline execute phase by phase
5. Inspect the quality scores, security audit, and generated Solidity

---

## Reproducing the Benchmark Experiments

The `Experiments/` directory contains the scripts used to produce the results in §5 of the paper. Each script corresponds to a specific experimental condition or validation study.

### Five-Condition Ablation Study (§5.7, Table 8)

The ablation runs five pipeline configurations on the same 300 contracts sampled with `seed=42`. The conditions are:

- **Condition A**: Schema-structured prompt, single LLM (no CrewAI agent orchestration)
- **Condition B**: Full pipeline, reinforcement disabled (`enable_reinforcement=False`)
- **Condition C**: Full pipeline, `max_refinement_iterations=1` (main paper configuration)
- **Condition D**: Full pipeline, `max_refinement_iterations=2`
- **Condition E**: Zero-shot baseline (single GPT-4o-mini call, no agents)


```bash
# Runs all five conditions (A-E) on the same 300-contract sample
python Experiments/experiment_1_ablation.py \
    --dataset requirement_fsm_code.jsonl \
    --n_samples 300 \
    --output_dir ./results/ablation \
    --conditions A B C D E \
    --seeds 42

# Ground-truth compilation rate analysis (compiler-version bias validation)
python Experiments/experiment_2_ground_truth_compilation.py \
    --dataset requirement_fsm_code.jsonl

# Temperature sensitivity study (reproducibility evidence)
python Experiments/experiment_3_temperature.py \
    --dataset requirement_fsm_code.jsonl \
    --n_samples 100 \
    --temperatures 0.0 0.3 0.7 1.0
```

Expected results (from Table 8):

| Condition | Score | Std | Compile% |
|---|---|---|---|
| A (schema-structured, single LLM) | 78.98 | 6.55 | 63.3% |
| E (zero-shot baseline) | -- | -- | 49.3% |
| B (no reinforcement) | 82.64 | 10.31 | 79.3% |
| C (max_iter=1) | 83.44 | 4.89 | 84.5% |
| D (max_iter=2) | 83.70 | 5.57 | 85.2% |

### External Slither Validation (§5.10, Table 11)

```bash
python Experiments/experiment_4_slither.py
```

This script runs all generated contracts through Slither before and after the reinforcement loop and computes the category agreement rate between the LLM auditor and the static analyzer. Requires `slither-analyzer` to be installed (included in `requirements.txt`). Expected output: 79.4% category agreement rate, 43.8% reduction in total Slither findings post-refinement.

### Verifying Batch Checkpoints

After running the main 9,000-contract evaluation (six parallel batches of 1,500), verify all batches completed successfully:

```bash
python test_checkpoints.py
```

This checks the `checkpoint.json` files in each batch output directory and prints the number of processed contracts per batch. All six batches should show 1,500 processed contracts each.

---

## Generating the Full 9,000-Contract Corpus

The main benchmark corpus was produced by running the pipeline on 9,000 contracts from the FSM-SCG dataset in six parallel batches of 1,500 using Condition C (`max_refinement_iterations=1`). To reproduce this:

**Step 1:** Download the FSM-SCG dataset:
```bash
# Download requirement_fsm_code.jsonl from the Kaggle dataset
# https://www.kaggle.com/datasets/neurips4242/smarteval-llm-generated-smart-contract-benchmark
```

**Step 2:** Sample 9,000 entries and split into batches. The paper used a random sample without replacement across six non-overlapping partitions.

**Step 3:** Run each batch in parallel (each takes ~45 hours at 109.96 seconds per contract):
```python
from applications.translator import IBMAgenticContractTranslator
import json, os, tempfile

translator = IBMAgenticContractTranslator(
    model="gpt-4o-mini",
    enable_reinforcement=True,
    max_refinement_iterations=1
)

# Process one batch
with open("requirement_fsm_code.jsonl") as f:
    contracts = [json.loads(line) for line in f]

batch = contracts[0:1500]  # Adjust slice per batch
for i, entry in enumerate(batch):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(entry["user_requirement"])
        tmp_path = f.name
    for phase_update in translator.translate_contract_streaming(
        input_path=tmp_path,
        output_dir="./output",
        require_audit_approval=False,
        generate_mcp_server=False,
    ):
        pass  # phase results accumulate; inspect phase_update["data"] per phase
    os.unlink(tmp_path)
```

The pre-generated corpus (all 9,000 contracts with scores, audits, and ABIs) is available on Kaggle so you do not need to re-run this unless you are extending the benchmark.

---

## Five-Dimensional Quality Rubric

The `Quality Evaluator Agent` in `agents.py` scores every generated contract against the five-dimensional rubric defined in §4.2 of the paper. Scores are deterministically recomputed from raw metric values in post-processing via:

```
Score = 0.25 * M1 + 0.15 * M2 + 0.15 * M3 + 0.35 * M4 + 0.10 * M5
```

| Metric | Weight | Measures |
|---|---|---|
| M1: Functional Completeness | 25% | Function name matching (exact +10 pts, semantic +7 pts) plus implementation quality across access control, events, and input validation |
| M2: Variable Fidelity | 15% | State variable naming consistency, correct Solidity types, and active use in logic |
| M3: State Machine Correctness | 15% | FSM state definition, transition implementation, and guard enforcement (Path A for explicit FSMs, Path B for stateless designs) |
| M4: Business Logic Fidelity | 35% | Obligation implementation, financial logic, temporal constraints, and conditional flows |
| M5: Code Quality | 10% | Absence of placeholders, NatSpec documentation, event quality, and code structure |

The evaluator never rounds scores to the nearest 5 and always shows its arithmetic in the `evidence` field of the output JSON, making all scores fully auditable.

---

## Implementation Modes

The launcher supports two implementation modes controlled by the `USE_MODULAR_CORE` environment variable:

**Modular Core (default, `USE_MODULAR_CORE=true`):** Uses the `contract-translator/core/` package with the full seven-agent pipeline described in the paper. This is the implementation used to produce all benchmark results and should be used for any replication work.

**Legacy Monolithic (`USE_MODULAR_CORE=false`):** Uses an older single-file implementation (`agentic_implementation.py`) retained for backward compatibility with early experiments. Results may differ from the paper's reported numbers.

```bash
# Use modular core (paper configuration)
python launch_demo.py

# Use legacy implementation
USE_MODULAR_CORE=false python launch_demo.py
```

---

## Optional: Testnet Deployment

The pipeline includes optional Phases 5 and 6 for deploying generated contracts to a local Ganache blockchain and generating a Model Context Protocol (MCP) server that exposes contract endpoints as AI-callable tools. This is disabled by default and not required to reproduce the benchmark results.

To enable deployment:

```bash
# Install Ganache
npm install -g ganache

# Start a local Ganache instance
ganache --port 8545

# Configure deployment in .env
RPC_URL=http://localhost:8545
PRIVATE_KEY=your_ganache_private_key_here
```

```python
translator = IBMAgenticContractTranslator(
    enable_deployment=True   # Enables Phases 5 and 6
)
```

When deployment is enabled, the pipeline additionally generates a `Contract_mcp_server.py` file that can be run as a standalone FastMCP server, making all contract functions accessible to any MCP-compatible AI assistant.

---

## Benchmark Results Summary

Full results are reported in §5 of the paper. Key figures:

| Metric | Value |
|---|---|
| Average composite score (generated) | 81.54 / 100 |
| Score standard deviation | 12.87 |
| Compilation success rate | 86.54% |
| Average score vs. ground truth | +8.29 points |
| Average processing time per contract | 109.96 seconds |
| Grade A (90+) | 7.3% |
| Grade B (80–89) | 66.4% |
| Grade C (70–79) | 23.1% |
| Grade D/F (below 70) | 3.2% |

The +8.29 composite gap between generated and ground-truth contracts reflects LLMs' literal specification-following behavior versus expert developers' architectural judgment and gas-efficiency trade-offs. Semantic debiasing narrows the gap to +6.61, confirming the residual difference is behavioral rather than artifactual.

---

## Dataset and Links

The complete benchmark dataset is hosted separately from this repository due to file size:

- **Kaggle (primary):** https://www.kaggle.com/datasets/neurips4242/smarteval-llm-generated-smart-contract-benchmark
- **DOI (persistent identifier):** https://doi.org/10.5281/zenodo.20046036

The dataset includes `requirement_fsm_code.jsonl` (21,976 FSM-SCG source entries), `smarteval_contracts.jsonl` (all 9,000 generated contracts with full annotations), and the `contracts/` directory with per-contract `.sol` and `quality_evaluation.json` files.

---

## License

The pipeline code in this repository is released under the MIT License.

The SmartEval generated contracts and evaluation artifacts are released under CC BY 4.0. The FSM-SCG source dataset (`requirement_fsm_code.jsonl`) is the work of Luo et al. (IJCAI 2025); see the dataset README for license details.

---

## Citation

Citation details to be added upon publication.
