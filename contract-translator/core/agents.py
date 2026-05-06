"""
CrewAI agent definitions for each translation phase.

Agents:
- Parser: extracts structured contract data
- Generator: writes Solidity code
- Auditor: finds security issues
- Refiner: fixes audit findings (used in the reinforcement loop)
- ABI: generates the contract ABI
- MCP: generates the MCP server
- Quality Evaluator: scores the output against the spec
"""

import os
from typing import Any, Dict, Optional

from crewai import LLM as CrewLLM
from crewai import Agent

DEFAULT_MAX_REFINEMENT_ITERATIONS = 2


def _convert_to_crew_llm(agentics_llm) -> CrewLLM:
    """Wraps an agentics LLM in the CrewAI LLM format."""
    model_name = getattr(agentics_llm, "model", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    return CrewLLM(model=model_name, api_key=api_key, temperature=0.7)


def create_agents(crew_llm: CrewLLM, enable_reinforcement: bool = True) -> dict:
    """
    Create all specialized agents for the translation pipeline.

    Args:
        crew_llm: CrewAI LLM instance
        enable_reinforcement: If True, includes a Refiner Agent for the reinforcement loop

    Returns:
        Dictionary with agent instances for each phase, including refiner_agent if enabled
    """

    # Parser Agent
    parser_agent = Agent(
        role="Contract Analysis Expert",
        goal=(
            "Extract every specific term, function name, variable name, state name, party role, "
            "financial amount, and obligation from the contract text exactly as written. "
            "Produce a fully-populated UniversalContractSchema JSON object with no generic placeholders "
            "and with obligations NEVER empty when functions or operations are described."
        ),
        backstory=(
            "You are an expert contract analyst who reads every sentence of a contract carefully. "
            "You extract EXACT terminology — if the contract says 'initializeLease' you write 'initializeLease', "
            "not 'initialize'. You map every described operation to an obligation with the correct authorized party. "
            "You never leave the obligations array empty when functions are described. "
            "You always produce valid JSON in the exact schema structure requested."
        ),
        llm=crew_llm,
        verbose=False,
        allow_delegation=False,
    )

    # Generator Agent
    generator_agent = Agent(
        role="Senior Solidity Smart Contract Engineer",
        goal=(
            "Implement the EXACT contract specification provided in the task. "
            "Read every MANDATORY requirement, every listed obligation, and every domain-specific rule, "
            "then implement each one completely with real on-chain logic. "
            "Produce a contract of 150-400 lines that fully and correctly satisfies the specification."
        ),
        backstory=(
            "You are a senior Solidity engineer with deep expertise in DeFi, tokens, governance, escrow, "
            "and marketplace contracts. You read every instruction in the task description carefully and "
            "implement every requirement with complete, production-quality code. "
            "You NEVER write empty functions, placeholder comments, or stub implementations. "
            "For every function you write, you ask: what real-world operation does this represent, "
            "what invariant must hold before and after, and what can go wrong? "
            "You enforce economic invariants (token supply conservation, escrow balance accounting), "
            "temporal logic (deadlines enforced with require(block.timestamp ...)), "
            "and access control (every sensitive function has a require()-backed modifier). "
            "You NEVER name a function parameter the same as a contract-level state variable "
            "(e.g. never use `owner` as a parameter if `address public owner` exists — use `tokenOwner` instead). "
            "You NEVER declare a public state variable with the same name as an interface function "
            "(e.g. never write `uint256 public totalSupply` when implementing IERC20 — use `uint256 private _totalSupply`). "
            "Your contracts are long, complete, and correct — a 300-line correct contract is "
            "far better to you than a 60-line stub."
        ),
        llm=crew_llm,
        verbose=False,
        allow_delegation=False,
    )

    # Auditor Agent
    auditor_agent = Agent(
        role="Blockchain Security Auditor",
        goal=(
            "Identify every exploitable vulnerability in the Solidity contract. "
            "For each issue, name the specific function affected and describe the exact exploit path. "
            "Provide severity_level, approved boolean, issues array, recommendations array with "
            "line-level fixes, vulnerability_count, and security_score in valid JSON."
        ),
        backstory=(
            "You are a blockchain security expert specializing in Solidity smart contract audits. "
            "You methodically check for reentrancy, access control gaps, integer overflow, "
            "timestamp manipulation, locked ether, unbounded loops, and input validation failures. "
            "Every issue you report names a specific function and explains how an attacker could exploit it. "
            "Every recommendation is a concrete code-level fix, not generic advice. "
            "You return only valid JSON — no markdown, no prose."
        ),
        llm=crew_llm,
        verbose=False,
        allow_delegation=False,
    )

    # ABI Agent
    abi_agent = Agent(
        role="Ethereum ABI Specialist",
        goal=(
            "Generate the complete, accurate ABI JSON array for the given Solidity contract. "
            "Include every public/external function with correct inputs, outputs, and stateMutability; "
            "every event with all parameters and indexed flags; and the constructor. "
            "Types must be exact Solidity types (uint256 not uint). Return ONLY the JSON array."
        ),
        backstory=(
            "You are an Ethereum developer who has spent years generating and validating ABI specifications. "
            "You know that 'uint' must be 'uint256', that view functions have no state mutations, "
            "that payable functions have stateMutability='payable', and that indexed event parameters "
            'must carry "indexed": true. You include every public/external function — never miss one. '
            "You preserve parameter names exactly. You return only the raw JSON array — no markdown fences, no prose."
        ),
        llm=crew_llm,
        verbose=False,
        allow_delegation=False,
    )

    # MCP Agent
    mcp_agent = Agent(
        role="MCP Server Developer",
        goal="Generate production-ready MCP server code for blockchain interaction",
        backstory=(
            "You are an expert Python developer specializing in Web3.py and MCP server generation. "
            "You create complete, self-contained MCP servers with proper error handling and "
            "transaction management for smart contract interaction."
        ),
        llm=crew_llm,
        verbose=False,
        allow_delegation=False,
    )

    # Quality Evaluator Agent
    quality_evaluator_agent = Agent(
        role="Smart Contract Quality Analyst",
        goal=(
            "Score the generated Solidity contract across five metrics (functional completeness, "
            "variable fidelity, state machine correctness, business logic fidelity, code quality). "
            "Produce precise integer scores based on exact point calculations — never round to the nearest 5. "
            "Return only valid JSON with metric_1 through metric_5 objects and a composite_score."
        ),
        backstory=(
            "You are an expert smart contract quality analyst who evaluates generated Solidity code "
            "against natural language specifications. You read the specification line by line, "
            "then inspect the code and assign scores based on exact evidence — counting matched functions, "
            "checking that variables are written and read, verifying state transitions are reachable, "
            "and confirming economic invariants are enforced. "
            "Your scores are precise (73 not 75) because you show the arithmetic. "
            "You return only valid JSON — no markdown, no prose — with the exact keys required."
        ),
        llm=crew_llm,
        verbose=False,
        allow_delegation=False,
    )

    agents = {
        "parser_agent": parser_agent,
        "generator_agent": generator_agent,
        "auditor_agent": auditor_agent,
        "abi_agent": abi_agent,
        "mcp_agent": mcp_agent,
        "quality_evaluator_agent": quality_evaluator_agent,
    }

    # Refiner Agent (used in the reinforcement loop)
    if enable_reinforcement:
        refiner_agent = Agent(
            role="Smart Contract Security Refiner",
            goal="Fix all identified security vulnerabilities in Solidity smart contracts",
            backstory=(
                "You are a Solidity security specialist who fixes smart contract vulnerabilities. "
                "Given a contract and a list of security issues from an audit, you rewrite the code "
                "to address every vulnerability while maintaining the original functionality. "
                "You follow the Checks-Effects-Interactions pattern, add reentrancy guards where needed, "
                "implement proper access control, validate all inputs with require(), "
                "and ensure no silent failures. You return ONLY the fixed Solidity code."
            ),
            llm=crew_llm,
            verbose=False,
            allow_delegation=False,
        )
        agents["refiner_agent"] = refiner_agent

    return agents


def should_refine(
    audit_report: Dict[str, Any],
    refinement_count: int,
    max_iterations: int = DEFAULT_MAX_REFINEMENT_ITERATIONS,
) -> bool:
    """Returns True if the contract should go through another refinement pass."""
    if refinement_count >= max_iterations:
        print(
            f"🔄 Refinement check: Max iterations reached ({refinement_count}/{max_iterations})"
        )
        return False

    severity = audit_report.get("severity_level", "unknown").lower()
    approved = audit_report.get("approved", False)

    print(
        f"🔄 Refinement check: severity={severity}, approved={approved}, iteration={refinement_count}/{max_iterations}"
    )

    # Refine if not approved and severity is medium or higher
    if not approved and severity in ["medium", "high", "critical"]:
        print(
            f"✓ Triggering refinement loop (severity={severity}, approved={approved})"
        )
        return True

    print(f"⏭️  Skipping refinement (severity={severity}, approved={approved})")
    return False


def create_refinement_task_description(
    solidity_code: str, audit_report: Dict[str, Any]
) -> str:
    """Build the task description for the Refiner Agent from audit findings."""
    issues = audit_report.get("issues", [])
    recommendations = audit_report.get("recommendations", [])
    severity = audit_report.get("severity_level", "unknown")

    issues_text = (
        "\n".join(f"  - {issue}" for issue in issues)
        if issues
        else "  - No specific issues listed"
    )
    recommendations_text = (
        "\n".join(f"  - {rec}" for rec in recommendations)
        if recommendations
        else "  - No specific recommendations"
    )

    return f"""Fix ALL security vulnerabilities in this Solidity smart contract.

CURRENT CONTRACT CODE:
```solidity
{solidity_code}
```

SECURITY AUDIT FINDINGS (Severity: {severity.upper()}):
{issues_text}

REQUIRED FIXES:
{recommendations_text}

CRITICAL REQUIREMENTS:
1. Fix EVERY issue listed above - do not skip any vulnerability
2. Follow the Checks-Effects-Interactions pattern for all external calls
3. Add reentrancy guards (nonReentrant modifier) where needed
4. Ensure ALL state changes happen BEFORE external calls
5. Add proper access control (onlyOwner, role-based) on sensitive functions
6. Validate ALL inputs with require() statements - no silent failures
7. Check for zero addresses on address parameters
8. Ensure arithmetic operations are safe (Solidity ^0.8.0 has built-in overflow protection)
9. Preserve the original contract functionality while fixing security issues

Return ONLY the complete, fixed Solidity code with ALL vulnerabilities addressed.
Do not include explanations - just the corrected code."""


__all__ = [
    "create_agents",
    "should_refine",
    "create_refinement_task_description",
    "_convert_to_crew_llm",
    "DEFAULT_MAX_REFINEMENT_ITERATIONS",
]
