"""
IBM Agentics Contract Translator - Main orchestrator class
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import PyPDF2
from crewai import Agent, Crew, Task

# agentics.LLM is only needed as a lightweight model-name holder before
# _convert_to_crew_llm wraps it.  Import it if available; fall back to a
# minimal stub so the module loads when the old IBM Agentics API is absent.
try:
    from agentics import LLM  # type: ignore
except ImportError:

    class LLM:  # type: ignore
        """Minimal stub replacing agentics.LLM when the package lacks it."""

        def __init__(self, model: str = "gpt-4o-mini", **kwargs):
            self.model = model

        def chat(self, messages: list, **kwargs) -> str:
            """Direct OpenAI fallback so legacy Program.forward() still works."""
            import openai

            client = openai.OpenAI()
            resp = client.chat.completions.create(model=self.model, messages=messages)
            return resp.choices[0].message.content


from .schemas import UniversalContractSchema

try:
    from .programs import (
        ABIGeneratorProgram,
        MCPServerGeneratorProgram,
        SecurityAuditorProgram,
        UniversalContractParserProgram,
        UniversalSolidityGeneratorProgram,
    )

    _programs_available = True
except ImportError:
    _programs_available = False
from .agents import (
    DEFAULT_MAX_REFINEMENT_ITERATIONS,
    _convert_to_crew_llm,
    create_agents,
    create_refinement_task_description,
    should_refine,
)
from .solidity_compiler import SolidityCompilationChecker
from .task_builders import (
    create_abi_generator_task_description,
    create_audit_task_description,
    create_mcp_task_description,
    create_parser_task_description,
    create_quality_evaluation_task_description,
    create_solidity_generator_task_description,
)


class IBMAgenticContractTranslator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        enable_reinforcement: bool = True,
        max_refinement_iterations: int = DEFAULT_MAX_REFINEMENT_ITERATIONS,
    ):
        """
        Initialize translator with Agentic pipeline using CrewAI Agents and Tasks

        Args:
            model: LLM model to use (default: gpt-4o-mini for OpenAI)
            enable_reinforcement: If True, enables automatic code refinement when audit fails
            max_refinement_iterations: Maximum number of refine-audit loops (default: 2)

        Note: IBM Agentics requires OPENAI_API_KEY in environment
        """

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY required in .env file. "
                "IBM Agentics uses OpenAI models by default."
            )

        # Store reinforcement settings
        self.enable_reinforcement = enable_reinforcement
        self.max_refinement_iterations = max_refinement_iterations

        self.llm = LLM(model=model)
        self.crew_llm = _convert_to_crew_llm(self.llm)

        print(f"✓ IBM Agentics LLM initialized with {model}")
        if enable_reinforcement:
            print(
                f"🔄 Reinforcement loop enabled (max {max_refinement_iterations} iterations)"
            )
        print("🤖 Initializing Agentic Pipeline with Agents...")

        # Keep legacy Program instances for backward compatibility
        if _programs_available:
            self.parser = UniversalContractParserProgram()
            self.generator = UniversalSolidityGeneratorProgram()
            self.auditor = SecurityAuditorProgram()
            self.abi_generator = ABIGeneratorProgram()
            self.mcp_generator = MCPServerGeneratorProgram()
        else:
            self.parser = self.generator = self.auditor = None
            self.abi_generator = self.mcp_generator = None

        # Create specialized agents for each phase
        self._create_agents()

        print("✓ All Agents initialized for agentic pipeline\n")

    def _create_agents(self):
        """Create specialized agents for each translation phase."""

        agents = create_agents(
            self.crew_llm, enable_reinforcement=self.enable_reinforcement
        )

        self.parser_agent = agents["parser_agent"]
        self.generator_agent = agents["generator_agent"]
        self.auditor_agent = agents["auditor_agent"]
        self.abi_agent = agents["abi_agent"]
        self.mcp_agent = agents["mcp_agent"]
        self.quality_evaluator_agent = agents["quality_evaluator_agent"]

        if self.enable_reinforcement and "refiner_agent" in agents:
            self.refiner_agent = agents["refiner_agent"]
        else:
            self.refiner_agent = None

    def _clean_code_block(self, code: str) -> str:
        """
        Strip markdown code fences and trailing text after the last closing brace.
        """
        # Remove markdown code fences (```solidity, ```, etc.)
        import re

        code = re.sub(r"^```\w*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```$", "", code, flags=re.MULTILINE)
        code = code.strip()

        # For Solidity code, remove any English text after the final closing brace
        if "contract " in code or "interface " in code or "library " in code:
            # Find the position of the last '}' at the beginning of a line or with minimal indentation
            lines = code.split("\n")
            last_brace_idx = -1

            for i in range(len(lines) - 1, -1, -1):
                stripped = lines[i].strip()
                if stripped == "}" or (
                    stripped.startswith("}")
                    and not stripped[1:].strip().startswith("//")
                ):
                    last_brace_idx = i
                    break

            if last_brace_idx != -1:
                code = "\n".join(lines[: last_brace_idx + 1])

        return code

    def _extract_json(self, text: str, expected_type):
        """Extract and parse JSON from text that may contain markdown or extra prose."""
        import json
        import re

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Remove any leading/trailing whitespace
        text = text.strip()

        # Try to find JSON object/array boundaries if text has extra content
        if not text.startswith(("{", "[")):
            # Look for first { or [
            json_start = min(
                (text.find("{") if text.find("{") != -1 else len(text)),
                (text.find("[") if text.find("[") != -1 else len(text)),
            )
            if json_start < len(text):
                text = text[json_start:]

        for attempt in range(3):
            try:
                parsed = json.loads(text)

                if hasattr(expected_type, "model_validate"):
                    return expected_type.model_validate(parsed)
                return parsed

            except json.JSONDecodeError as e:
                if attempt == 0:
                    print(f"   ⚠️  JSON parsing failed: {e}, attempting fixes...")
                    text = re.sub(r",(\s*[}\]])", r"\1", text)
                    text = re.sub(r'([}\]])\s*\n\s*(["{[])', r"\1,\2", text)
                    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
                    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
                    continue
                elif attempt == 1:
                    print(
                        f"   ⚠️  Still failing, trying to find valid JSON substring..."
                    )
                    brace_count = 0
                    valid_end = -1
                    for i, char in enumerate(text):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                valid_end = i + 1
                                break
                    if valid_end > 0:
                        text = text[:valid_end]
                        continue
                else:
                    print(f"   ❌ All JSON parsing attempts failed")
                    raise ValueError(
                        f"Could not parse JSON after multiple attempts. Error: {e}"
                    )

        raise ValueError(f"Could not parse JSON from text: {text[:200]}...")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        print(f"📄 Reading PDF: {pdf_path}")
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()

    def _run_agentic_pipeline(
        self, contract_text: str, generate_mcp_server: bool = True
    ) -> Dict:
        """
        Run the 6-phase translation pipeline.
        Returns dict with keys: schema, solidity, audit, abi, mcp_server (optional).
        """

        print("\n[AGENTIC PIPELINE] Using Agent-Task orchestration")

        results = {}

        # ===== PHASE 2: Contract Analysis (Parser Agent) =====
        print("\n[Phase 2/6] Contract Analysis (Parser Agent)")

        task_parse = Task(
            description=create_parser_task_description(contract_text),
            expected_output=(
                "Valid JSON object matching the UniversalContractSchema structure. "
                "MUST contain: contract_type string, parties array (each with name and role), "
                "financial_terms array (each with amount, currency, purpose), "
                "dates array (each with date_type), assets array, "
                "obligations array — NEVER EMPTY if functions are described (each with party and description), "
                "special_terms array, conditions dict with keys: function_names, variable_names, "
                "state_names, state_transitions, events, logic_conditions; "
                "and termination_conditions array. "
                "Use EXACT terminology from the contract text — no generic placeholders."
            ),
            agent=self.parser_agent,
        )

        crew_parse = Crew(agents=[self.parser_agent], tasks=[task_parse], verbose=False)

        parse_result = crew_parse.kickoff()

        try:
            parse_text = str(parse_result).strip()
            if "```json" in parse_text:
                parse_text = parse_text.split("```json")[1].split("```")[0].strip()

            parsed_json = json.loads(parse_text)

            # Validate financial_terms: drop entries with non-numeric amounts
            if "financial_terms" in parsed_json and parsed_json["financial_terms"]:
                cleaned_terms = []
                for term in parsed_json["financial_terms"]:
                    try:
                        if not isinstance(term.get("amount"), (int, float)):
                            continue
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(term.get("currency"), str):
                        term["currency"] = "ETH"
                    if not term.get("purpose"):
                        term["purpose"] = "Contract payment"
                    cleaned_terms.append(term)
                parsed_json["financial_terms"] = cleaned_terms

            if "parties" in parsed_json and parsed_json["parties"]:
                cleaned_parties = []
                for party in parsed_json["parties"]:
                    if party.get("name") and party.get("role"):
                        cleaned_parties.append(party)
                parsed_json["parties"] = cleaned_parties

            if not parsed_json.get("parties"):
                parsed_json["parties"] = [{"name": "Unknown Party", "role": "other"}]

            if not parsed_json.get("contract_type"):
                parsed_json["contract_type"] = "other"

            schema = UniversalContractSchema(**parsed_json)
            results["schema"] = schema
            _conds = schema.conditions if schema.conditions else {}
            print(
                f"✓ Parsed: {len(_conds.get('state_names', []))} states, {len(_conds.get('events', []))} events, {len(_conds.get('function_names', []))} functions, {len(schema.obligations)} obligations"
            )

        except Exception as e:
            print(f"⚠️ Error parsing schema: {e}")
            # Fallback to Program-based parsing
            schema = self.parser.forward(contract_text, self.llm)
            results["schema"] = schema

        # ===== PHASE 3: Solidity Generation (Generator Agent) =====
        print("\n[Phase 3/6] Code Generation (Generator Agent)")

        task_generate = Task(
            description=create_solidity_generator_task_description(schema),
            expected_output=(
                "Raw Solidity ^0.8.0 source code — NO markdown fences, NO explanation text. "
                "The contract MUST implement every obligation, function, state, and rule listed in the task description. "
                "Every MANDATORY section (TOKEN CONTRACT, GOVERNANCE, ESCROW, etc.) must be fully implemented if present. "
                "Target 150-400 lines. A correct 300-line contract is far better than a 60-line stub. "
                "Every function must contain complete logic: require() checks, state changes, value transfers, events."
            ),
            agent=self.generator_agent,
        )

        crew_generate = Crew(
            agents=[self.generator_agent], tasks=[task_generate], verbose=False
        )

        generate_result = crew_generate.kickoff()
        solidity_code = str(generate_result).strip()

        # Clean markdown code fences
        if "```solidity" in solidity_code:
            solidity_code = (
                solidity_code.split("```solidity")[1].split("```")[0].strip()
            )
        elif "```" in solidity_code:
            solidity_code = solidity_code.split("```")[1].split("```")[0].strip()

        results["solidity"] = solidity_code
        print(f"✓ Generated {len(solidity_code.splitlines())} lines")

        # ===== PHASE 4: Security Audit (Auditor Agent) =====
        print("\n[Phase 4/6] Security Analysis (Auditor Agent)")

        task_audit = Task(
            description=create_audit_task_description(solidity_code),
            expected_output=(
                "Valid JSON object with exactly these keys: "
                "severity_level (none/low/medium/high/critical), "
                "approved (boolean, true only if severity is none or low), "
                "issues (array — every item must name a specific function and describe the exploit path), "
                "recommendations (array of specific line-level fixes, not generic advice), "
                "vulnerability_count (integer matching issues array length), "
                "security_score (A=none, B=low, C=medium, D=high, F=critical). "
                "Return ONLY the JSON object — no markdown, no prose."
            ),
            agent=self.auditor_agent,
        )

        crew_audit = Crew(
            agents=[self.auditor_agent], tasks=[task_audit], verbose=False
        )

        audit_result = crew_audit.kickoff()
        audit_text = str(audit_result).strip()

        # Parse audit JSON
        if "```json" in audit_text:
            audit_text = audit_text.split("```json")[1].split("```")[0].strip()
        elif "```" in audit_text:
            audit_text = audit_text.split("```")[1].split("```")[0].strip()

        try:
            audit_report = json.loads(audit_text)
        except:
            audit_report = {
                "severity_level": "unknown",
                "approved": False,
                "issues": ["Failed to parse audit report"],
                "recommendations": [],
                "vulnerability_count": 0,
                "security_score": "N/A",
            }

        results["audit"] = audit_report
        severity = audit_report.get("severity_level", "unknown")
        score = audit_report.get("security_score", "N/A")
        print(f"✓ Audit Complete: Severity={severity}, Score={score}")

        # ===== REINFORCEMENT LOOP: Refine if needed =====
        refinement_count = 0
        while (
            self.enable_reinforcement
            and self.refiner_agent
            and should_refine(
                audit_report, refinement_count, self.max_refinement_iterations
            )
        ):
            refinement_count += 1
            print(
                f"\n[Phase 4.{refinement_count}] Reinforcement: Refining contract (iteration {refinement_count}/{self.max_refinement_iterations})"
            )

            # Create refinement task based on audit findings
            task_refine = Task(
                description=create_refinement_task_description(
                    solidity_code, audit_report
                ),
                expected_output=(
                    "Raw Solidity ^0.8.0 source code — NO markdown fences, NO explanation text. "
                    "Every vulnerability listed in the audit MUST be fixed. "
                    "Apply Checks-Effects-Interactions: state changes MUST come before external calls. "
                    "Add reentrancy guards (bool locked pattern) on all functions making external calls. "
                    "All inputs validated with require(). All original function names and business logic preserved."
                ),
                agent=self.refiner_agent,
            )

            crew_refine = Crew(
                agents=[self.refiner_agent], tasks=[task_refine], verbose=False
            )

            refine_result = crew_refine.kickoff()
            refined_code = str(refine_result).strip()

            # Strip markdown fences from refined code
            if "```solidity" in refined_code:
                refined_code = (
                    refined_code.split("```solidity")[1].split("```")[0].strip()
                )
            elif "```" in refined_code:
                refined_code = refined_code.split("```")[1].split("```")[0].strip()

            solidity_code = refined_code
            results["solidity"] = solidity_code
            print(f"✓ Refined contract: {len(solidity_code.splitlines())} lines")

            # Re-audit the refined code
            print(f"\n[Phase 4.{refinement_count}b] Re-auditing refined contract")

            task_re_audit = Task(
                description=create_audit_task_description(solidity_code),
                expected_output=(
                    "Valid JSON object with exactly these keys: "
                    "severity_level (none/low/medium/high/critical), "
                    "approved (boolean, true only if severity is none or low), "
                    "issues (array — every item must name a specific function and describe the exploit path), "
                    "recommendations (array of specific line-level fixes, not generic advice), "
                    "vulnerability_count (integer matching issues array length), "
                    "security_score (A=none, B=low, C=medium, D=high, F=critical). "
                    "Return ONLY the JSON object — no markdown, no prose."
                ),
                agent=self.auditor_agent,
            )

            crew_re_audit = Crew(
                agents=[self.auditor_agent], tasks=[task_re_audit], verbose=False
            )

            re_audit_result = crew_re_audit.kickoff()
            re_audit_text = str(re_audit_result).strip()

            # Parse re-audit JSON
            if "```json" in re_audit_text:
                re_audit_text = (
                    re_audit_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in re_audit_text:
                re_audit_text = re_audit_text.split("```")[1].split("```")[0].strip()

            try:
                audit_report = json.loads(re_audit_text)
            except:
                audit_report = {
                    "severity_level": "unknown",
                    "approved": False,
                    "issues": ["Failed to parse re-audit report"],
                    "recommendations": [],
                    "vulnerability_count": 0,
                    "security_score": "N/A",
                }

            results["audit"] = audit_report
            severity = audit_report.get("severity_level", "unknown")
            score = audit_report.get("security_score", "N/A")
            print(f"✓ Re-audit Complete: Severity={severity}, Score={score}")

        if refinement_count > 0:
            print(
                f"\n✓ Reinforcement loop completed after {refinement_count} iteration(s)"
            )

        # ===== PHASE 5: ABI Generation (ABI Agent) =====
        print("\n[Phase 5/6] Interface Generation (ABI Agent)")

        task_abi = Task(
            description=create_abi_generator_task_description(solidity_code),
            expected_output=(
                "Valid JSON array of ABI elements — NO markdown fences, NO explanation text. "
                "MUST include: constructor entry with correct input types and stateMutability, "
                "every public/external function with correct inputs, outputs, and stateMutability "
                "(pure/view/payable/nonpayable), every event with all parameters and indexed flags. "
                "Types must be exact Solidity types (uint256 not uint). "
                "Parameter names must be preserved exactly as in the source code."
            ),
            agent=self.abi_agent,
        )

        crew_abi = Crew(agents=[self.abi_agent], tasks=[task_abi], verbose=False)

        abi_result = crew_abi.kickoff()
        abi_text = str(abi_result).strip()

        # Parse ABI JSON
        if "```json" in abi_text:
            abi_text = abi_text.split("```json")[1].split("```")[0].strip()
        elif "```" in abi_text:
            abi_text = abi_text.split("```")[1].split("```")[0].strip()

        try:
            abi = json.loads(abi_text)
        except:
            abi = []

        results["abi"] = abi
        print(f"✓ Generated {len(abi)} ABI elements")

        # ===== PHASE 6: MCP Server Generation (Optional) =====
        if generate_mcp_server:
            print("\n[Phase 6/6] MCP Server Generation (MCP Agent)")

            contract_name = (
                "_".join([p.name.replace(" ", "_")[:10] for p in schema.parties[:2]])
                if schema.parties
                else "Contract"
            )
            contract_name = contract_name[:40]

            mcp_server_code = self.mcp_generator.forward(
                abi, schema, contract_name, self.llm
            )
            results["mcp_server"] = mcp_server_code
            print(f"✓ Generated MCP server ({len(mcp_server_code.splitlines())} lines)")
        else:
            print("\n[Phase 6/6] MCP Server Generation - SKIPPED")

        return results

    def translate_contract_streaming(
        self,
        input_path: str,
        output_dir: str = "./output",
        require_audit_approval: bool = True,
        generate_mcp_server: bool = True,
        use_agentic_pipeline: bool = True,
    ):
        """
        Streaming version that yields phase updates as they complete.
        Yields dict with {phase: int, status: str, data: dict}

        Args:
            input_path: Path to contract file (PDF or text)
            output_dir: Output directory for generated files
            require_audit_approval: Whether to require user approval on security issues
            generate_mcp_server: Whether to generate MCP server code
            use_agentic_pipeline: If True, use Agent/Task/Crew approach. If False, use legacy Program approach.
        """

        print("\n" + "=" * 70)
        if use_agentic_pipeline:
            print("IBM AGENTICS CONTRACT TRANSLATOR (STREAMING - Agent/Task Pipeline)")
        else:
            print(
                "IBM AGENTICS CONTRACT TRANSLATOR (STREAMING - Legacy Program Pipeline)"
            )
        print("=" * 70)

        results = {}

        # Phase 1: Document Processing
        print("\n[Phase 1/6] Document Processing")
        if input_path.endswith(".pdf"):
            contract_text = self.extract_text_from_pdf(input_path)
            source = "PDF"
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                contract_text = f.read()
            source = "text file"
        print(f"✓ Extracted {len(contract_text)} characters from {source}")

        yield {
            "phase": 1,
            "status": "complete",
            "data": {
                "title": "Document Processing",
                "message": f"Extracted {len(contract_text)} characters from {source}",
            },
        }

        # Execute phases 2-6 based on mode
        if use_agentic_pipeline:
            # NEW: Use Agent/Task/Crew orchestration with streaming yields

            # Phase 2: Contract Analysis (Parser Agent)
            print("\n[Phase 2/7] Contract Analysis (Parser Agent)")
            task_desc = create_parser_task_description(contract_text)
            task = Task(
                description=task_desc,
                expected_output=(
                    "Valid JSON object matching the UniversalContractSchema structure. "
                    "MUST contain: contract_type, parties (name+role), financial_terms (amount+currency+purpose), "
                    "dates (date_type), assets, obligations (NEVER EMPTY if functions described — party+description), "
                    "special_terms, conditions dict (function_names, variable_names, state_names, state_transitions, "
                    "events, logic_conditions), termination_conditions. "
                    "Use EXACT terminology from the contract — no generic placeholders."
                ),
                agent=self.parser_agent,
            )
            crew = Crew(agents=[self.parser_agent], tasks=[task], verbose=False)

            try:
                result_raw = crew.kickoff()
                result_text = (
                    str(result_raw.raw)
                    if hasattr(result_raw, "raw")
                    else str(result_raw)
                )
                schema = self._extract_json(result_text, UniversalContractSchema)
                results["schema"] = schema
                _conds = schema.conditions if schema.conditions else {}
                print(
                    f"✓ Parsed: {len(_conds.get('state_names', []))} states, {len(_conds.get('events', []))} events, {len(_conds.get('function_names', []))} functions, {len(schema.obligations)} obligations"
                )
            except Exception as e:
                print(f"   ⚠️  Agent approach failed, using fallback Program: {e}")
                schema = self.parser.forward(contract_text, self.llm)
                results["schema"] = schema

            # Convert schema to dict for JSON serialization
            try:
                schema_dict = (
                    schema.model_dump()
                    if hasattr(schema, "model_dump")
                    else schema.__dict__
                )
            except:
                schema_dict = {
                    "contract_type": str(schema.contract_type),
                    "parties": (
                        [{"name": p.name, "role": p.role} for p in schema.parties]
                        if schema.parties
                        else []
                    ),
                    "financial_terms": (
                        [
                            {
                                "amount": t.amount,
                                "currency": t.currency,
                                "purpose": t.purpose,
                            }
                            for t in schema.financial_terms
                        ]
                        if schema.financial_terms
                        else []
                    ),
                }

            # Extract counts and details for informative message
            num_parties = (
                len(schema.parties)
                if hasattr(schema, "parties") and schema.parties
                else 0
            )
            num_terms = (
                len(schema.financial_terms)
                if hasattr(schema, "financial_terms") and schema.financial_terms
                else 0
            )
            num_obligations = (
                len(schema.obligations)
                if hasattr(schema, "obligations") and schema.obligations
                else 0
            )
            num_assets = (
                len(schema.assets) if hasattr(schema, "assets") and schema.assets else 0
            )
            num_dates = (
                len(schema.dates) if hasattr(schema, "dates") and schema.dates else 0
            )

            # Also check the schema_dict in case attributes aren't populated
            if num_parties == 0 and isinstance(schema_dict, dict):
                num_parties = len(schema_dict.get("parties", []))
            if num_terms == 0 and isinstance(schema_dict, dict):
                num_terms = len(schema_dict.get("financial_terms", []))
            if num_obligations == 0 and isinstance(schema_dict, dict):
                num_obligations = len(schema_dict.get("obligations", []))
            if num_assets == 0 and isinstance(schema_dict, dict):
                num_assets = len(schema_dict.get("assets", []))
            if num_dates == 0 and isinstance(schema_dict, dict):
                num_dates = len(schema_dict.get("dates", []))

            # Check conditions dict for function/variable/state names
            conditions = (
                schema.conditions
                if hasattr(schema, "conditions")
                else schema_dict.get("conditions", {})
            )
            num_functions = (
                len(conditions.get("function_names", []))
                if isinstance(conditions, dict)
                else 0
            )
            num_variables = (
                len(conditions.get("variable_names", []))
                if isinstance(conditions, dict)
                else 0
            )
            num_states = (
                len(conditions.get("state_names", []))
                if isinstance(conditions, dict)
                else 0
            )
            num_events = (
                len(conditions.get("events", [])) if isinstance(conditions, dict) else 0
            )

            # Build rich, informative message with actual extracted data
            message_parts = []
            if num_parties > 0:
                message_parts.append(f"{num_parties} parties")
            if num_terms > 0:
                message_parts.append(f"{num_terms} financial terms")
            if num_obligations > 0:
                message_parts.append(f"{num_obligations} obligations")
            if num_assets > 0:
                message_parts.append(f"{num_assets} assets")
            if num_dates > 0:
                message_parts.append(f"{num_dates} dates")
            if num_functions > 0:
                message_parts.append(f"{num_functions} functions")
            if num_variables > 0:
                message_parts.append(f"{num_variables} variables")
            if num_states > 0:
                message_parts.append(f"{num_states} states")
            if num_events > 0:
                message_parts.append(f"{num_events} events")

            if message_parts:
                message = f"Extracted: {', '.join(message_parts)}"
            else:
                # Really minimal fallback
                contract_type_name = (
                    schema.contract_type.replace("_", " ").title()
                    if hasattr(schema, "contract_type")
                    else schema_dict.get("contract_type", "Unknown")
                    .replace("_", " ")
                    .title()
                )
                message = f"Analyzed {contract_type_name} contract structure"

            yield {
                "phase": 2,
                "status": "complete",
                "data": {
                    "title": "Contract Analysis",
                    "message": message,
                    "contract_type": schema.contract_type,
                    "parties": (
                        schema_dict.get("parties", [])
                        if isinstance(schema_dict, dict)
                        else []
                    ),
                    "financial_terms": (
                        schema_dict.get("financial_terms", [])
                        if isinstance(schema_dict, dict)
                        else []
                    ),
                    "schema": schema_dict,
                },
            }

            # Phase 3: Solidity Generation (Generator Agent)
            print("\n[Phase 3/7] Code Generation (Generator Agent)")
            task_desc = create_solidity_generator_task_description(schema)
            task = Task(
                description=task_desc,
                expected_output=(
                    "Raw Solidity ^0.8.0 source code — NO markdown fences, NO explanation text. "
                    "The contract MUST implement every obligation, function, state, and rule listed in the task description. "
                    "Every MANDATORY section (TOKEN CONTRACT, GOVERNANCE, ESCROW, etc.) must be fully implemented if present. "
                    "Target 150-400 lines. A correct 300-line contract is far better than a 60-line stub. "
                    "Every function must contain complete logic: require() checks, state changes, value transfers, events."
                ),
                agent=self.generator_agent,
            )
            crew = Crew(agents=[self.generator_agent], tasks=[task], verbose=False)

            try:
                result_raw = crew.kickoff()
                solidity_code = (
                    str(result_raw.raw)
                    if hasattr(result_raw, "raw")
                    else str(result_raw)
                )
                solidity_code = self._clean_code_block(solidity_code)
                results["solidity"] = solidity_code
                print(f"✓ Generated {len(solidity_code.splitlines())} lines")
            except Exception as e:
                print(f"   ⚠️  Agent approach failed, using fallback Program: {e}")
                solidity_code = self.generator.forward(schema, self.llm)
                results["solidity"] = solidity_code

            yield {
                "phase": 3,
                "status": "complete",
                "data": {
                    "title": "Code Generation",
                    "message": f"Generated {len(solidity_code.splitlines())} lines of Solidity",
                    "solidity": solidity_code,
                },
            }

            # Phase 4: Security Audit (Auditor Agent)
            print("\n[Phase 4/7] Security Analysis (Auditor Agent)")
            task_desc = create_audit_task_description(solidity_code)
            task = Task(
                description=task_desc,
                expected_output=(
                    "Valid JSON object with exactly these keys: "
                    "severity_level (none/low/medium/high/critical), "
                    "approved (boolean, true only if severity is none or low), "
                    "issues (array — each item names a specific function and describes the exploit path), "
                    "recommendations (array of specific line-level fixes), "
                    "vulnerability_count (integer matching issues array length), "
                    "security_score (A/B/C/D/F). Return ONLY the JSON — no markdown, no prose."
                ),
                agent=self.auditor_agent,
            )
            crew = Crew(agents=[self.auditor_agent], tasks=[task], verbose=False)

            try:
                result_raw = crew.kickoff()
                result_text = (
                    str(result_raw.raw)
                    if hasattr(result_raw, "raw")
                    else str(result_raw)
                )
                audit_report = self._extract_json(result_text, dict)
                results["audit"] = audit_report
            except Exception as e:
                print(f"   ⚠️  Agent approach failed, using fallback Program: {e}")
                audit_report = self.auditor.forward(solidity_code, self.llm)
                results["audit"] = audit_report

            severity = audit_report.get("severity_level", "unknown")
            score = audit_report.get("security_score", "N/A")
            issues = audit_report.get("issues", [])
            print(f"✓ Audit Complete: Severity={severity}, Score={score}")

            # ===== REINFORCEMENT LOOP: Refine if needed =====
            refinement_count = 0
            print(
                f"🔍 Reinforcement loop check: enable={self.enable_reinforcement}, has_refiner={self.refiner_agent is not None}, max_iter={self.max_refinement_iterations}"
            )
            while (
                self.enable_reinforcement
                and self.refiner_agent
                and should_refine(
                    audit_report, refinement_count, self.max_refinement_iterations
                )
            ):
                refinement_count += 1
                print(
                    f"\n[Phase 4.{refinement_count}] Reinforcement: Refining contract (iteration {refinement_count}/{self.max_refinement_iterations})"
                )

                yield {
                    "phase": 4,
                    "status": "refining",
                    "data": {
                        "title": "Security Refinement",
                        "message": f"Refining contract (iteration {refinement_count}/{self.max_refinement_iterations})",
                        "iteration": refinement_count,
                        "max_iterations": self.max_refinement_iterations,
                    },
                }

                # Create refinement task based on audit findings
                task_refine_desc = create_refinement_task_description(
                    solidity_code, audit_report
                )
                task_refine = Task(
                    description=task_refine_desc,
                    expected_output=(
                        "Raw Solidity ^0.8.0 source code — NO markdown fences, NO explanation text. "
                        "Every vulnerability listed in the audit MUST be fixed. "
                        "Apply Checks-Effects-Interactions: state changes BEFORE external calls. "
                        "Add reentrancy guards on all functions making external calls. "
                        "All inputs validated with require(). All original function names and business logic preserved."
                    ),
                    agent=self.refiner_agent,
                )
                crew_refine = Crew(
                    agents=[self.refiner_agent], tasks=[task_refine], verbose=False
                )

                try:
                    result_raw = crew_refine.kickoff()
                    refined_code = (
                        str(result_raw.raw)
                        if hasattr(result_raw, "raw")
                        else str(result_raw)
                    )
                    refined_code = self._clean_code_block(refined_code)
                    solidity_code = refined_code
                    results["solidity"] = solidity_code
                    print(
                        f"✓ Refined contract: {len(solidity_code.splitlines())} lines"
                    )
                except Exception as e:
                    print(f"   ⚠️  Refinement failed: {e}")
                    break

                # Re-audit the refined code
                print(f"\n[Phase 4.{refinement_count}b] Re-auditing refined contract")
                task_re_audit_desc = create_audit_task_description(solidity_code)
                task_re_audit = Task(
                    description=task_re_audit_desc,
                    expected_output=(
                        "Valid JSON object with exactly these keys: "
                        "severity_level (none/low/medium/high/critical), "
                        "approved (boolean, true only if severity is none or low), "
                        "issues (array — each item names a specific function and describes the exploit path), "
                        "recommendations (array of specific line-level fixes), "
                        "vulnerability_count (integer matching issues array length), "
                        "security_score (A/B/C/D/F). Return ONLY the JSON — no markdown, no prose."
                    ),
                    agent=self.auditor_agent,
                )
                crew_re_audit = Crew(
                    agents=[self.auditor_agent], tasks=[task_re_audit], verbose=False
                )

                try:
                    result_raw = crew_re_audit.kickoff()
                    result_text = (
                        str(result_raw.raw)
                        if hasattr(result_raw, "raw")
                        else str(result_raw)
                    )
                    audit_report = self._extract_json(result_text, dict)
                    results["audit"] = audit_report
                except Exception as e:
                    print(f"   ⚠️  Re-audit failed: {e}")
                    break

                severity = audit_report.get("severity_level", "unknown")
                score = audit_report.get("security_score", "N/A")
                issues = audit_report.get("issues", [])
                print(f"✓ Re-audit Complete: Severity={severity}, Score={score}")

            if refinement_count > 0:
                print(
                    f"\n✓ Reinforcement loop completed after {refinement_count} iteration(s)"
                )

            yield {
                "phase": 4,
                "status": "needs_approval",
                "data": {
                    "title": "Security Audit",
                    "message": f"Severity: {severity.upper()}, Score: {score}"
                    + (
                        f" (after {refinement_count} refinement(s))"
                        if refinement_count > 0
                        else ""
                    ),
                    "severity_level": severity,
                    "security_score": score,
                    "issues": issues,
                    "vulnerability_count": audit_report.get("vulnerability_count", 0),
                    "recommendations": audit_report.get("recommendations", []),
                    "refinement_iterations": refinement_count,
                    "solidity": solidity_code,  # Include updated code if refined
                },
            }

            # Phase 5: ABI Generation (ABI Agent)
            print("\n[Phase 5/7] Interface Generation (ABI Agent)")
            task_desc = create_abi_generator_task_description(solidity_code)
            task = Task(
                description=task_desc,
                expected_output=(
                    "Valid JSON array of ABI elements — NO markdown fences, NO explanation text. "
                    "MUST include: constructor with correct input types and stateMutability, "
                    "every public/external function with correct inputs, outputs, and stateMutability, "
                    "every event with all parameters and indexed flags correct. "
                    "Types must be exact Solidity types (uint256 not uint). Parameter names preserved exactly."
                ),
                agent=self.abi_agent,
            )
            crew = Crew(agents=[self.abi_agent], tasks=[task], verbose=False)

            try:
                result_raw = crew.kickoff()
                result_text = (
                    str(result_raw.raw)
                    if hasattr(result_raw, "raw")
                    else str(result_raw)
                )
                abi = self._extract_json(result_text, list)
                results["abi"] = abi
                print(f"✓ Generated {len(abi)} ABI elements")
            except Exception as e:
                print(f"   ⚠️  Agent approach failed, using fallback Program: {e}")
                abi = self.abi_generator.forward(solidity_code, self.llm)
                results["abi"] = abi

            yield {
                "phase": 5,
                "status": "complete",
                "data": {
                    "title": "ABI Generation",
                    "message": f"Generated {len(abi)} ABI elements",
                    "abi": abi,
                },
            }

            # Phase 6: MCP Server Generation (MCP Agent)
            if generate_mcp_server:
                print("\n[Phase 6/7] MCP Server Generation (MCP Agent)")
                contract_name = (
                    "_".join(
                        [p.name.replace(" ", "_")[:10] for p in schema.parties[:2]]
                    )
                    if schema.parties
                    else "Contract"
                )
                contract_name = contract_name[:40]

                task_desc = create_mcp_task_description(abi, schema, contract_name)
                task = Task(
                    description=task_desc,
                    expected_output=(
                        "Complete, self-contained Python file — NO markdown fences, NO explanation text. "
                        "MUST use FastMCP, load .env and ABI JSON from same directory as script, "
                        "include @mcp.tool() decorated function for EVERY ABI function with correct "
                        'payable/nonpayable/view handling, try/except on each tool returning {"error": str(e)}, '
                        "and end with: if __name__ == '__main__': mcp.run()"
                    ),
                    agent=self.mcp_agent,
                )
                crew = Crew(agents=[self.mcp_agent], tasks=[task], verbose=False)

                try:
                    result_raw = crew.kickoff()
                    result_text = (
                        str(result_raw.raw)
                        if hasattr(result_raw, "raw")
                        else str(result_raw)
                    )
                    mcp_server_code = self._clean_code_block(result_text)
                    results["mcp_server"] = mcp_server_code
                except Exception as e:
                    print(f"   ⚠️  Agent approach failed, using fallback Program: {e}")
                    mcp_server_code = self.mcp_generator.forward(
                        abi, schema, contract_name, self.llm
                    )
                    results["mcp_server"] = mcp_server_code

                print(
                    f"✓ Generated MCP server ({len(mcp_server_code.splitlines())} lines)"
                )
            else:
                print("\n[Phase 6/7] MCP Server Generation - SKIPPED")

            yield {
                "phase": 6,
                "status": "complete",
                "data": {
                    "title": "MCP Server Generation",
                    "message": (
                        "Generated MCP server" if generate_mcp_server else "Skipped"
                    ),
                    "mcp_server": results.get("mcp_server", ""),
                },
            }

            # Phase 7: Quality Evaluation (Quality Evaluator Agent)
            print("\n[Phase 7/7] Quality Evaluation (Quality Evaluator Agent)")
            contract_name = (
                "_".join([p.name.replace(" ", "_")[:10] for p in schema.parties[:2]])
                if schema.parties
                else "Contract"
            )
            contract_name = contract_name[:40]

            task_desc = create_quality_evaluation_task_description(
                solidity_code, schema, contract_name
            )
            task = Task(
                description=task_desc,
                expected_output=(
                    "Valid JSON object with exactly these top-level keys: "
                    "metric_1_functional_completeness (object with score 0-100 and evidence), "
                    "metric_2_variable_fidelity (object with score 0-100 and evidence), "
                    "metric_3_state_machine (object with score 0-100, state_machine_required bool, scoring_path string, and evidence), "
                    "metric_4_business_logic (object with score 0-100 and evidence), "
                    "metric_5_code_quality (object with score 0-100 and evidence), "
                    "composite_score (object with final_score and grade). "
                    "For metric_3: if spec has no explicit state_names, set state_machine_required=false and use Path B scoring. "
                    "Scores must be precise integers (e.g. 73 not 75). Return ONLY the JSON — no markdown."
                ),
                agent=self.quality_evaluator_agent,
            )
            crew = Crew(
                agents=[self.quality_evaluator_agent], tasks=[task], verbose=False
            )

            try:
                result_raw = crew.kickoff()
                result_text = (
                    str(result_raw.raw)
                    if hasattr(result_raw, "raw")
                    else str(result_raw)
                )
                quality_evaluation = self._extract_json(result_text, dict)

                # Recalculate final_score to ensure correctness
                m1 = quality_evaluation.get("metric_1_functional_completeness", {}).get(
                    "score", 0
                )
                m2 = quality_evaluation.get("metric_2_variable_fidelity", {}).get(
                    "score", 0
                )
                m3 = quality_evaluation.get("metric_3_state_machine", {}).get(
                    "score", 0
                )
                m4 = quality_evaluation.get("metric_4_business_logic", {}).get(
                    "score", 0
                )
                m5 = quality_evaluation.get("metric_5_code_quality", {}).get("score", 0)

                # Calculate weighted score: (M1 * 0.25) + (M2 * 0.15) + (M3 * 0.15) + (M4 * 0.35) + (M5 * 0.10)
                calculated_score = (
                    (m1 * 0.25) + (m2 * 0.15) + (m3 * 0.15) + (m4 * 0.35) + (m5 * 0.10)
                )

                # Determine grade based on calculated score
                if calculated_score >= 90:
                    grade = "A"
                elif calculated_score >= 80:
                    grade = "B"
                elif calculated_score >= 70:
                    grade = "C"
                elif calculated_score >= 60:
                    grade = "D"
                else:
                    grade = "F"

                # Override composite_score with calculated values
                quality_evaluation["composite_score"] = {
                    "functional_completeness_weighted": round(m1 * 0.25, 2),
                    "variable_fidelity_weighted": round(m2 * 0.15, 2),
                    "state_machine_weighted": round(m3 * 0.15, 2),
                    "business_logic_weighted": round(m4 * 0.35, 2),
                    "code_quality_weighted": round(m5 * 0.10, 2),
                    "final_score": round(calculated_score, 1),
                    "grade": grade,
                }

                results["quality_evaluation"] = quality_evaluation

                # Print summary
                final_score = calculated_score
                print(
                    f"✓ Quality Evaluation Complete: Score={final_score:.1f}/100, Grade={grade}"
                )

                # Print metric breakdown
                print(f"   📊 Functional Completeness: {m1}/100")
                print(f"   📊 Variable Fidelity: {m2}/100")
                print(f"   📊 State Machine Correctness: {m3}/100")
                print(f"   📊 Business Logic Fidelity: {m4}/100")
                print(f"   📊 Code Quality: {m5}/100")

                # Check Solidity compilation
                print(f"\n   🔍 Checking Solidity compilation...")
                compiler_checker = SolidityCompilationChecker()
                compilation_result = compiler_checker.check_compilation(solidity_code)
                quality_evaluation["compilation_check"] = compilation_result

                if compilation_result["compiles"] is None:
                    print(f"   ⚠️  {compilation_result['error_message']}")
                elif compilation_result["compiles"]:
                    warnings_text = (
                        f" ({len(compilation_result['warnings'])} warnings)"
                        if compilation_result["warnings"]
                        else ""
                    )
                    print(f"   ✅ Contract compiles successfully{warnings_text}")
                else:
                    error_preview = (
                        compilation_result["error_message"][:100] + "..."
                        if len(compilation_result["error_message"]) > 100
                        else compilation_result["error_message"]
                    )
                    print(f"   ❌ Compilation failed: {error_preview}")

            except Exception as e:
                print(f"   ⚠️  Quality evaluation failed: {e}")
                import traceback

                traceback.print_exc()
                quality_evaluation = {
                    "error": str(e),
                    "composite_score": {"final_score": 0, "grade": "F"},
                    "compilation_check": {
                        "compiles": None,
                        "error_message": "Quality evaluation failed",
                    },
                }
                results["quality_evaluation"] = quality_evaluation

            yield {
                "phase": 7,
                "status": "complete",
                "data": {
                    "title": "Quality Evaluation",
                    "message": f"Score: {quality_evaluation.get('composite_score', {}).get('final_score', 0):.1f}/100, Grade: {quality_evaluation.get('composite_score', {}).get('grade', 'N/A')}",
                    "quality_evaluation": quality_evaluation,
                },
            }

        else:
            # Legacy: Use Program.forward() calls with streaming yields

            # Phase 2: Contract Analysis
            print("\n[Phase 2/6] Contract Analysis (Parser Program)")
            schema = self.parser.forward(contract_text, self.llm)
            results["schema"] = schema
            _conds = schema.conditions if schema.conditions else {}
            print(
                f"✓ Parsed: {len(_conds.get('state_names', []))} states, {len(_conds.get('events', []))} events, {len(_conds.get('function_names', []))} functions, {len(schema.obligations)} obligations"
            )

            # Convert schema to dict for JSON serialization
            try:
                schema_dict = (
                    schema.model_dump()
                    if hasattr(schema, "model_dump")
                    else schema.__dict__
                )
            except Exception as e:
                print(f"   ⚠️  Error converting schema to dict: {e}")
                schema_dict = {
                    "contract_type": str(schema.contract_type),
                    "parties": (
                        [{"name": p.name, "role": p.role} for p in schema.parties]
                        if schema.parties
                        else []
                    ),
                    "financial_terms": (
                        [
                            {
                                "amount": t.amount,
                                "currency": t.currency,
                                "purpose": t.purpose,
                            }
                            for t in schema.financial_terms
                        ]
                        if schema.financial_terms
                        else []
                    ),
                }

            yield {
                "phase": 2,
                "status": "complete",
                "data": {
                    "title": "Contract Analysis",
                    "message": f'Parsed: {len(_conds.get("state_names", []))} states, {len(_conds.get("events", []))} events, {len(_conds.get("function_names", []))} functions, {len(schema.obligations)} obligations',
                    "contract_type": schema.contract_type,
                    "parties": (
                        schema_dict.get("parties", [])
                        if isinstance(schema_dict, dict)
                        else []
                    ),
                    "financial_terms": (
                        schema_dict.get("financial_terms", [])
                        if isinstance(schema_dict, dict)
                        else []
                    ),
                    "schema": schema_dict,
                },
            }

            # Phase 3: Solidity Generation
            print("\n[Phase 3/6] Code Generation (Generator Program)")
            solidity_code = self.generator.forward(schema, self.llm)
            results["solidity"] = solidity_code
            print(f"✓ Generated {len(solidity_code.splitlines())} lines")

            yield {
                "phase": 3,
                "status": "complete",
                "data": {
                    "title": "Code Generation",
                    "message": f"Generated {len(solidity_code.splitlines())} lines of Solidity",
                    "solidity": solidity_code,
                },
            }

            # Phase 4: Security Audit
            print("\n[Phase 4/6] Security Analysis (Auditor Program)")
            audit_report = self.auditor.forward(solidity_code, self.llm)
            results["audit"] = audit_report
            severity = audit_report.get("severity_level", "unknown")
            score = audit_report.get("security_score", "N/A")
            issues = audit_report.get("issues", [])
            print(f"✓ Audit Complete: Severity={severity}, Score={score}")

            yield {
                "phase": 4,
                "status": "needs_approval",
                "data": {
                    "title": "Security Audit",
                    "message": f"Severity: {severity.upper()}, Score: {score}",
                    "severity_level": severity,
                    "security_score": score,
                    "issues": issues,
                    "vulnerability_count": audit_report.get("vulnerability_count", 0),
                    "recommendations": audit_report.get("recommendations", []),
                },
            }

            # Phase 5: ABI Generation
            print("\n[Phase 5/6] Interface Generation (ABI Program)")
            abi = self.abi_generator.forward(solidity_code, self.llm)
            results["abi"] = abi
            print(f"✓ Generated {len(abi)} ABI elements")

            yield {
                "phase": 5,
                "status": "complete",
                "data": {
                    "title": "ABI Generation",
                    "message": f"Generated {len(abi)} ABI elements",
                    "abi": abi,
                },
            }

            # Phase 6: MCP Server Generation
            if generate_mcp_server:
                print("\n[Phase 6/6] MCP Server Generation (MCP Generator Program)")
                contract_name = (
                    "_".join(
                        [p.name.replace(" ", "_")[:10] for p in schema.parties[:2]]
                    )
                    if schema.parties
                    else "Contract"
                )
                contract_name = contract_name[:40]
                mcp_server_code = self.mcp_generator.forward(
                    abi, schema, contract_name, self.llm
                )
                results["mcp_server"] = mcp_server_code
                print(
                    f"✓ Generated MCP server ({len(mcp_server_code.splitlines())} lines)"
                )
            else:
                print("\n[Phase 6/7] MCP Server Generation - SKIPPED")

            yield {
                "phase": 6,
                "status": "complete",
                "data": {
                    "title": "MCP Server Generation",
                    "message": (
                        "Generated MCP server" if generate_mcp_server else "Skipped"
                    ),
                    "mcp_server": results.get("mcp_server", ""),
                },
            }

            # Phase 7: Quality Evaluation
            print("\n[Phase 7/7] Quality Evaluation (Quality Evaluator Agent)")
            contract_name = (
                "_".join([p.name.replace(" ", "_")[:10] for p in schema.parties[:2]])
                if schema.parties
                else "Contract"
            )
            contract_name = contract_name[:40]

            task_desc = create_quality_evaluation_task_description(
                solidity_code, schema, contract_name
            )
            task = Task(
                description=task_desc,
                expected_output=(
                    "Valid JSON object with exactly these top-level keys: "
                    "metric_1_functional_completeness (object with score 0-100 and evidence), "
                    "metric_2_variable_fidelity (object with score 0-100 and evidence), "
                    "metric_3_state_machine (object with score 0-100, state_machine_required bool, scoring_path string, and evidence), "
                    "metric_4_business_logic (object with score 0-100 and evidence), "
                    "metric_5_code_quality (object with score 0-100 and evidence), "
                    "composite_score (object with final_score and grade). "
                    "For metric_3: if spec has no explicit state_names, set state_machine_required=false and use Path B scoring. "
                    "Scores must be precise integers (e.g. 73 not 75). Return ONLY the JSON — no markdown."
                ),
                agent=self.quality_evaluator_agent,
            )
            crew = Crew(
                agents=[self.quality_evaluator_agent], tasks=[task], verbose=False
            )

            try:
                result_raw = crew.kickoff()
                result_text = (
                    str(result_raw.raw)
                    if hasattr(result_raw, "raw")
                    else str(result_raw)
                )
                quality_evaluation = self._extract_json(result_text, dict)

                # Recalculate final_score to ensure correctness
                m1 = quality_evaluation.get("metric_1_functional_completeness", {}).get(
                    "score", 0
                )
                m2 = quality_evaluation.get("metric_2_variable_fidelity", {}).get(
                    "score", 0
                )
                m3 = quality_evaluation.get("metric_3_state_machine", {}).get(
                    "score", 0
                )
                m4 = quality_evaluation.get("metric_4_business_logic", {}).get(
                    "score", 0
                )
                m5 = quality_evaluation.get("metric_5_code_quality", {}).get("score", 0)

                # Calculate weighted score: (M1 * 0.25) + (M2 * 0.15) + (M3 * 0.15) + (M4 * 0.35) + (M5 * 0.10)
                calculated_score = (
                    (m1 * 0.25) + (m2 * 0.15) + (m3 * 0.15) + (m4 * 0.35) + (m5 * 0.10)
                )

                # Determine grade based on calculated score
                if calculated_score >= 90:
                    grade = "A"
                elif calculated_score >= 80:
                    grade = "B"
                elif calculated_score >= 70:
                    grade = "C"
                elif calculated_score >= 60:
                    grade = "D"
                else:
                    grade = "F"

                # Override composite_score with calculated values
                quality_evaluation["composite_score"] = {
                    "functional_completeness_weighted": round(m1 * 0.25, 2),
                    "variable_fidelity_weighted": round(m2 * 0.15, 2),
                    "state_machine_weighted": round(m3 * 0.15, 2),
                    "business_logic_weighted": round(m4 * 0.35, 2),
                    "code_quality_weighted": round(m5 * 0.10, 2),
                    "final_score": round(calculated_score, 1),
                    "grade": grade,
                }

                results["quality_evaluation"] = quality_evaluation

                # Print summary
                final_score = calculated_score
                print(
                    f"✓ Quality Evaluation Complete: Score={final_score:.1f}/100, Grade={grade}"
                )

                # Print metric breakdown
                print(f"   Functional Completeness: {m1}/100")
                print(f"   Variable Fidelity: {m2}/100")
                print(f"   State Machine Correctness: {m3}/100")
                print(f"   Business Logic Fidelity: {m4}/100")
                print(f"   Code Quality: {m5}/100")

            except Exception as e:
                print(f"   ⚠️  Quality evaluation failed: {e}")
                import traceback

                traceback.print_exc()
                quality_evaluation = {
                    "error": str(e),
                    "composite_score": {"final_score": 0, "grade": "F"},
                }
                results["quality_evaluation"] = quality_evaluation

            yield {
                "phase": 7,
                "status": "complete",
                "data": {
                    "title": "Quality Evaluation",
                    "message": f"Score: {quality_evaluation.get('composite_score', {}).get('final_score', 0):.1f}/100, Grade: {quality_evaluation.get('composite_score', {}).get('grade', 'N/A')}",
                    "quality_evaluation": quality_evaluation,
                },
            }

        # Save all outputs (applies to both modes)
        self._save_outputs(results, output_dir, schema, contract_text)

        print("\n" + "=" * 70)
        print("✅ TRANSLATION COMPLETE")
        print("=" * 70)

    def evaluate_ground_truth(
        self, ground_truth_code: str, schema, contract_name: str = "GroundTruth"
    ) -> Dict:
        """
        Evaluate ground truth Solidity code using the quality evaluator

        Args:
            ground_truth_code: Ground truth Solidity code
            schema: Contract schema
            contract_name: Name for the evaluation

        Returns:
            Quality evaluation results
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING GROUND TRUTH CODE")
        print(f"{'='*70}")

        # Use the quality evaluator agent
        task_desc = create_quality_evaluation_task_description(
            ground_truth_code, schema, contract_name
        )
        task = Task(
            description=task_desc,
            expected_output=(
                "Valid JSON object with exactly these top-level keys: "
                "metric_1_functional_completeness (object with score 0-100 and evidence), "
                "metric_2_variable_fidelity (object with score 0-100 and evidence), "
                "metric_3_state_machine (object with score 0-100, state_machine_required bool, scoring_path string, and evidence), "
                "metric_4_business_logic (object with score 0-100 and evidence), "
                "metric_5_code_quality (object with score 0-100 and evidence), "
                "composite_score (object with final_score and grade). "
                "For metric_3: if spec has no explicit state_names, set state_machine_required=false and use Path B scoring. "
                "Scores must be precise integers (e.g. 73 not 75). Return ONLY the JSON — no markdown."
            ),
            agent=self.quality_evaluator_agent,
        )
        crew = Crew(agents=[self.quality_evaluator_agent], tasks=[task], verbose=False)

        try:
            result_raw = crew.kickoff()
            result_text = (
                str(result_raw.raw) if hasattr(result_raw, "raw") else str(result_raw)
            )

            try:
                quality_evaluation = self._extract_json(result_text, dict)
            except ValueError as json_err:
                # If JSON parsing completely fails, save the raw output for debugging
                print(
                    f"   ⚠️  JSON parsing failed completely, saving raw output for debugging"
                )
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix="_gt_eval_failed.txt",
                    delete=False,
                    encoding="utf-8",
                ) as f:
                    f.write(result_text)
                    print(f"   📝 Raw output saved to: {f.name}")

                # Return minimal evaluation structure
                return {
                    "error": str(json_err),
                    "composite_score": {"final_score": 0, "grade": "F"},
                    "metric_1_functional_completeness": {"score": 0},
                    "metric_2_variable_fidelity": {"score": 0},
                    "metric_3_state_machine": {"score": 0},
                    "metric_4_business_logic": {"score": 0},
                    "metric_5_code_quality": {"score": 0},
                    "compilation_check": {
                        "compiles": None,
                        "error_message": "JSON parsing failed",
                    },
                }

            # Recalculate final_score to ensure correctness
            m1 = quality_evaluation.get("metric_1_functional_completeness", {}).get(
                "score", 0
            )
            m2 = quality_evaluation.get("metric_2_variable_fidelity", {}).get(
                "score", 0
            )
            m3 = quality_evaluation.get("metric_3_state_machine", {}).get("score", 0)
            m4 = quality_evaluation.get("metric_4_business_logic", {}).get("score", 0)
            m5 = quality_evaluation.get("metric_5_code_quality", {}).get("score", 0)

            # Calculate weighted score: (M1 * 0.25) + (M2 * 0.15) + (M3 * 0.15) + (M4 * 0.35) + (M5 * 0.10)
            calculated_score = (
                (m1 * 0.25) + (m2 * 0.15) + (m3 * 0.15) + (m4 * 0.35) + (m5 * 0.10)
            )

            # Determine grade based on calculated score
            if calculated_score >= 90:
                grade = "A"
            elif calculated_score >= 80:
                grade = "B"
            elif calculated_score >= 70:
                grade = "C"
            elif calculated_score >= 60:
                grade = "D"
            else:
                grade = "F"

            # Override composite_score with calculated values
            quality_evaluation["composite_score"] = {
                "functional_completeness_weighted": round(m1 * 0.25, 2),
                "variable_fidelity_weighted": round(m2 * 0.15, 2),
                "state_machine_weighted": round(m3 * 0.15, 2),
                "business_logic_weighted": round(m4 * 0.35, 2),
                "code_quality_weighted": round(m5 * 0.10, 2),
                "final_score": round(calculated_score, 1),
                "grade": grade,
            }

            # Print summary
            final_score = calculated_score
            print(
                f"✓ Ground Truth Evaluation: Score={final_score:.1f}/100, Grade={grade}"
            )

            # Print metric breakdown for ground truth (same as generated code)
            print(f"   📊 Functional Completeness: {m1}/100")
            print(f"   📊 Variable Fidelity: {m2}/100")
            print(f"   📊 State Machine Correctness: {m3}/100")
            print(f"   📊 Business Logic Fidelity: {m4}/100")
            print(f"   📊 Code Quality: {m5}/100")

            # Check compilation
            compiler_checker = SolidityCompilationChecker()
            compilation_result = compiler_checker.check_compilation(ground_truth_code)
            quality_evaluation["compilation_check"] = compilation_result

            if compilation_result["compiles"] is None:
                print(f"   ⚠️  {compilation_result['error_message']}")
            elif compilation_result["compiles"]:
                warnings_text = (
                    f" ({len(compilation_result['warnings'])} warnings)"
                    if compilation_result["warnings"]
                    else ""
                )
                print(f"   ✅ Ground truth compiles successfully{warnings_text}")
            else:
                error_preview = (
                    compilation_result["error_message"][:100] + "..."
                    if len(compilation_result["error_message"]) > 100
                    else compilation_result["error_message"]
                )
                print(f"   ❌ Compilation failed: {error_preview}")

            return quality_evaluation

        except Exception as e:
            print(f"   ⚠️  Ground truth evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "error": str(e),
                "composite_score": {"final_score": 0, "grade": "F"},
                "compilation_check": {
                    "compiles": None,
                    "error_message": "Evaluation failed",
                },
            }

    def translate_contract(
        self,
        input_path: str,
        output_dir: str = "./output",
        require_audit_approval: bool = True,
        generate_mcp_server: bool = True,
        use_agentic_pipeline: bool = True,
    ) -> Dict:
        """
        Complete translation workflow.

        Args:
            input_path: Path to contract file (PDF or text)
            output_dir: Output directory for generated files
            require_audit_approval: Whether to require user approval on security issues
            generate_mcp_server: Whether to generate MCP server code
            use_agentic_pipeline: If True, use Agent/Task/Crew approach. If False, use legacy Program approach.

        Returns:
            Dict with translation results
        """

        print("\n" + "=" * 70)
        if use_agentic_pipeline:
            print("IBM AGENTICS CONTRACT TRANSLATOR (Agent/Task Pipeline)")
        else:
            print("IBM AGENTICS CONTRACT TRANSLATOR (Legacy Program Pipeline)")
        print("=" * 70)

        # Phase 1: Document Processing
        print("\n[Phase 1/6] Document Processing")
        if input_path.endswith(".pdf"):
            contract_text = self.extract_text_from_pdf(input_path)
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                contract_text = f.read()
        print(f"✓ Extracted {len(contract_text)} characters")

        # Execute pipeline based on mode
        if use_agentic_pipeline:
            # NEW: Use Agent/Task/Crew orchestration
            results = self._run_agentic_pipeline(contract_text, generate_mcp_server)
        else:
            # Legacy: Use Program.forward() calls
            results = {}

            # Phase 2: Contract Analysis
            print("\n[Phase 2/6] Contract Analysis (Parser Program)")
            schema = self.parser.forward(contract_text, self.llm)
            results["schema"] = schema
            _conds = schema.conditions if schema.conditions else {}
            print(
                f"✓ Parsed: {len(_conds.get('state_names', []))} states, {len(_conds.get('events', []))} events, {len(_conds.get('function_names', []))} functions, {len(schema.obligations)} obligations"
            )

            # Phase 3: Solidity Generation
            print("\n[Phase 3/6] Code Generation (Generator Program)")
            solidity_code = self.generator.forward(schema, self.llm)
            results["solidity"] = solidity_code
            print(f"✓ Generated {len(solidity_code.splitlines())} lines")

            # Phase 4: Security Audit
            print("\n[Phase 4/6] Security Analysis (Auditor Program)")
            audit_report = self.auditor.forward(solidity_code, self.llm)
            results["audit"] = audit_report
            severity = audit_report.get("severity_level", "unknown")
            score = audit_report.get("security_score", "N/A")
            print(f"✓ Audit: Severity={severity}, Score={score}")

            # Phase 5: ABI Generation
            print("\n[Phase 5/6] Interface Generation (ABI Program)")
            abi = self.abi_generator.forward(solidity_code, self.llm)
            results["abi"] = abi
            print(f"✓ Generated {len(abi)} ABI elements")

            # Phase 6: MCP Server Generation
            if generate_mcp_server:
                print("\n[Phase 6/6] MCP Server Generation (MCP Generator Program)")
                contract_name = (
                    "_".join(
                        [p.name.replace(" ", "_")[:10] for p in schema.parties[:2]]
                    )
                    if schema.parties
                    else "Contract"
                )
                contract_name = contract_name[:40]
                mcp_server_code = self.mcp_generator.forward(
                    abi, schema, contract_name, self.llm
                )
                results["mcp_server"] = mcp_server_code
                print(
                    f"✓ Generated MCP server ({len(mcp_server_code.splitlines())} lines)"
                )
            else:
                print("\n[Phase 6/6] MCP Server Generation - SKIPPED")

        # Check audit approval (applies to both modes)
        schema = results.get("schema")
        audit_report = results.get("audit", {})

        if require_audit_approval and not audit_report.get("approved", False):
            print("\n⚠️  Security issues detected!")
            for i, issue in enumerate(audit_report.get("issues", [])[:3], 1):
                print(f"   {i}. {issue}")

            response = input("\n   Continue? (yes/no): ").lower()
            if response != "yes":
                raise Exception("Halted due to security concerns")

        # Save all outputs
        self._save_outputs(results, output_dir, schema, contract_text)

        print("\n" + "=" * 70)
        print("✅ TRANSLATION COMPLETE")
        print("=" * 70)

        return results

    def _save_outputs(
        self, results: Dict, output_dir: str, schema, contract_text: str = ""
    ):
        """Save all outputs including MCP server"""

        print("\n💾 Saving outputs...")

        # Create directories
        base_output_path = Path(output_dir)
        base_output_path.mkdir(exist_ok=True, parents=True)

        contract_type = schema.contract_type.replace("_", " ").title()
        subdirectory_name = contract_type.replace(" ", "_")

        run_number = 1
        subdir_path = base_output_path / f"{subdirectory_name}_{run_number}"
        while subdir_path.exists():
            run_number += 1
            subdir_path = base_output_path / f"{subdirectory_name}_{run_number}"

        subdir_path.mkdir(exist_ok=True, parents=True)

        # Generate contract filename
        contract_name = (
            "_".join([p.name.replace(" ", "_")[:10] for p in schema.parties[:2]])
            if schema.parties
            else "Contract"
        )
        contract_name = contract_name[:40]

        # Save original English contract input
        if contract_text:
            with open(subdir_path / "contract_input.txt", "w", encoding="utf-8") as f:
                f.write(contract_text)
            print(f"   ✓ contract_input.txt")

        # Save Solidity
        with open(subdir_path / f"{contract_name}.sol", "w", encoding="utf-8") as f:
            f.write(results["solidity"])
        print(f"   ✓ {contract_name}.sol")

        # Save ABI
        abi_filename = f"{contract_name}.abi.json"
        with open(subdir_path / abi_filename, "w", encoding="utf-8") as f:
            json.dump(results["abi"], f, indent=2)
        print(f"   ✓ {abi_filename}")

        # Save schema
        with open(subdir_path / "contract_schema.json", "w", encoding="utf-8") as f:
            json.dump(results["schema"].model_dump(), f, indent=2)
        print(f"   ✓ contract_schema.json")

        # Save audit
        with open(subdir_path / "security_audit.json", "w", encoding="utf-8") as f:
            json.dump(results["audit"], f, indent=2)
        print(f"   ✓ security_audit.json")

        # Save quality evaluation
        if "quality_evaluation" in results:
            with open(
                subdir_path / "quality_evaluation.json", "w", encoding="utf-8"
            ) as f:
                json.dump(results["quality_evaluation"], f, indent=2)
            print(f"   ✓ quality_evaluation.json")

        # Save MCP Server
        if "mcp_server" in results:
            mcp_filename = f"{contract_name}_mcp_server.py"
            with open(subdir_path / mcp_filename, "w", encoding="utf-8") as f:
                f.write(results["mcp_server"])
            print(f"   ✓ {mcp_filename}")

            # Create .env file
            env_content = f"""# MCP Server Configuration for {contract_name}
# Fill in your values below, then run the MCP server

# Blockchain RPC endpoint (e.g., http://127.0.0.1:8545 for Ganache)
RPC_URL=http://127.0.0.1:8545

# Private key for signing transactions (get from Ganache, without 0x prefix)
PRIVATE_KEY=your_private_key_here

# Deployed contract address (get after deploying Solidity contract)
CONTRACT_ADDRESS=0x...
"""
            with open(subdir_path / ".env", "w", encoding="utf-8") as f:
                f.write(env_content)
            print(f"   ✓ .env")

            # Create .env.example
            env_example = f"""# MCP Server Configuration for {contract_name}
# This is an example. Copy to .env and fill in your values

# Blockchain RPC endpoint (Infura, Alchemy, or local Ganache)
RPC_URL=http://127.0.0.1:8545

# Private key for signing transactions (without 0x prefix)
PRIVATE_KEY=your_private_key_here

# Deployed contract address (will be filled after deployment)
CONTRACT_ADDRESS=0x...
"""
            with open(subdir_path / ".env.example", "w", encoding="utf-8") as f:
                f.write(env_example)
            print(f"   ✓ .env.example")

        # Create README
        schema = results["schema"]
        audit = results["audit"]

        readme = f"""# IBM Agentics Contract Translation

## Contract Summary
- **Type**: {schema.contract_type}
- **Parties**: {', '.join(p.name for p in schema.parties)}
- **Financial Terms**: {len(schema.financial_terms)} term(s)

## Security Audit
- **Status**: {'✅ APPROVED' if audit.get('approved') else '⚠️ REVIEW NEEDED'}
- **Severity**: {audit['severity_level'].upper()}
- **Score**: {audit.get('security_score', 'N/A')}

## Generated Files

### Smart Contract Files
1. **{contract_name}.sol** - Solidity smart contract ({len(results['solidity'].splitlines())} lines)
2. **{contract_name}.abi.json** - Contract ABI ({len(results['abi'])} elements)

### Configuration & Documentation
3. **contract_schema.json** - Structured contract data
4. **security_audit.json** - Security audit report

### MCP Server
5. **{contract_name}_mcp_server.py** - Custom MCP server ({len(results.get('mcp_server', '').splitlines())} lines)
6. **.env.example** - Environment configuration template

## Using the MCP Server

### 1. Setup Environment
```bash
# Copy and configure environment file
cp .env.example .env

# Edit .env with your values:
# - RPC_URL: Your blockchain endpoint
# - PRIVATE_KEY: Your wallet private key
# - CONTRACT_ADDRESS: Deployed contract address
```

### 2. Install Dependencies
```bash
pip install web3 python-dotenv fastmcp
```

### 3. Deploy Contract
First deploy the Solidity contract to get CONTRACT_ADDRESS:
```bash
# Using Remix, Hardhat, or web3.py
# Update CONTRACT_ADDRESS in .env after deployment
```

### 4. Run MCP Server
```bash
python {contract_name}_mcp_server.py
```

### 5. Available Tools
The MCP server exposes these tools based on the contract ABI:
{self._generate_tool_list(results.get('abi', []))}

## Next Steps
1. ✅ Review security audit
2. ✅ Deploy Solidity contract to testnet
3. ✅ Update .env with CONTRACT_ADDRESS
4. ✅ Run MCP server
5. ✅ Connect AI agents to MCP server
6. ✅ Test contract interactions

---
*Generated by IBM Agentics Framework*
*MCP Server auto-generated from ABI*
"""

        with open(subdir_path / "README.md", "w", encoding="utf-8") as f:
            f.write(readme)
        print(f"   ✓ README.md")

        try:
            display_path = subdir_path.relative_to(Path.cwd())
        except ValueError:
            display_path = subdir_path.resolve()
        print(f"\n📁 Outputs saved to: {display_path}")

    def _generate_tool_list(self, abi: List[Dict]) -> str:
        """Generate markdown list of available MCP tools"""

        tools = []
        for item in abi:
            if item.get("type") == "function":
                name = item.get("name")
                stateMutability = item.get("stateMutability", "nonpayable")

                if stateMutability == "payable":
                    tools.append(f"- `{name}()` - Payable transaction")
                elif stateMutability in ["view", "pure"]:
                    tools.append(f"- `{name}()` - Read-only query")
                else:
                    tools.append(f"- `{name}()` - State-changing transaction")

        return "\n".join(tools) if tools else "- No tools available"


__all__ = ["IBMAgenticContractTranslator"]
