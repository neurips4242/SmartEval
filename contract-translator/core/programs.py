"""
Legacy Program classes kept for backward compatibility.

These IBM Agentics Program classes use the traditional `forward()` pattern.
The current pipeline uses Agent/Task/Crew orchestration instead.
"""

import json
from typing import Dict, List

from .schemas import UniversalContractSchema

# agentics.LLM / Program / message helpers were removed in agentics >=2.
# Provide minimal stubs so the module imports cleanly; the legacy forward()
# methods will raise NotImplementedError if actually called.
try:
    from agentics import LLM, Program, system_message, user_message  # type: ignore
except ImportError:

    class LLM:  # type: ignore
        def __init__(self, model: str = "gpt-4o-mini", **kw):
            self.model = model

    class Program:  # type: ignore
        def forward(self, *a, **kw):
            raise NotImplementedError(
                "Legacy agentics.Program API is not available in this environment. "
                "Use the CrewAI agent pipeline instead."
            )

    def user_message(content: str, **kw) -> dict:  # type: ignore
        return {"role": "user", "content": content}

    def system_message(content: str, **kw) -> dict:  # type: ignore
        return {"role": "system", "content": content}


# Import all Program classes from the original file temporarily
import sys
from pathlib import Path

from .task_builders import (
    create_abi_generator_task_description,
    create_audit_task_description,
    create_solidity_generation_prompt,
)

# Add parent directory to import from original file
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ==================== PROGRAM CLASS 1: UniversalContractParserProgram ====================


class UniversalContractParserProgram(Program):
    """
    Legacy Program class - kept for compatibility.
    Use create_parser_instructions() for Agent-based approach.
    """

    def forward(self, contract_text: str, lm: LLM) -> UniversalContractSchema:
        """Parse any contract type"""

        messages = [
            system_message(
                """You are an expert contract analyst who extracts EXACT, SPECIFIC information from contracts.

                CRITICAL INSTRUCTIONS:
                1. Extract the EXACT function names mentioned in the contract (e.g., "initializeLease", "payRent", "confirmDelivery")
                2. Extract the EXACT variable names mentioned (e.g., "monthlyRent", "securityDeposit", "deliveryDate")
                3. Extract the EXACT state names mentioned (e.g., "Pending", "Active", "Completed", "Terminated")
                4. Extract the EXACT party roles as described in the contract
                5. DO NOT use generic placeholders - use the specific terminology from the contract
                6. Capture ALL conditions, transitions, and logic flows mentioned

                Your goal: Create a structured representation that preserves ALL specific details from the contract text."""
            ),
            user_message(
                f"""Analyze this contract and extract ALL SPECIFIC information exactly as mentioned.

CONTRACT TEXT:
{contract_text}

PAY CLOSE ATTENTION TO:
1. **Specific Function Names**: If the contract says "The main functions include [initializeLease(), payRent(), terminateLease()]", extract EXACTLY those names
2. **Specific Variable Names**: If it mentions "monthlyRent", "securityDeposit", "leaseStartDate", extract those EXACT names
3. **Specific States**: If it mentions states like "Initializing", "Active", "Processing", "Terminated", extract those EXACT state names
4. **State Transitions**: Capture the EXACT transition logic (e.g., "Initializing → Active when lease starts")
5. **Specific Conditions**: Extract the EXACT conditions mentioned (e.g., "rent must be paid by day 5 of each month")
6. **Specific Events**: If events are mentioned like "LeaseInitialized", "RentPaid", extract those EXACT names
7. **Obligations**: For smart contracts, EVERY described function or operation is an obligation. Map each to: party = who is authorized to call it (e.g. "token holder", "owner"), description = what it does on-chain. NEVER leave obligations empty when functions are described.

Return ONLY valid JSON with this structure:
{{
    "contract_type": "rental|employment|sales|service|loan|nda|partnership|investment|other",
    "title": "specific contract title from text",
    "parties": [
        {{
            "name": "EXACT party name from contract",
            "role": "EXACT role as described (not generic)",
            "address": "blockchain address if mentioned",
            "email": "if mentioned",
            "entity_type": "individual|company|organization"
        }}
    ],
    "financial_terms": [
        {{
            "amount": number,
            "currency": "ETH|USD|etc",
            "purpose": "EXACT purpose from contract (not 'payment' but 'monthly rent' or 'security deposit')",
            "frequency": "EXACT frequency from contract",
            "due_date": "EXACT due date or day of month"
        }}
    ],
    "dates": [
        {{
            "date_type": "EXACT date type from contract (leaseStartDate, deliveryDeadline, etc)",
            "value": "date string if provided",
            "day_of_month": number or null,
            "frequency": "if recurring"
        }}
    ],
    "assets": [
        {{
            "type": "SPECIFIC asset type from contract",
            "description": "EXACT description from contract",
            "location": "if mentioned",
            "quantity": number or null,
            "value": number or null
        }}
    ],
    "obligations": [
        {{
            "party": "WHO performs or is authorized to call this (e.g. 'token holder', 'contract owner', 'buyer', 'tenant')",
            "description": "WHAT they must/can do - map EVERY described function or operation to an obligation here",
            "deadline": "EXACT deadline if mentioned, else null",
            "penalty_for_breach": "EXACT penalty if mentioned, else null"
        }}
    ],

CRITICAL FOR OBLIGATIONS: For smart contracts, EVERY described function or operation IS an obligation.
Do NOT leave this array empty if functions are described.
For each function/operation mentioned, create one obligation entry:
  - party = the role authorized to call it (e.g. 'token holder', 'owner', 'spender', 'buyer')
  - description = what the operation does on-chain (e.g. 'transfer tokens to another address')
Example: if contract mentions 'transferring tokens' and 'approving spenders', extract:
  {{"party": "token holder", "description": "transfer tokens to any address", "deadline": null, "penalty_for_breach": null}}
  {{"party": "token holder", "description": "approve a spender to transfer tokens on their behalf", "deadline": null, "penalty_for_breach": null}}

    "special_terms": ["EXACT special conditions word-for-word"],
    "conditions": {{
        "function_names": ["EXACT function names from contract: initializeLease, payRent, etc"],
        "variable_names": ["EXACT variable names: monthlyRent, securityDeposit, tenantAddress, etc"],
        "state_names": ["EXACT state names: Pending, Active, Completed, Terminated, etc"],
        "state_transitions": ["EXACT transitions: Pending->Active when X, Active->Completed when Y"],
        "events": ["EXACT event names: LeaseInitialized, RentPaid, LeaseTerminated, etc"],
        "logic_conditions": ["EXACT conditions: rent due on day 5, penalty if late > 7 days, etc"]
    }},
    "termination_conditions": ["EXACT termination conditions from contract"]
}}

EXTRACT EVERYTHING SPECIFIC - DO NOT USE GENERIC NAMES OR PLACEHOLDERS.
OBLIGATIONS MUST NOT BE EMPTY if any functions or operations are described."""
            ),
        ]

        response = lm.chat(messages=messages)
        response_text = str(response).strip()

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()

        parsed = json.loads(response_text)

        # ===== VALIDATION & CLEANUP =====
        # Ensure financial_terms have required fields
        if "financial_terms" in parsed and parsed["financial_terms"]:
            cleaned_terms = []
            for term in parsed["financial_terms"]:
                # Skip terms with None amount or currency
                if term.get("amount") is None or term.get("currency") is None:
                    continue
                # Ensure amount is a number
                try:
                    term["amount"] = float(term["amount"])
                except (ValueError, TypeError):
                    continue
                # Ensure currency is a string
                if not isinstance(term.get("currency"), str):
                    term["currency"] = "ETH"
                # Ensure purpose exists
                if not term.get("purpose"):
                    term["purpose"] = "Contract payment"
                cleaned_terms.append(term)
            parsed["financial_terms"] = cleaned_terms

        # Ensure parties have required fields
        if "parties" in parsed and parsed["parties"]:
            cleaned_parties = []
            for party in parsed["parties"]:
                if party.get("name"):  # Only keep parties with names
                    if not party.get("role"):
                        party["role"] = "other"
                    cleaned_parties.append(party)
            parsed["parties"] = cleaned_parties

        # Ensure at least one party
        if not parsed.get("parties"):
            parsed["parties"] = [{"name": "Unknown Party", "role": "other"}]

        # Ensure contract_type is set
        if not parsed.get("contract_type"):
            parsed["contract_type"] = "other"

        return UniversalContractSchema(**parsed)


# ==================== PROGRAM CLASS 2: UniversalSolidityGeneratorProgram ====================


class UniversalSolidityGeneratorProgram(Program):
    """Generates Solidity for ANY contract type"""

    def forward(self, schema: UniversalContractSchema, lm: LLM) -> str:
        """Generate contract-type-specific Solidity using task_builders prompt"""

        # Use the comprehensive prompt from task_builders.py
        prompt = create_solidity_generation_prompt(schema)

        messages = [
            system_message(
                """You are an expert Solidity engineer. Your sole job is to implement the contract specification given to you.

CRITICAL RULES (violating any of these means you have failed):
- Return ONLY raw Solidity code — no markdown fences, no explanation text
- Every function body must contain real logic: require() checks, state changes, value transfers, event emissions
- No empty functions, no TODO comments, no placeholder logic
- Every state variable must be written by at least one function and read by at least one function
- Every obligation listed in the spec must become a complete function with full on-chain logic
- If the spec is a token contract, implement the FULL ERC20 interface
- If the spec has temporal deadlines, enforce them with require(block.timestamp ...)
- Access control modifiers must use require() that actually reverts on failure
- Length target: 150-400 lines — a correct 300-line contract is far better than a 60-line stub
- NEVER name a function parameter the same as a contract-level state variable (e.g. do not use `owner` as a param if `address public owner` exists — use `tokenOwner` or `account` instead). This causes "Identifier already declared" compile errors.
- NEVER declare a public state variable with the same name as an interface function you are implementing (e.g. do NOT write `uint256 public totalSupply` when implementing IERC20 — use `uint256 private _totalSupply` and return it from the interface function). The auto-generated getter will clash with the interface function and cause a compile error.

Read every line of the specification prompt. The MANDATORY sections are mandatory. Do not skip any."""
            ),
            user_message(prompt),
        ]

        response = lm.chat(messages=messages)
        solidity_code = str(response).strip()

        # Remove markdown code fences if present
        if "```solidity" in solidity_code:
            solidity_code = (
                solidity_code.split("```solidity")[1].split("```")[0].strip()
            )
        elif "```" in solidity_code:
            solidity_code = solidity_code.split("```")[1].split("```")[0].strip()

        # Validate code quality
        quality_issues = self._validate_code_quality(solidity_code, schema)
        if quality_issues:
            print(f"\n⚠️  CODE QUALITY ISSUES DETECTED:")
            for issue in quality_issues:
                print(f"   - {issue}")
            print(f"\n   These issues should be addressed in future iterations.")

        return solidity_code

    def _validate_code_quality(
        self, solidity_code: str, schema: UniversalContractSchema
    ) -> List[str]:
        """Validate generated code for common quality issues"""
        issues = []

        # Check for placeholder comments
        if (
            "// logic goes here" in solidity_code.lower()
            or "// todo" in solidity_code.lower()
        ):
            issues.append("Contains placeholder comments - logic not fully implemented")

        # Check for silent failure pattern
        if "if (" in solidity_code and "return;" in solidity_code:
            lines = solidity_code.split("\n")
            for i, line in enumerate(lines):
                if "if (" in line and i + 1 < len(lines) and "return;" in lines[i + 1]:
                    issues.append(
                        f"Silent failure detected (if/return pattern) - should use require()"
                    )

        # Check if declared variables are used
        conditions = schema.conditions if schema.conditions else {}
        variable_names = conditions.get("variable_names", [])
        for var_name in variable_names[:5]:  # Check first 5 variables
            if var_name and var_name not in solidity_code:
                issues.append(
                    f"Variable '{var_name}' from contract not found in generated code"
                )

        # Check if function names are used
        function_names = conditions.get("function_names", [])
        for func_name in function_names[:5]:  # Check first 5 functions
            if func_name and f"function {func_name}" not in solidity_code:
                issues.append(f"Function '{func_name}' from contract not implemented")

        # Check for empty function bodies
        if "function " in solidity_code and "{ }" in solidity_code:
            issues.append("Contains empty function bodies")

        # Check for access control
        if "onlyOwner" not in solidity_code and "owner" in solidity_code.lower():
            issues.append("Owner declared but no access control modifier used")

        # Check for time-based variables that aren't checked
        if (
            "deadline" in solidity_code.lower()
            and "block.timestamp" not in solidity_code
        ):
            issues.append(
                "Time variable declared but never checked against block.timestamp"
            )

        return issues

    def regenerate_with_error_feedback(
        self, schema: UniversalContractSchema, error_message: str, lm: LLM
    ) -> str:
        """Regenerate contract with compilation error feedback"""

        print(f"\n🔧 REGENERATING CONTRACT WITH ERROR FEEDBACK")
        print(f"   Error reported: {error_message[:100]}...")

        # Extract specific names from schema
        conditions = schema.conditions if schema.conditions else {}
        function_names = conditions.get("function_names", [])
        variable_names = conditions.get("variable_names", [])
        state_names = conditions.get("state_names", [])
        events = conditions.get("events", [])

        messages = [
            system_message(
                f"""You are a Solidity expert debugging and regenerating smart contracts.

CRITICAL INSTRUCTIONS:
1. FIX the compilation error: {error_message[:150]}
2. KEEP the EXACT function names: {', '.join(function_names) if function_names else 'from schema'}
3. KEEP the EXACT variable names: {', '.join(variable_names) if variable_names else 'from schema'}
4. KEEP the EXACT state names: {', '.join(state_names) if state_names else 'from schema'}
5. KEEP the EXACT event names: {', '.join(events) if events else 'from schema'}
6. DO NOT change to generic names - preserve all specific terminology

MUST GENERATE:
- Valid Solidity ^0.8.0 syntax
- Every statement ends with semicolon
- All function bodies complete
- All parentheses/brackets matched
- Exact names from the contract preserved"""
            ),
            user_message(
                f"""REGENERATE the contract fixing this error: {error_message[:200]}

CONTRACT SCHEMA:
{schema.model_dump_json(indent=2)}

PRESERVE THESE EXACT NAMES:
- Functions: {', '.join(function_names) if function_names else 'extract from obligations'}
- Variables: {', '.join(variable_names) if variable_names else 'extract from terms'}
- States: {', '.join(state_names) if state_names else 'extract from transitions'}
- Events: {', '.join(events) if events else 'create from function names'}

REQUIREMENTS:
1. Fix the syntax error completely
2. Use EXACT names from above (not generic replacements)
3. Every statement must end with semicolon
4. Complete all function bodies
5. Handle all parties: {[p.name + ' (' + p.role + ')' for p in schema.parties]}
6. Implement all financial terms: {[f"{t.purpose}: {t.amount} {t.currency}" for t in schema.financial_terms]}
7. Implement all obligations as functions

Return ONLY valid, compilable Solidity code with EXACT names preserved."""
            ),
        ]

        print(f"   Requesting LLM to regenerate with error feedback...")
        response = lm.chat(messages=messages)
        solidity_code = str(response).strip()

        # Remove markdown code fences if present
        if "```solidity" in solidity_code:
            solidity_code = (
                solidity_code.split("```solidity")[1].split("```")[0].strip()
            )
        elif "```" in solidity_code:
            solidity_code = solidity_code.split("```")[1].split("```")[0].strip()

        print(f"   ✓ Regenerated contract ({len(solidity_code.splitlines())} lines)")
        return solidity_code

    def _get_requirements_for_type(self, contract_type: str) -> str:
        """Get contract-type-specific requirements"""

        requirements_map = {
            "non_disclosure_agreement": """
REQUIRED FUNCTIONS FOR NDA:
VIEW FUNCTIONS (getters - must handle missing data gracefully):
- getPartyA() returns address
- getPartyB() returns address
- getConfidentialityPeriodDays() returns uint
- getBreachPenaltyAmount() returns uint
- isConfidentialityActive() returns bool
- getBreachCount() returns uint

ACTION FUNCTIONS (only if relevant):
- confirmConfidentiality(bool agreeToTerms)
- reportBreach(string memory description)
- calculatePenalty(uint breachCount) returns uint256
- checkTerminationDate() returns (bool isExpired, uint daysRemaining)

STATE VARIABLES TO STORE:
- partyA, partyB (addresses, use address(0) if missing)
- confidentialityStartDate, confidentialityPeriodDays (0 if not specified)
- breachPenalty (0 if not specified)
- breachReportedCount (tracks breaches)
- isActive (bool)

IMPORTANT: All functions must be defensive - return safely even if data missing.""",
            "rental_agreement": """
REQUIRED FUNCTIONS FOR RENTAL:
VIEW FUNCTIONS:
- getLandlord() returns address
- getTenant() returns address
- getMonthlyRent() returns uint
- getSecurityDeposit() returns uint
- getLeaseStartDate() returns uint
- getLeaseEndDate() returns uint
- isLeaseActive() returns bool
- getTotalRentPaid() returns uint
- getDaysUntilLeaseEnd() returns uint

ACTION FUNCTIONS:
- payRent(uint amountInWei) payable
- inspectProperty(string memory notes)
- terminateLease(string memory reason)
- refundSecurityDeposit()

STATE VARIABLES TO STORE:
- landlord, tenant (addresses)
- monthlyRent, securityDeposit (amounts)
- leaseStartDate, leaseEndDate (dates)
- totalRentPaid (tracking)
- isActive (bool)

IMPORTANT: All getters return safely with defaults (0, address(0)) if data missing.""",
            "employment_contract": """
REQUIRED FUNCTIONS FOR EMPLOYMENT:
VIEW FUNCTIONS:
- getEmployee() returns address
- getEmployer() returns address
- getBaseSalary() returns uint
- getPerformanceBonus() returns uint
- getEmploymentStartDate() returns uint
- getEmploymentEndDate() returns uint
- isEmploymentActive() returns bool
- getTotalSalaryEarned() returns uint
- getOutstandingSalary() returns uint

ACTION FUNCTIONS:
- payEmployeeSalary() payable
- payBonus(uint bonusAmount) payable
- terminateEmployment(string memory reason)
- claimSeverancePayment()

STATE VARIABLES TO STORE:
- employee, employer (addresses)
- baseSalary, performanceBonus (amounts, 0 if not specified)
- employmentStartDate, employmentEndDate (dates, 0 if missing)
- totalSalaryPaid, isEmployed (tracking)

IMPORTANT: Handle missing salary/bonus gracefully - return 0, don't fail.""",
            "sales_agreement": """
REQUIRED FUNCTIONS FOR SALES:
VIEW FUNCTIONS:
- getSeller() returns address
- getBuyer() returns address
- getGoodsDescription() returns string
- getPurchasePrice() returns uint
- getPaymentTerms() returns string
- isDeliveryComplete() returns bool
- hasInspectionPassed() returns bool
- getOutstandingPayment() returns uint

ACTION FUNCTIONS:
- confirmOrderDetails(string memory terms)
- makePayment() payable
- shipGoods(string memory trackingNumber)
- confirmDelivery()
- inspectGoods(bool passed, string memory notes)
- releaseFunds()

STATE VARIABLES TO STORE:
- seller, buyer (addresses)
- goodsDescription (string)
- purchasePrice, totalPaidAmount (amounts, 0 if missing)
- deliveryConfirmed, inspectionPassed (bools)

IMPORTANT: Handle missing descriptions and prices gracefully.""",
            "service_agreement": """
REQUIRED FUNCTIONS FOR SERVICE:
VIEW FUNCTIONS:
- getServiceProvider() returns address
- getClient() returns address
- getServiceDescription() returns string
- getMilestoneAmount() returns uint
- getTotalMilestones() returns uint
- getCompletedMilestones() returns uint
- getMonthlyServiceFee() returns uint
- getTotalAmountPaid() returns uint
- isServiceActive() returns bool

ACTION FUNCTIONS:
- confirmServiceStart()
- payMonthlyServiceFee() payable
- payMilestonePayment(uint milestoneNumber) payable
- reportMilestoneCompletion(string memory evidence)
- approveMilestoneCompletion(uint milestoneNumber)
- reportServiceIssue(string memory issue)
- terminateService(string memory reason)

STATE VARIABLES TO STORE:
- serviceProvider, client (addresses)
- milestoneAmount, monthlyServiceFee (amounts, 0 if missing)
- completedMilestones, totalAmountPaid (tracking)
- isActive (bool)

IMPORTANT: All functions safe with missing milestone/fee data.""",
            "loan_agreement": """
REQUIRED FUNCTIONS FOR LOAN:
VIEW FUNCTIONS:
- getLender() returns address
- getBorrower() returns address
- getPrincipalAmount() returns uint
- getInterestRate() returns uint
- getMonthlyPayment() returns uint
- getLoanTermMonths() returns uint
- getTotalAmountRepaid() returns uint
- getRemainingBalance() returns uint
- isLoanActive() returns bool
- isLoanInDefault() returns bool

ACTION FUNCTIONS:
- disburseLoan() payable
- makeMonthlyPayment() payable
- makePrepayment(uint amount) payable
- calculateInterestAccrued() returns uint
- reportPaymentDefault()
- cureDefaultPayment() payable
- terminateLoanEarly() payable

STATE VARIABLES TO STORE:
- lender, borrower (addresses)
- principalAmount, interestRate, monthlyPayment (amounts, 0 if missing)
- totalAmountRepaid, isActive, inDefault (tracking)

IMPORTANT: Interest calculations return 0 if rate not specified.""",
            "investment_agreement": """
REQUIRED FUNCTIONS FOR INVESTMENT:
VIEW FUNCTIONS:
- getInvestor() returns address
- getCompany() returns address
- getInvestmentAmount() returns uint
- getEquityPercentage() returns uint
- getSharesPurchased() returns uint
- getInvestmentDate() returns uint
- getDividendRate() returns uint
- getTotalDividendsPaid() returns uint
- canRedeemShares() returns bool

ACTION FUNCTIONS:
- fundInvestment() payable
- claimBoardSeat()
- requestFinancialStatements()
- receiveDividendPayment() payable
- claimDividends()
- reportDownRound(uint newValuation)
- requestRedemption()
- settleRedemption() payable

STATE VARIABLES TO STORE:
- investor, company (addresses)
- investmentAmount, equityPercentage, sharesPurchased (amounts, 0 if missing)
- investmentDate, dividendRate (0 if missing)
- totalDividendsPaid, boardSeatGranted (tracking)

IMPORTANT: All functions safe with missing valuation/dividend data.""",
        }

        return requirements_map.get(
            contract_type,
            """
REQUIRED FOR ALL CONTRACTS:
VIEW FUNCTIONS:
- Create getters for all mentioned parties, amounts, and dates
- All getters must return safely with sensible defaults
- Return 0 for uint, address(0) for address, false for bool, "" for string

ACTION FUNCTIONS:
- Create functions for all contract obligations mentioned
- Use if/else for optional conditions, NOT require()
- Never fail just because optional data is missing

STATE VARIABLES:
- Store all parties (use address(0) if missing)
- Store all amounts (use 0 if missing)
- Store all dates (use 0 if missing)
- Store tracking variables initialized to 0 or false

DEFENSIVE PROGRAMMING RULES:
- EVERY function must handle missing data gracefully
- Return sensible defaults, never revert on missing fields
- Check if amounts > 0 before operations
- Never require() to fail due to missing optional terms""",
        )


# ==================== PROGRAM CLASS 3: SecurityAuditorProgram ====================


class SecurityAuditorProgram(Program):
    """IBM Agentics Program for security auditing"""

    def forward(self, solidity_code: str, lm: LLM) -> Dict:
        """Perform security audit"""

        messages = [
            system_message(
                """You are a blockchain security expert specializing in Solidity smart contract auditing.

CRITICAL RULES:
- Return ONLY valid JSON — no markdown fences, no explanations
- Every issue MUST reference a specific function name and describe the exact exploit path
- severity_level must be one of: none, low, medium, high, critical
- approved must be true only if severity is none or low
- security_score: A=no issues, B=low only, C=medium, D=high, F=critical
- vulnerability_count must equal the exact number of items in the issues array
- recommendations must be specific line-level fixes, not generic advice"""
            ),
            user_message(create_audit_task_description(solidity_code)),
        ]

        response = lm.chat(messages=messages)
        audit_text = str(response).strip()

        if "```json" in audit_text:
            audit_text = audit_text.split("```json")[1].split("```")[0].strip()
        elif "```" in audit_text:
            audit_text = audit_text.split("```")[1].split("```")[0].strip()

        return json.loads(audit_text)


# ==================== PROGRAM CLASS 4: ABIGeneratorProgram ====================


class ABIGeneratorProgram(Program):

    def forward(self, solidity_code: str, lm: LLM) -> List[Dict]:
        """Generate ABI"""

        messages = [
            system_message(
                """You are an Ethereum ABI expert generating complete, accurate ABI JSON from Solidity contracts.

CRITICAL RULES:
- Return ONLY the JSON array — no markdown fences, no explanations
- Include EVERY public/external function, EVERY event, and the constructor
- Types must be exact Solidity types: uint256 (not uint), address, bool, string, bytes32, etc.
- stateMutability must be correct: pure/view/payable/nonpayable
- Indexed event parameters must have "indexed": true
- Preserve all parameter names exactly as in the source code"""
            ),
            user_message(create_abi_generator_task_description(solidity_code)),
        ]

        response = lm.chat(messages=messages)
        abi_text = str(response).strip()

        if "```json" in abi_text:
            abi_text = abi_text.split("```json")[1].split("```")[0].strip()
        elif "```" in abi_text:
            abi_text = abi_text.split("```")[1].split("```")[0].strip()

        return json.loads(abi_text)


# ==================== PROGRAM CLASS 5: MCPServerGeneratorProgram ====================


class MCPServerGeneratorProgram(Program):
    """
    IBM Agentics Program for generating custom MCP servers from ABI files
    """

    def forward(
        self,
        abi: List[Dict],
        schema: UniversalContractSchema,
        contract_name: str,
        lm: LLM,
    ) -> str:
        """
        Generate custom MCP server code from ABI

        Args:
            abi: The contract ABI
            schema: The contract schema (for context)
            contract_name: Name of the contract file
            lm: Language model

        Returns:
            str: Complete Python code for MCP server
        """

        # Extract function information from ABI
        functions = [item for item in abi if item.get("type") == "function"]
        payable_functions = [
            f for f in functions if f.get("stateMutability") == "payable"
        ]
        nonpayable_functions = [
            f for f in functions if f.get("stateMutability") == "nonpayable"
        ]
        view_functions = [
            f for f in functions if f.get("stateMutability") in ["view", "pure"]
        ]

        # Get constructor for understanding contract initialization
        constructor = next(
            (item for item in abi if item.get("type") == "constructor"), None
        )

        # Create detailed function descriptions
        function_details = self._create_function_descriptions(functions, schema)

        messages = [
            system_message(
                """You are an expert Python developer specializing in blockchain integration and MCP servers.

                You understand:
                - Web3.py for Ethereum interaction
                - FastMCP (v0.7+) for creating MCP tool servers using FastMCP class
                - Smart contract function calling patterns
                - Transaction signing and gas management
                - Error handling for blockchain operations

                CRITICAL: Use FastMCP (not the old MCP class). The correct import is:
                  from fastmcp import FastMCP
                  mcp = FastMCP("ContractName")
                  @mcp.tool()
                  def function_name():
                      ...

                The main block must end with:
                  if __name__ == "__main__":
                      mcp.run()

                You write clean, well-documented, production-ready code."""
            ),
            user_message(
                f"""Generate a complete MCP server for this {schema.contract_type} smart contract.

CONTRACT NAME: {contract_name}
CONTRACT TYPE: {schema.contract_type}
PARTIES: {[f"{p.name} ({p.role})" for p in schema.parties]}

ABI SUMMARY:
- Payable functions: {len(payable_functions)}
- Non-payable functions: {len(nonpayable_functions)}
- View functions: {len(view_functions)}

COMPLETE ABI:
{json.dumps(abi, indent=2)}

FUNCTION DETAILS:
{function_details}

Generate a Python MCP server file with CORRECT FastMCP API:

1. **Imports (MUST use FastMCP, not old MCP)**:
   ```python
   import os
   import json
   from pathlib import Path
   from web3 import Web3
   from dotenv import load_dotenv
   from fastmcp import FastMCP
   ```

2. **Setup (Load .env from SAME DIRECTORY as script, Load ABI from .abi.json file)**:
   ```python
   import os
   import json
   from pathlib import Path
   from dotenv import load_dotenv

   # Load .env from the same directory as this script
   env_path = Path(__file__).parent / '.env'
   load_dotenv(dotenv_path=env_path)

   # Load ABI from the same directory as this script
   abi_path = Path(__file__).parent / '{contract_name}.abi.json'
   with open(abi_path, 'r') as f:
       contract_abi = json.load(f)

   RPC_URL = os.getenv('RPC_URL')
   PRIVATE_KEY = os.getenv('PRIVATE_KEY')
   CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')

   web3 = Web3(Web3.HTTPProvider(RPC_URL))
   account = web3.eth.account.from_key(PRIVATE_KEY)
   contract = web3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=contract_abi)
   ```

3. **Create FastMCP instance**:
   ```python
   mcp = FastMCP("{contract_name}")
   ```

4. **Create @mcp.tool() decorated functions for EACH ABI function**:

   For PAYABLE functions:
   - Build transaction with correct value
   - Sign and send transaction
   - Return {{"tx_hash": hash}}

   For NON-PAYABLE functions:
   - Build transaction (no value)
   - Sign and send transaction
   - Return {{"tx_hash": hash}}

   For VIEW functions:
   - Call function (read-only)
   - Return result directly

5. **Documentation**:
   - Each @mcp.tool() function must have docstring
   - Explain what it does, who can call it, parameters, return value
   - Handle errors gracefully

6. **Error Handling**:
   - Wrap each tool in try/except
   - Return {{"error": str(e)}} on failure

7. **Main Block (CRITICAL)**:
   ```python
   if __name__ == "__main__":
       mcp.run()
   ```

CRITICAL CODE EXAMPLES:

Setup must use local .env file:
```python
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
```

Transaction building for non-payable:
```python
txn = contract.functions.function_name(param1).buildTransaction({{
    'from': account.address,
    'nonce': web3.eth.get_transaction_count(account.address),
    'gas': 2000000,
    'gasPrice': web3.to_wei('20', 'gwei')
}})
signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
return {{"tx_hash": tx_hash.hex()}}
```

Transaction building for payable:
```python
txn = contract.functions.makePayment().buildTransaction({{
    'from': account.address,
    'nonce': web3.eth.get_transaction_count(account.address),
    'gas': 2000000,
    'gasPrice': web3.to_wei('20', 'gwei'),
    'value': web3.to_wei(5, 'ether')
}})
signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
return {{"tx_hash": tx_hash.hex()}}
```

View function:
```python
result = contract.functions.getBalance().call()
return {{"result": result}}
```

IMPORTANT RULES:
- Function names must match ABI exactly
- Include ALL functions from ABI
- Use @mcp.tool() decorator (not @tool())
- Load ABI from {contract_name}.abi.json file in same directory (do NOT hardcode ABI)
- Load .env from same directory as script
- ALWAYS initialize: account = web3.eth.account.from_key(PRIVATE_KEY)
- Use 'from': account.address in all transactions
- Use web3.eth.get_transaction_count(), NOT web3.eth.getTransactionCount()
- Use web3.to_wei(), NOT web3.toWei()
- Use web3.eth.send_raw_transaction(), NOT web3.eth.sendRawTransaction()
- Use Web3.to_checksum_address(), NOT Web3.toChecksumAddress()
- For payable functions, include 'value': web3.to_wei(amount, 'ether')
- File must be self-contained and runnable

Return ONLY the complete Python code, no explanations."""
            ),
        ]

        response = lm.chat(messages=messages)
        server_code = str(response).strip()

        # Clean markdown
        if "```python" in server_code:
            server_code = server_code.split("```python")[1].split("```")[0].strip()
        elif "```" in server_code:
            server_code = server_code.split("```")[1].split("```")[0].strip()

        return server_code

    def _create_function_descriptions(
        self, functions: List[Dict], schema: UniversalContractSchema
    ) -> str:
        """Create human-readable descriptions of functions based on contract context"""

        descriptions = []

        for func in functions:
            name = func.get("name", "unknown")
            inputs = func.get("inputs", [])
            outputs = func.get("outputs", [])
            stateMutability = func.get("stateMutability", "nonpayable")

            # Create parameter description
            params = ", ".join(
                [
                    f"{inp.get('name', 'param')}:{inp.get('type', 'unknown')}"
                    for inp in inputs
                ]
            )
            returns = (
                ", ".join(
                    [
                        f"{out.get('name', 'result')}:{out.get('type', 'unknown')}"
                        for out in outputs
                    ]
                )
                if outputs
                else "void"
            )

            descriptions.append(f"  - {name}({params}) → {returns} [{stateMutability}]")

        return "\n".join(descriptions)


__all__ = [
    "UniversalContractParserProgram",
    "UniversalSolidityGeneratorProgram",
    "SecurityAuditorProgram",
    "ABIGeneratorProgram",
    "MCPServerGeneratorProgram",
]
