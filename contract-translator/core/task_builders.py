"""
Task description builders for CrewAI agents

These functions create detailed task descriptions for each phase:
- Parser: Extract structured data from contract text
- Generator: Create Solidity smart contract
- Auditor: Security analysis
- ABI Generator: Generate contract ABI
"""

from .schemas import UniversalContractSchema


def create_parser_task_description(contract_text: str) -> str:
    """
    Create task description for the Contract Parser Agent.
    Returns the full task description including the contract text.
    Uses the comprehensive prompt from programs.py for maximum accuracy.
    """
    return f"""Analyze this contract and extract ALL SPECIFIC information exactly as mentioned.

CONTRACT TEXT:
{contract_text}

CRITICAL INSTRUCTIONS:
1. Extract the EXACT function names mentioned in the contract (e.g., "initializeLease", "payRent", "confirmDelivery")
2. Extract the EXACT variable names mentioned (e.g., "monthlyRent", "securityDeposit", "deliveryDate")
3. Extract the EXACT state names mentioned (e.g., "Pending", "Active", "Completed", "Terminated")
4. Extract the EXACT party roles as described in the contract
5. DO NOT use generic placeholders - use the specific terminology from the contract
6. Capture ALL conditions, transitions, and logic flows mentioned

Your goal: Create a structured representation that preserves ALL specific details from the contract text.

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


def create_solidity_generation_prompt(schema):
    conditions = schema.conditions if schema.conditions else {}
    function_names = conditions.get("function_names", [])
    variable_names = conditions.get("variable_names", [])
    state_names = conditions.get("state_names", [])
    state_transitions = conditions.get("state_transitions", [])
    events = conditions.get("events", [])
    logic_conditions = conditions.get("logic_conditions", [])
    # Detect contract type from type field and function names
    ct = (schema.contract_type or "").lower()
    fn_lower = [f.lower() for f in function_names]

    is_token = "token" in ct or any(
        f in fn_lower
        for f in [
            "transfer",
            "approve",
            "transferfrom",
            "balanceof",
            "totalsupply",
            "mint",
            "burn",
        ]
    )
    is_governance = (
        "governance" in ct
        or "voting" in ct
        or any(
            f in fn_lower
            for f in [
                "delegate",
                "vote",
                "propose",
                "execute",
                "getpriorvotes",
                "castballot",
            ]
        )
    )
    is_escrow = (
        "escrow" in ct
        or "payment" in ct
        or "lending" in ct
        or "loan" in ct
        or any(
            f in fn_lower
            for f in ["deposit", "release", "refund", "repay", "collateral"]
        )
    )
    is_marketplace = (
        "marketplace" in ct
        or "auction" in ct
        or "nft" in ct
        or "sale" in ct
        or any(f in fn_lower for f in ["list", "bid", "buy", "purchase", "auction"])
    )
    is_staking = (
        "staking" in ct
        or "reward" in ct
        or "yield" in ct
        or any(f in fn_lower for f in ["stake", "unstake", "claim", "harvest"])
    )

    # Type-specific mandatory requirements
    type_rules = ""
    if is_token:
        type_rules = """
TOKEN CONTRACT — MANDATORY (every item below is required, omitting any = broken contract):
- ERC20 interface: transfer(address,uint256), approve(address,uint256), transferFrom(address,address,uint256), balanceOf(address) view, allowance(address,address) view, totalSupply() view
- State variables: string public name, string public symbol, uint8 public decimals, uint256 public totalSupply, mapping(address=>uint256) public balances, mapping(address=>mapping(address=>uint256)) public allowances
- Events: Transfer(address indexed from, address indexed to, uint256 value), Approval(address indexed owner, address indexed spender, uint256 value)
- transfer/approve/transferFrom MUST be callable by any user simultaneously — do NOT put them behind a state machine phase
- Internal _transfer() helper that validates balances and updates both sender and recipient atomically
- If minting exists: track totalSupply, emit Transfer(address(0), to, amount)
- If burning exists: track totalSupply, emit Transfer(from, address(0), amount)
"""
    if is_governance:
        type_rules += """
GOVERNANCE/DELEGATION — MANDATORY:
- mapping(address=>address) public delegates — who each account delegates voting power to
- mapping(address=>uint256) public votingPower — actual current voting power (separate from balance)
- delegate() must MOVE power: subtract from old delegate's votingPower, add to new delegate's votingPower
- _afterTokenTransfer() internal: when tokens move, update votingPower for both sender's and receiver's current delegates
- Checkpoints: store an array of (uint256 fromBlock, uint256 votes) per account — not a single value
- getPriorVotes(address, uint256 blockNumber) must binary-search checkpoints for historical data
- Invariant: sum of all votingPower == sum of all balances at all times
"""
    if is_escrow:
        type_rules += """
ESCROW/PAYMENT — MANDATORY:
- Track deposits per depositor: mapping(address=>uint256) public deposits (or per-deal struct)
- Funds enter via payable function; funds leave via (bool ok,) = recipient.call{value: amount}(""); require(ok)
- Implement conditional release: require all conditions are met before releasing
- Implement refund path: if conditions fail or deadline expires, depositors can reclaim
- Enforce: contract.balance == sum of all unreleased deposits at all times
"""
    if is_marketplace:
        type_rules += """
MARKETPLACE/AUCTION — MANDATORY:
- Each listing/auction stored in a struct with: seller, price/minBid, deadline, status
- Bidding: track highestBid and highestBidder; refund previous highest bidder on outbid
- Purchase/settlement: transfer ownership AND funds atomically
- Fee calculation: if royalties/fees mentioned, compute as basis points (e.g. 250 = 2.5%) and distribute
- Prevent sniping if mentioned: extend deadline on late bids
"""
    if is_staking:
        type_rules += """
STAKING/REWARDS — MANDATORY:
- Track per-staker: mapping(address=>uint256) public staked, mapping(address=>uint256) public rewardDebt
- Reward accrual: use a rewardPerShare accumulator updated on every stake/unstake/claim
- At minimum: stake(uint256), unstake(uint256), claimRewards(), pendingRewards(address) view
- Early withdrawal penalty: if mentioned, calculate and transfer penalty to treasury/burn
- Correct accounting: rewards already paid to a staker must not be paid again (rewardDebt pattern)
"""

    # Temporal/business logic section
    temporal_lines = []
    if schema.dates:
        for d in schema.dates:
            temporal_lines.append(
                f"- {d.date_type}: store as uint256, enforce with require(block.timestamp ...) — "
                f"behavior MUST differ before vs after this date"
            )
    if logic_conditions:
        for lc in logic_conditions:
            temporal_lines.append(
                f"- Business rule: {lc} → implement as require() check or modifier using actual state/mappings"
            )
    if schema.financial_terms:
        for t in schema.financial_terms:
            freq = f", recurring {t.frequency}" if t.frequency else ", one-time"
            temporal_lines.append(
                f"- Financial: {t.purpose} {t.amount} {t.currency}{freq} → "
                f"implement full on-chain transfer with balance tracking and emit event"
            )

    temporal_section = ""
    if temporal_lines:
        temporal_section = (
            "\nTEMPORAL AND FINANCIAL LOGIC TO IMPLEMENT:\n"
            + "\n".join(temporal_lines)
            + "\n"
        )

    # Obligations
    obligation_lines = ""
    if schema.obligations:
        obligation_lines = "\n".join(
            f"- {o.party}: {o.description}"
            + (f" [deadline: {o.deadline}]" if o.deadline else "")
            + (f" [penalty: {o.penalty_for_breach}]" if o.penalty_for_breach else "")
            for o in schema.obligations
        )
    else:
        obligation_lines = (
            "Derive obligations from function_names and contract purpose above."
        )

    # Party access control
    party_lines = ""
    if schema.parties:
        party_lines = "\n".join(
            f"- {p.name} ({p.role}): store as address state variable, write onlyX modifier, apply to appropriate functions"
            for p in schema.parties
        )
    else:
        party_lines = "At minimum: address public owner with onlyOwner modifier."

    return f"""You are an expert Solidity engineer. Generate a COMPLETE, PRODUCTION-QUALITY Solidity ^0.8.0 smart contract that fully and correctly implements the specification below.

═══════════════════════════════════════
CONTRACT SPECIFICATION
═══════════════════════════════════════
{schema.model_dump_json(indent=2)}

EXTRACTED SCHEMA FIELDS:
- Functions:    {', '.join(function_names) if function_names else 'derive from obligations and contract type'}
- Variables:    {', '.join(variable_names) if variable_names else 'derive from financial terms and parties'}
- States:       {', '.join(state_names) if state_names else 'NONE — do NOT invent an enum state machine; implement direct access control and logic only'}
- Transitions:  {', '.join(state_transitions) if state_transitions else 'NONE — no state transitions required; use modifiers/require() for access control only'}
- Events:       {', '.join(events) if events else 'one per function action'}
- Rules:        {'; '.join(logic_conditions) if logic_conditions else 'derive from special_terms and obligations'}

OBLIGATIONS (implement each as a complete function):
{obligation_lines}

PARTIES AND ACCESS CONTROL:
{party_lines}
{type_rules}{temporal_section}
═══════════════════════════════════════
GENERATION PRINCIPLES
═══════════════════════════════════════

1. UNDERSTAND THE BUSINESS LOGIC, DO NOT JUST MATCH NAMES
   The biggest failure mode is implementing function signatures with hollow bodies.
   Before writing each function, determine:
   - What real-world operation does this represent?
   - What invariant must hold before AND after this call?
   - What state changes, who gains/loses what, and what can go wrong?
   Examples of correct reasoning:
   - "transfer" → sender balance decreases by X, recipient increases by X, X never appears twice (atomic)
   - "release escrow" → verify release conditions, mark released, send exact escrowed amount to beneficiary, cannot release twice
   - "complete milestone" → mark milestone struct as complete, check if all milestones done, if so auto-release remaining funds
   - "bid in auction" → new bid > current highest, lock new bidder's ETH, refund previous highest bidder, update state
   If you are assigning a spec field to a variable without using it in any transfer or decision, you are doing it wrong.

2. WRITE LONG, COMPLETE CONTRACTS
   Length is a virtue here. Include every helper that makes correctness provable:
   - Internal _transfer(), _approve(), _mint(), _burn() helpers
   - Private calculation functions (_calculateFee, _computeReward, _checkVested)
   - Explicit getter functions for every important mapping or struct field
   - Full input validation on every external function (zero-address, range, balance, state checks)
   - Descriptive revert messages on every require()
   - NatSpec comments (@notice, @param, @return) on all external/public functions
   A 300-line contract with correct business logic is far better than a 60-line stub.

3. SEMANTIC DOMAIN FIDELITY
   Domain terms carry full semantic weight:
   - "Token" → complete ERC20 with transfer hooks, allowance flow, supply tracking
   - "Vesting" → cliff timestamp, linear release calculation, released-amount tracking, clawback if mentioned
   - "Milestone" → struct per milestone with (amount, completed, paid) fields, sequential or parallel completion logic
   - "Royalty" → basis-point percentage applied to every qualifying transfer, distributed to rights holder automatically
   - "Delegation" → voting power physically moves between accounts, cascades when tokens transfer
   - "Auction" → time-bounded, refunds outbid participants, handles ties per spec, settles atomically
   Do not reduce these to a counter or a flag — implement the full domain semantics.

4. TEMPORAL LOGIC MUST CHANGE BEHAVIOR
   Every time-related variable must:
   - Be stored as uint256 (Unix timestamp or block number)
   - Be compared to block.timestamp or block.number in a require() or modifier
   - Produce genuinely different outcomes before vs after the boundary
   If a deadline is stored but never compared, or compared but behavior is identical either side, it is a bug.
   Implement both the "before deadline" path AND the "after deadline" path explicitly.

5. ECONOMIC INVARIANTS MUST HOLD UNDER ANY CALL SEQUENCE
   Identify the core balance invariant for this contract and enforce it:
   - Tokens: sum(all balances) == totalSupply after every function
   - Escrow: contract.balance == sum(unreleased deposits) after every function
   - Auction: contract.balance >= highestBid after every function
   Write the invariant as a comment above the contract and design all functions to preserve it.
   Test mentally: "If I call these functions in a hostile order, can I steal funds or mint tokens for free?"

6. STATES = LIFECYCLE PHASES ONLY (and ONLY when the spec requires them)
   IF the spec lists explicit state names: implement them as a Solidity enum representing lifecycle stages.
   States must represent lifecycle phases (Pending → Active → Completed → Cancelled).
   Never name a state after an operation (Transfer, Approve, Deposit).
   Every state must be SET in at least one function and CHECKED in at least one modifier or require().
   Terminal states (Completed, Cancelled, Expired) must be reachable.
   For tokens: do NOT block transfer/approve behind a state machine — users must be able to call them at all times.
   IF the spec has NO explicit state names (States: NONE): do NOT add an enum state machine.
   For state-less contracts, use access control modifiers (onlyOwner, etc.) and inline require() checks directly — do not fabricate lifecycle states that were never asked for.

7. ACCESS CONTROL IS NON-NEGOTIABLE
   Every function that changes critical parameters, transfers funds, or mints/burns tokens must have a modifier.
   Derive roles from the parties in the spec. Use least-privilege.
   Every modifier must contain a real require() that reverts on failure.

8. NO DECORATIVE CODE
   Every state variable must be written by ≥1 function and read by ≥1 function.
   Every function must either change state or transfer value or return computed data.
   Use require(condition, "message") — never if(condition) return.
   Emit events only after successful state/value changes; include all relevant parameters.

9. AVOID IDENTIFIER SHADOWING IN PARAMETERS
   NEVER name a function parameter the same as a contract-level state variable.
   Common offenders that WILL cause "Identifier already declared" compile errors:
   - Don't use `owner` as a parameter — use `tokenOwner`, `account`, or `addr` instead.
   - Don't use `spender`, `sender`, `recipient` if you also have state variables of those names.
   - Don't use `amount`, `balance`, `value` as parameter names in a contract that has state variables with those names.
   Check every parameter name against the list of state variables before writing the function signature.

10. AVOID COLLISIONS BETWEEN PUBLIC STATE VARIABLES AND INTERFACE FUNCTIONS
    If a contract implements an interface (e.g. IERC20), NEVER declare a public state variable with
    the same name as an interface function. The automatically-generated getter will clash.
    Correct patterns:
    - WRONG: `uint256 public totalSupply;` + implementing `function totalSupply() external view returns (uint256)`
      → TWO declarations of the same identifier — compile error.
    - RIGHT:  `uint256 private _totalSupply;` and return it from the required interface function.
    - WRONG: `mapping(address => uint256) public balances;` if the interface also has `function balances(address) external view returns (uint256)`.
    - RIGHT:  Use private/internal storage variables (prefixed with `_`) and expose them only via the required interface functions.
    Apply this rule for ALL interface methods (`totalSupply`, `balanceOf`, `allowance`, etc.).

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════

Return ONLY Solidity code. No explanations, no markdown fences.

Structure your contract in this order:
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

1. Interfaces (if implementing a standard like ERC20)
2. Contract declaration with NatSpec @title/@notice
3. Enums (lifecycle states, if needed)
4. Structs (for complex resources)
5. State variables (all used, grouped: access control, state, economics, time, domain-specific)
6. Events (one per distinct action, indexed params)
7. Modifiers
8. Constructor (initialize all state, assign roles, set initial phase)
9. External/public functions (complete logic — validation → state update → transfer → event)
10. Internal helper functions
11. View/pure getter functions

Aim for 150–400 lines. Completeness and correctness over brevity.
"""


def create_mcp_task_description(abi, schema, contract_name: str) -> str:
    """
    Create task description for the MCP Server Generator Agent.
    Uses the comprehensive prompt from programs.py.
    """
    # Extract function information from ABI
    functions = [item for item in abi if item.get("type") == "function"]
    payable_functions = [f for f in functions if f.get("stateMutability") == "payable"]
    nonpayable_functions = [
        f for f in functions if f.get("stateMutability") == "nonpayable"
    ]
    view_functions = [
        f for f in functions if f.get("stateMutability") in ["view", "pure"]
    ]

    import json

    return f"""Generate a complete MCP server for this {schema.contract_type} smart contract.

CONTRACT NAME: {contract_name}
CONTRACT TYPE: {schema.contract_type}
PARTIES: {[f"{p.name} ({p.role})" for p in schema.parties]}

ABI SUMMARY:
- Payable functions: {len(payable_functions)}
- Non-payable functions: {len(nonpayable_functions)}
- View functions: {len(view_functions)}

COMPLETE ABI:
{json.dumps(abi, indent=2)}

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


# Alias for backward compatibility
def create_solidity_generator_task_description(schema):
    """
    Alias for create_solidity_generation_prompt for backward compatibility.
    """
    return create_solidity_generation_prompt(schema)


def create_audit_task_description(solidity_code: str) -> str:
    """
    Create task description for the Security Auditor Agent.
    """
    return f"""Perform a comprehensive security audit on this Solidity smart contract:

{solidity_code}

SYSTEMATIC AUDIT CHECKLIST - Check each category:

1. **REENTRANCY ATTACKS**:
   - Look for external calls (call, transfer, send, delegatecall) followed by state changes
   - Check if state is updated BEFORE external calls (Checks-Effects-Interactions pattern)
   - Verify reentrancy guards (nonReentrant modifier) on sensitive functions

2. **ACCESS CONTROL**:
   - Verify onlyOwner or role-based modifiers on critical functions (withdraw, changeOwner, etc.)
   - Check if constructor properly initializes owner
   - Look for functions that should be internal/private but are public/external

3. **ARITHMETIC SAFETY**:
   - Check for unchecked arithmetic that could overflow/underflow
   - Verify SafeMath usage or Solidity ^0.8.0 built-in checks
   - Look for division by zero possibilities

4. **ETHER HANDLING**:
   - Check payable functions have proper access control
   - Verify withdraw/transfer functions validate amounts and recipients
   - Look for locked ether (payable functions with no withdrawal mechanism)

5. **DOS VULNERABILITIES**:
   - Look for unbounded loops that could hit gas limits
   - Check for external calls in loops
   - Verify functions can't be blocked by reverting recipients

6. **INPUT VALIDATION**:
   - Check require() statements validate all critical parameters
   - Verify address parameters check for address(0)
   - Ensure amount checks prevent zero or negative values

7. **TIMESTAMP DEPENDENCE**:
   - Check if block.timestamp is used for critical logic
   - Verify it's not used for randomness or precise timing

8. **EXTERNAL CALL SAFETY**:
   - Verify return values of external calls are checked
   - Check low-level calls (call, delegatecall) have proper error handling

Return ONLY valid JSON with specific findings:
{{
    "severity_level": "none|low|medium|high|critical",
    "approved": boolean (true if severity is none/low, false for medium/high/critical),
    "issues": [
        "SPECIFIC issue with line reference and exploit scenario",
        "Example: HIGH: Reentrancy in withdraw() - external call before balance update allows recursive calls"
    ],
    "recommendations": [
        "SPECIFIC remediation step, not generic advice",
        "Example: Move 'balances[msg.sender] = 0' BEFORE 'msg.sender.call{{value: amount}}()'"
    ],
    "vulnerability_count": number (total count of issues found),
    "security_score": "A|B|C|D|F" (A = no issues, B = only low, C = medium, D = high, F = critical)
}}

Be specific about WHERE issues are and HOW to fix them. Reference actual function names and variables from the code."""


def create_abi_generator_task_description(solidity_code: str) -> str:
    """
    Create task description for the ABI Generator Agent.
    """
    return f"""Generate the complete, accurate ABI (Application Binary Interface) for this Solidity contract:

{solidity_code}

REQUIREMENTS - Extract ALL of these:

1. **CONSTRUCTOR**:
   - Include ALL constructor parameters with exact types (address, uint256, string, etc.)
   - stateMutability should be "nonpayable" unless constructor is payable
   - type: "constructor"

2. **ALL PUBLIC/EXTERNAL FUNCTIONS**:
   - Extract function name exactly as written
   - Include ALL parameters with correct types and names
   - Include ALL outputs with correct types
   - Set stateMutability: "pure" (no state read/write), "view" (reads state), "payable" (accepts ETH), or "nonpayable" (default)
   - type: "function"

3. **ALL EVENTS**:
   - Extract event name exactly as written
   - Include ALL parameters with correct types and names
   - Mark indexed parameters with "indexed": true
   - type: "event"

4. **TYPE ACCURACY**:
   - Use exact Solidity types: uint256 (not uint), address, bool, string, bytes, bytes32, etc.
   - For arrays: uint256[], address[], etc.
   - For mappings in public vars: treat as getter function
   - For enums: use uint8
   - For structs: expand to individual fields if returned

5. **PARAMETER NAMES**:
   - Preserve parameter names from code (critical for debugging)
   - Use empty string "" only if parameter has no name in code

VALIDATION:
- Every public/external function must be included
- Every event must be included
- Constructor must be included if it exists
- Types must match Solidity declarations EXACTLY
- Output must be valid JSON array

Return ONLY the JSON array (no markdown, no explanation):
[
  {{
    "type": "constructor",
    "stateMutability": "nonpayable",
    "inputs": [...]
  }},
  {{
    "type": "function",
    "name": "functionName",
    "stateMutability": "view|pure|payable|nonpayable",
    "inputs": [...],
    "outputs": [...]
  }},
  {{
    "type": "event",
    "name": "EventName",
    "inputs": [
      {{"name": "param", "type": "uint256", "indexed": true}}
    ]
  }}
]"""


def create_quality_evaluation_task_description(
    solidity_code: str, schema, contract_name: str
) -> str:
    """
    Create task description for the Quality Evaluator Agent.
    Performs comprehensive multi-metric evaluation of generated contract quality.
    """
    # Handle both Pydantic models and dictionaries
    if isinstance(schema, dict):
        conditions = schema.get("conditions", {})
        schema_dict = schema
    else:
        conditions = (
            schema.conditions
            if hasattr(schema, "conditions") and schema.conditions
            else {}
        )
        schema_dict = (
            schema.model_dump() if hasattr(schema, "model_dump") else schema.dict()
        )

    function_names = conditions.get("function_names", [])
    variable_names = conditions.get("variable_names", [])
    state_names = conditions.get("state_names", [])
    state_transitions = conditions.get("state_transitions", [])
    events = conditions.get("events", [])
    logic_conditions = conditions.get("logic_conditions", [])

    import json

    return f"""Perform a comprehensive quality evaluation of this generated Solidity smart contract against the original natural language specification.

═══════════════════════════════════════════════════════════════════════════════
GENERATED SOLIDITY CONTRACT
═══════════════════════════════════════════════════════════════════════════════

{solidity_code}

═══════════════════════════════════════════════════════════════════════════════
ORIGINAL CONTRACT SPECIFICATION
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(schema_dict, indent=2)}

EXTRACTED KEY ELEMENTS:
- Expected Functions: {function_names if function_names else 'Not specified'}
- Expected Variables: {variable_names if variable_names else 'Not specified'}
- Expected States: {state_names if state_names else 'Not specified'}
- State Transitions: {state_transitions if state_transitions else 'Not specified'}
- Expected Events: {events if events else 'Not specified'}
- Logic Conditions: {logic_conditions if logic_conditions else 'Not specified'}

═══════════════════════════════════════════════════════════════════════════════
EVALUATION INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

Evaluate the contract across FIVE dimensions. For each, provide:
1. A numerical score (0-100) - BE PRECISE! Use exact scores like 73, 81, 92, not just multiples of 5
2. Detailed breakdown showing what earned/lost points
3. Specific evidence from the code with line references

**CRITICAL: Scores must reflect actual points earned/lost. If you calculate 73 points, score is 73, NOT 75.**

═══════════════════════════════════════════════════════════════════════════════
METRIC 1: FUNCTIONAL COMPLETENESS (0-100 points)
═══════════════════════════════════════════════════════════════════════════════

**Calculate exact score based on points earned. Example: 4 functions × 10pts = 40, quality penalties -7 = 33.**

**Scoring Rules:**

A. Function Name Matching (Max 50 points):
   - For each expected function in specification:
     * Exact match found in code: +10 points
     * Semantic match (similar name/purpose): +7 points
     * Missing completely: -10 points
   - List each expected function and whether it was found

B. Function Implementation Quality (Max 50 points):
   - For each implemented function, check:
     * Has complete logic (not placeholder/TODO): +5 points
     * Has proper access control (modifiers/require): +3 points
     * Emits appropriate events: +2 points
     * Has input validation with require(): +2 points
   - Provide specific examples of good/bad implementations

**Output Format:**
```json
{{
  "function_matching": {{
    "expected_functions": ["list from spec"],
    "found_exact": ["functions with exact matches"],
    "found_semantic": ["functions with semantic matches"],
    "missing": ["functions not found"],
    "unexpected": ["functions not in spec"],
    "points": 0-50
  }},
  "implementation_quality": {{
    "complete_logic": ["functions with full implementation"],
    "incomplete_logic": ["functions with placeholders"],
    "proper_access_control": ["functions with modifiers/checks"],
    "missing_access_control": ["functions needing protection"],
    "event_emissions": ["functions emitting events"],
    "missing_events": ["functions not emitting events"],
    "input_validation": ["functions with require checks"],
    "missing_validation": ["functions needing validation"],
    "points": 0-50
  }},
  "score": 0-100,
  "evidence": ["specific code examples with line numbers"]
}}
```

═══════════════════════════════════════════════════════════════════════════════
METRIC 2: VARIABLE/PARAMETER FIDELITY (0-100 points)
═══════════════════════════════════════════════════════════════════════════════

**Calculate exact score based on points earned. Example: 8 vars × 10pts = 80, 2 wrong types × -5pts = 70, NOT 75.**

**Scoring Rules:**

A. State Variable Completeness (Max 60 points):
   - For each expected variable in specification:
     * Variable declared: +10 points
     * Correct type (uint256 for amounts, address for parties, etc.): +5 points
     * Actually used in logic (written AND read): +5 points
     * Missing or decorative only: -10 points

B. Function Parameter Quality (Max 40 points):
   - Check representative functions:
     * Correct parameter count: +10 points
     * Correct parameter types: +10 points
     * Descriptive parameter names: +10 points
     * Parameters actually used in function: +10 points

**Output Format:**
```json
{{
  "state_variables": {{
    "expected_variables": ["list from spec"],
    "declared": ["variables found in contract"],
    "correct_types": ["variables with correct types"],
    "actively_used": ["variables used in logic"],
    "decorative_only": ["declared but never used"],
    "missing": ["expected but not found"],
    "points": 0-60
  }},
  "function_parameters": {{
    "functions_checked": ["sample of functions analyzed"],
    "correct_count": ["functions with right param count"],
    "correct_types": ["functions with right param types"],
    "descriptive_names": ["functions with good naming"],
    "parameters_used": ["functions using all params"],
    "points": 0-40
  }},
  "score": 0-100,
  "evidence": ["specific examples from code"]
}}
```

═══════════════════════════════════════════════════════════════════════════════
METRIC 3: STATE MACHINE CORRECTNESS (0-100 points)
═══════════════════════════════════════════════════════════════════════════════

**FIRST: Determine which scoring path applies based on the spec's `state_names` field.**

─── PATH A: SPEC HAS EXPLICIT STATES (state_names is non-empty) ───────────────
Use this path ONLY if the specification lists explicit state names (e.g. Pending, Active, Completed).

A. State Definition (Max 25 points):
   - All expected states present in enum: +15 points
   - States used in state variable declaration: +10 points
   - Extra unnecessary states: -5 points each

B. State Transitions (Max 50 points):
   - For each expected transition:
     * Transition implemented in code: +10 points
     * Has proper require() check: +10 points
     * Matches specification logic: +10 points
   - Missing transitions: -10 points each
   - Invalid transitions possible (unauthorized skips): -10 points each

C. State Guards (Max 25 points):
   - Functions use state-based modifiers: +10 points
   - Functions use state require() checks: +10 points
   - No state bypass vulnerabilities: +5 points

─── PATH B: SPEC HAS NO EXPLICIT STATES (state_names is empty/null) ─────────
Use this path if the specification does NOT list any state names.
DO NOT invent lifecycle states ("Token Creation", "Active Trading", "Transfer", "Approval" etc.).
DO NOT penalize the contract for lacking an enum — no enum is CORRECT here.

A. Correctness of No-State Design (Max 40 points):
   - Contract does NOT have an unnecessary enum state machine: +20 points
     (if contract DOES have an enum with fabricated states like "TokenCreation": -15 points)
   - No functions are gated behind an enum state that was not in the spec: +20 points

B. Access Control as Transition Substitute (Max 35 points):
   - onlyOwner or role-based modifiers present for admin functions: +15 points
   - Inline require() checks enforce who can call what: +10 points
   - Events emitted to log major state-changing operations: +10 points

C. Guard Quality (Max 25 points):
   - All external/public functions have appropriate access control: +10 points
   - Input/boundary validation (zero-address, range checks, balance checks): +10 points
   - No privilege escalation vulnerabilities: +5 points

**Scoring Summary:**
Path A max = 100. Path B max = 100.
In both paths, score must reflect actual arithmetic — do not round to nearest 5.

**Output Format (use regardless of path):**
```json
{{
  "state_machine_required": true/false,
  "scoring_path": "A (explicit states)" or "B (no states in spec)",
  "state_definition": {{
    "expected_states": ["states from spec, or [] if none"],
    "defined_states": ["states in enum in code"],
    "state_variable_exists": true/false,
    "extra_states": ["fabricated states not in spec"],
    "points": 0-25 (Path A) or 0-40 (Path B)
  }},
  "state_transitions": {{
    "expected_transitions": ["from spec, or [] if none"],
    "implemented_correctly": ["transitions with proper logic"],
    "missing_transitions": ["expected but not found"],
    "invalid_transitions_possible": ["security issues"],
    "points": 0-50 (Path A) or 0-35 (Path B)
  }},
  "state_guards": {{
    "functions_with_modifiers": ["functions using modifiers"],
    "functions_with_checks": ["functions using require()"],
    "bypass_vulnerabilities": ["issues found"],
    "points": 0-25
  }},
  "score": 0-100,
  "evidence": ["specific code examples with line references"]
}}
```

═══════════════════════════════════════════════════════════════════════════════
METRIC 4: BUSINESS LOGIC FIDELITY (0-100 points) - MOST IMPORTANT
═══════════════════════════════════════════════════════════════════════════════

**Calculate exact score. Example: obligations 23pts + financial 19pts + temporal 8pts = 50.**

**Scoring Rules:**

A. Obligation Implementation (Max 30 points):
   - For each obligation in specification:
     * Obligation mapped to function: +10 points
     * Logic correctly implements obligation: +10 points
     * Proper enforcement (access control, checks): +10 points

B. Financial Logic (Max 30 points):
   - For each financial term:
     * Payment handling implemented: +10 points
     * Correct amounts/calculations: +10 points
     * Proper fund tracking: +10 points

C. Temporal Logic (Max 20 points):
   - For each date/deadline:
     * Deadline enforcement with block.timestamp: +10 points
     * Time-based behavior changes implemented: +10 points

D. Conditional Logic (Max 20 points):
   - For each logic condition:
     * Condition implemented in code: +10 points
     * Correct logic/calculations: +10 points

**Output Format:**
```json
{{
  "obligation_implementation": {{
    "total_obligations": 0,
    "obligations_with_functions": ["obligation → function mapping"],
    "correct_logic": ["obligations correctly implemented"],
    "missing_obligations": ["obligations not implemented"],
    "improper_enforcement": ["weak or missing checks"],
    "points": 0-30
  }},
  "financial_logic": {{
    "total_financial_terms": 0,
    "payment_handling": ["terms with payment functions"],
    "correct_calculations": ["terms with right amounts"],
    "fund_tracking": ["terms with proper accounting"],
    "missing_financial_logic": ["terms not implemented"],
    "points": 0-30
  }},
  "temporal_logic": {{
    "total_dates": 0,
    "deadline_enforcement": ["dates checked with block.timestamp"],
    "time_based_behavior": ["time-dependent logic implemented"],
    "missing_temporal_logic": ["dates not enforced"],
    "points": 0-20
  }},
  "conditional_logic": {{
    "total_conditions": 0,
    "implemented_conditions": ["conditions in code"],
    "correct_logic": ["conditions with right calculations"],
    "missing_conditions": ["conditions not implemented"],
    "points": 0-20
  }},
  "score": 0-100,
  "evidence": ["specific examples from code and spec"]
}}
```

═══════════════════════════════════════════════════════════════════════════════
METRIC 5: CODE QUALITY (0-100 points)
═══════════════════════════════════════════════════════════════════════════════

**Calculate exact score. Example: placeholders -2pts from 30, events 13pts, structure 17pts = 58.**

**Scoring Rules:**

A. Placeholder Detection (Max 30 points - DEDUCTIONS):
   - Start at 30 points
   - Search for: "// TODO", "// implement", "// logic", empty function bodies
   - Deduct 10 points for each placeholder found (max -30)

B. Error Message Quality (Max 25 points):
   - Count require() statements
   - Score = (requires with messages / total requires) * 25
   - Examples of good vs missing messages

C. Event Quality (Max 20 points):
   - Events match actions (not decorative): +10 points
   - Proper indexing of parameters: +5 points
   - All major actions have events: +5 points

D. Code Structure (Max 15 points):
   - Proper logical grouping: +5 points
   - No redundant code: +5 points
   - Clear naming conventions: +5 points

E. Documentation (Max 10 points):
   - NatSpec comments on functions: +5 points
   - Clear variable names: +5 points

**Output Format:**
```json
{{
  "placeholder_detection": {{
    "placeholders_found": ["list with line numbers"],
    "placeholder_count": 0,
    "points": 0-30
  }},
  "error_messages": {{
    "total_requires": 0,
    "requires_with_messages": 0,
    "requires_missing_messages": ["line numbers"],
    "percentage": 0-100,
    "points": 0-25
  }},
  "event_quality": {{
    "total_events": 0,
    "events_match_actions": ["good events"],
    "decorative_events": ["events without actions"],
    "proper_indexing": ["events with indexed params"],
    "missing_events": ["actions without events"],
    "points": 0-20
  }},
  "code_structure": {{
    "logical_grouping": true/false,
    "redundant_code_found": ["examples"],
    "naming_conventions": "good|fair|poor",
    "points": 0-15
  }},
  "documentation": {{
    "natspec_coverage": 0-100,
    "clear_variable_names": true/false,
    "points": 0-10
  }},
  "score": 0-100,
  "evidence": ["specific examples"]
}}
```

═══════════════════════════════════════════════════════════════════════════════
FINAL OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return ONLY valid JSON (no markdown, no explanation):

{{
  "contract_name": "{contract_name}",
  "evaluation_timestamp": "ISO timestamp",

  "metric_1_functional_completeness": {{
    "score": 0-100,
    "function_matching": {{}},
    "implementation_quality": {{}},
    "evidence": []
  }},

  "metric_2_variable_fidelity": {{
    "score": 0-100,
    "state_variables": {{}},
    "function_parameters": {{}},
    "evidence": []
  }},

  "metric_3_state_machine": {{
    "score": 0-100,
    "state_definition": {{}},
    "state_transitions": {{}},
    "state_guards": {{}},
    "evidence": []
  }},

  "metric_4_business_logic": {{
    "score": 0-100,
    "obligation_implementation": {{}},
    "financial_logic": {{}},
    "temporal_logic": {{}},
    "conditional_logic": {{}},
    "evidence": []
  }},

  "metric_5_code_quality": {{
    "score": 0-100,
    "placeholder_detection": {{}},
    "error_messages": {{}},
    "event_quality": {{}},
    "code_structure": {{}},
    "documentation": {{}},
    "evidence": []
  }},

  "composite_score": {{
    "functional_completeness_weighted": 0-25,
    "variable_fidelity_weighted": 0-15,
    "state_machine_weighted": 0-15,
    "business_logic_weighted": 0-35,
    "code_quality_weighted": 0-10,
    "final_score": 0-100,
    "grade": "A|B|C|D|F"
  }},

  "summary": {{
    "strengths": [
      "For EACH metric that scored >75, provide specific detail: 'Metric X (Score Y): [specific achievement with evidence/line numbers]'",
      "Examples: 'Functional Completeness (87): All 5 required functions implemented with proper signatures (lines 45-120)'",
      "'Variable Fidelity (92): Excellent mapping - all 7 state variables match specification types (address seller at line 12, uint256 price at line 15)'",
      "'State Machine (81): Clear 3-state lifecycle (Created→Active→Completed) with proper guards on state transitions'",
      "'Business Logic (78): Financial calculations correctly implement payment split logic (lines 89-95)'",
      "'Code Quality (88): Zero placeholders found, all 12 require statements have descriptive error messages'"
    ],
    "weaknesses": [
      "For EACH metric that scored <75, provide specific issues: 'Metric X (Score Y): [specific problem with evidence/line numbers]'",
      "Examples: 'Functional Completeness (62): Missing 2 of 5 required functions (refundBuyer, cancelOrder not implemented)'",
      "'Variable Fidelity (58): 4 critical variables missing - no deliveryDeadline, escrowAmount, or dispute-related state variables'",
      "'State Machine (45): No state variable defined - contract lacks lifecycle management, transitions unguarded'",
      "'Business Logic (51): No obligation implementation - missing payment logic, no temporal conditions for deadlines (spec requires 30-day timeout)'",
      "'Code Quality (38): Found 5 TODO placeholders (lines 67, 89, 102, 145, 178), 8 of 15 requires missing error messages'"
    ],
    "critical_gaps": [
      "Must-fix issues that directly impact contract usability/security",
      "Reference specific metric scores and line numbers",
      "Examples: 'Security: No access control on critical functions (lines 45-67) - Business Logic metric affected (scored 42)'",
      "'Missing core functionality: Payment escrow not implemented - Functional Completeness scored only 55'"
    ],
    "recommendation": "ACCEPT|REVISE|REJECT with rationale explaining which metrics need improvement"
  }}
}}

CRITICAL INSTRUCTIONS FOR SUMMARY SECTION:
1. Strengths MUST reference specific metrics by name and their scores (e.g., "Functional Completeness (85)")
2. Weaknesses MUST reference specific metrics by name and their scores (e.g., "State Machine (62)")
3. For weaknesses, be VERY specific: cite line numbers, name missing elements, quantify gaps
4. Strengths should highlight concrete achievements with evidence (e.g., "all 7 functions implemented", "proper error handling on 15/15 requires")
5. DO NOT use generic statements like "No financial logic" - instead say "Business Logic (51): Payment calculation missing (spec lines 23-28 require escrow handling), no withdrawal function implemented"
6. Every strength/weakness should directly map to one of the 5 metrics and explain WHY that metric scored high or low
7. Include line numbers wherever possible for both strengths and weaknesses
8. Critical gaps should identify security issues, missing core functionality, or specification violations

SCORING INSTRUCTIONS:
1. Be thorough but objective - cite specific evidence
2. Use line numbers when referencing code
3. Compare implementation to specification systematically
4. Calculate final_score = (M1 * 0.25) + (M2 * 0.15) + (M3 * 0.15) + (M4 * 0.35) + (M5 * 0.10)
5. Grade: A(90-100), B(80-89), C(70-79), D(60-69), F(<60)
6. Be specific about what is missing vs what is implemented incorrectly
7. Return ONLY the JSON, no additional text"""
