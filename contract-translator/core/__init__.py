"""
Contract translator core package.

Contains:
- schemas: Pydantic data models
- programs: Legacy Program classes
- task_builders: Task description builders for CrewAI
- agents: Agent creation and configuration
- translator: Main translation pipeline
"""

from .schemas import (
    ContractAsset,
    ContractDate,
    ContractObligation,
    ContractParty,
    ContractType,
    FinancialTerm,
    PartyRole,
    UniversalContractSchema,
)

# Legacy Programs depend on the old agentics API which may not be installed
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
from .task_builders import (
    create_abi_generator_task_description,
    create_audit_task_description,
    create_parser_task_description,
    create_quality_evaluation_task_description,
    create_solidity_generator_task_description,
)
from .translator import IBMAgenticContractTranslator

__all__ = [
    # Schemas
    "PartyRole",
    "ContractType",
    "ContractParty",
    "FinancialTerm",
    "ContractDate",
    "ContractObligation",
    "ContractAsset",
    "UniversalContractSchema",
    # Programs
    "UniversalContractParserProgram",
    "UniversalSolidityGeneratorProgram",
    "SecurityAuditorProgram",
    "ABIGeneratorProgram",
    "MCPServerGeneratorProgram",
    # Task Builders
    "create_parser_task_description",
    "create_solidity_generator_task_description",
    "create_audit_task_description",
    "create_abi_generator_task_description",
    "create_quality_evaluation_task_description",
    # Agent utilities
    "create_agents",
    "should_refine",
    "create_refinement_task_description",
    "_convert_to_crew_llm",
    "DEFAULT_MAX_REFINEMENT_ITERATIONS",
    # Main Translator
    "IBMAgenticContractTranslator",
]
