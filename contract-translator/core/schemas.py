"""
Pydantic data models for contract representation

Defines all schema classes used for structured contract data:
- Party roles and contract types (enums)
- Contract components (parties, financial terms, dates, assets, obligations)
- Universal contract schema that works across all contract types
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PartyRole(str, Enum):
    """Common party roles across all contracts"""

    BUYER = "buyer"
    SELLER = "seller"
    LANDLORD = "landlord"
    TENANT = "tenant"
    EMPLOYER = "employer"
    EMPLOYEE = "employee"
    LENDER = "lender"
    BORROWER = "borrower"
    SERVICE_PROVIDER = "service_provider"
    CLIENT = "client"
    INVESTOR = "investor"
    COMPANY = "company"
    OTHER = "other"


class ContractType(str, Enum):
    """All supported contract types"""

    RENTAL = "rental_agreement"
    EMPLOYMENT = "employment_contract"
    SALES = "sales_agreement"
    SERVICE = "service_agreement"
    LOAN = "loan_agreement"
    NDA = "non_disclosure_agreement"
    PARTNERSHIP = "partnership_agreement"
    INVESTMENT = "investment_agreement"
    LEASE = "lease_agreement"
    PURCHASE = "purchase_agreement"
    OTHER = "other"


class ContractParty(BaseModel):
    """Represents a party involved in the contract"""

    name: str
    role: str
    address: Optional[str] = None
    email: Optional[str] = None
    entity_type: Optional[str] = None


class FinancialTerm(BaseModel):
    """Universal financial term"""

    amount: Optional[float] = None
    currency: Optional[str] = "ETH"
    purpose: str
    frequency: Optional[str] = None
    due_date: Optional[str] = None


class ContractDate(BaseModel):
    """Date-related information in contract"""

    date_type: str
    value: Optional[str] = None
    day_of_month: Optional[int] = None
    frequency: Optional[str] = None


class ContractObligation(BaseModel):
    """Obligation or requirement in the contract"""

    party: str  # Who has this obligation
    description: str
    deadline: Optional[str] = None
    penalty_for_breach: Optional[str] = None


class ContractAsset(BaseModel):
    """Asset mentioned in the contract"""

    type: str
    description: str
    location: Optional[str] = None
    quantity: Optional[int] = None
    value: Optional[float] = None


class UniversalContractSchema(BaseModel):
    """
    Universal schema that works for any contract type.
    Combines common fields with flexible conditions dictionary.
    """

    contract_type: str
    title: Optional[str] = None
    parties: List[ContractParty]
    financial_terms: List[FinancialTerm] = []
    dates: List[ContractDate] = []
    assets: List[ContractAsset] = []
    obligations: List[ContractObligation] = []
    special_terms: List[str] = []
    conditions: Dict[str, Any] = {}
    termination_conditions: List[str] = []
