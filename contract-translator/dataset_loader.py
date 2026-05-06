"""
Contract Dataset Loader
Loads and samples contracts from requirement_fsm_code.jsonl
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class ContractDatasetLoader:
    def __init__(self, dataset_path: str = None):
        """Initialize the dataset loader"""
        if dataset_path is None:
            # Default: requirement_fsm_code.jsonl lives one level above contract-translator/
            dataset_path = str(Path(__file__).parent.parent / "requirement_fsm_code.jsonl")
        self.dataset_path = Path(dataset_path)
        self.contracts = []
        self.loaded = False

    def load_contracts(self, max_contracts: Optional[int] = None) -> List[Dict]:
        """Load contracts from the dataset file

        Args:
            max_contracts: Maximum number of contracts to load (None = all)

        Returns:
            List of contract dictionaries
        """
        print(f"📂 Loading contracts from {self.dataset_path}")

        self.contracts = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_contracts and i >= max_contracts:
                    break

                try:
                    contract = json.loads(line.strip())
                    self.contracts.append(contract)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {i+1}: Invalid JSON")
                    continue

        self.loaded = True
        print(f"✓ Loaded {len(self.contracts)} contracts")
        return self.contracts

    def get_sample(self, n: int = 100, seed: Optional[int] = None) -> List[Dict]:
        """Get a random sample of contracts

        Args:
            n: Number of contracts to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled contracts
        """
        if not self.loaded:
            self.load_contracts()

        if n >= len(self.contracts):
            print(
                f"⚠️  Requested {n} contracts but only {len(self.contracts)} available"
            )
            return self.contracts

        if seed is not None:
            random.seed(seed)

        sample = random.sample(self.contracts, n)
        print(f"📊 Sampled {len(sample)} contracts")
        return sample

    def get_contract_by_index(self, index: int) -> Dict:
        """Get a specific contract by index

        Args:
            index: Index of the contract (0-based)

        Returns:
            Contract dictionary
        """
        if not self.loaded:
            self.load_contracts()

        if index < 0 or index >= len(self.contracts):
            raise IndexError(
                f"Contract index {index} out of range (0-{len(self.contracts)-1})"
            )

        return self.contracts[index]

    def extract_contract_text(self, contract: Dict) -> str:
        """Extract the natural language requirement from a contract

        Args:
            contract: Contract dictionary

        Returns:
            Natural language contract description
        """
        return contract.get("user_requirement", "")

    def extract_ground_truth_code(self, contract: Dict) -> str:
        """Extract the ground truth Solidity code from a contract

        Args:
            contract: Contract dictionary

        Returns:
            Ground truth Solidity code
        """
        return contract.get("code", "")

    def get_contract_metadata(self, contract: Dict) -> Dict:
        """Extract metadata from a contract

        Args:
            contract: Contract dictionary

        Returns:
            Dictionary with metadata (version, has FSM, has code, etc.)
        """
        return {
            "version": contract.get("version", "unknown"),
            "has_fsm": "FSM" in contract and contract["FSM"],
            "has_code": "code" in contract and contract["code"],
            "requirement_length": len(contract.get("user_requirement", "")),
        }

    def save_sample_to_file(self, sample: List[Dict], output_path: str):
        """Save a sample of contracts to a new file

        Args:
            sample: List of contracts to save
            output_path: Path to save the sample
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for contract in sample:
                f.write(json.dumps(contract) + "\n")

        print(f"✓ Saved {len(sample)} contracts to {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = ContractDatasetLoader()

    # Load first 1500 contracts
    contracts = loader.load_contracts(max_contracts=1500)
    print(f"\nTotal contracts loaded: {len(contracts)}")

    # Get a sample of 100
    sample = loader.get_sample(n=100, seed=42)

    # Display first contract
    if sample:
        first_contract = sample[0]
        print("\n" + "=" * 70)
        print("SAMPLE CONTRACT:")
        print("=" * 70)
        print(f"\nRequirement: {loader.extract_contract_text(first_contract)[:200]}...")
        print(f"\nMetadata: {loader.get_contract_metadata(first_contract)}")
