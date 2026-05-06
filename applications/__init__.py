"""
Namespace shim for the applications package.

Because `contract-translator` has a hyphen it can't be imported directly.
This file adds contract-translator/ to sys.path and aliases core/* as
applications.* so experiment files can do:

    from applications.translator import IBMAgenticContractTranslator
    from applications.solidity_compiler import SolidityCompilationChecker
"""

import sys
from pathlib import Path

# Add contract-translator/ to sys.path so `import core` works
_ct_path = str(Path(__file__).parent.parent / "contract-translator")
if _ct_path not in sys.path:
    sys.path.insert(0, _ct_path)

# Import in dependency order: schemas must come before anything that uses them
import core.schemas  # noqa: F401

try:
    import core.programs  # noqa: F401
    _programs_ok = True
except ImportError:
    _programs_ok = False

import core.agents  # noqa: F401
import core.solidity_compiler  # noqa: F401
import core.task_builders  # noqa: F401
import core.translator  # noqa: F401

# Register module aliases so `from applications.X import Y` resolves
sys.modules.setdefault("applications.schemas", core.schemas)
if _programs_ok:
    sys.modules.setdefault("applications.programs", core.programs)
sys.modules.setdefault("applications.task_builders", core.task_builders)
sys.modules.setdefault("applications.agents", core.agents)
sys.modules.setdefault("applications.solidity_compiler", core.solidity_compiler)
sys.modules.setdefault("applications.translator", core.translator)
