"""
Token Sort Methods for Prefix-Optimal Ordering

This package contains implementations of different token sorting methods:
- A: Differentiable Sorting (SoftSort/NeuralSort) - stable, prefix-optimal
- B: Gating-based Selection - simple, efficient
- C: Multi-Distillation - complex, high performance

Each method implements the BaseTokenSorter interface and focuses on
learning optimal token ordering for prefix selection.
"""

# Import base classes first
from .base import BaseTokenSorter, register_token_sort, get_token_sort_class

# Import all token sort methods to trigger registration
from . import softsort
from . import gatingsort

# Import multi-distillation with correct file name
try:
    # Import the module first to avoid circular import issues
    from . import multi_distillation as multi_dist_module
except ImportError:
    # Fallback if file doesn't exist or has issues
    multi_dist_module = None

# Import the specific classes to make them available
from .softsort import DifferentiableSortingTokenSorter
try:
    from .gatingsort import RandomGatingTokenSorter
except ImportError:
    RandomGatingTokenSorter = None

if multi_dist_module is not None:
    try:
        from .multi_distillation import MultiBudgetDistillationWrapper as MultiDistillationTokenSorter
    except (ImportError, AttributeError):
        MultiDistillationTokenSorter = None
else:
    MultiDistillationTokenSorter = None

__all__ = [
    "BaseTokenSorter",
    "register_token_sort",
    "get_token_sort_class",
    "softsort",
    "gatingsort", 
    "multi_distillation",
    "DifferentiableSortingTokenSorter",
]

if RandomGatingTokenSorter is not None:
    __all__.append("RandomGatingTokenSorter")
if MultiDistillationTokenSorter is not None:
    __all__.append("MultiDistillationTokenSorter")