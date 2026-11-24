"""
NonNULL Framework v1.0.1 - Unified Alignment & Meta-Control Package
File: doctrine/__init__.py
"""

__version__ = "1.0.1"
__author__ = "Giang Nhan Nghia"

from .core_models import CoreLearner, resolve_compute_dtype
from .meta_layers import MetaController, ValueCritic, ContextPairedValueCritic, EulerConsensusGate
from .hidden_gate import HiddenStateGate, SplitLayerGateController
from .doctrine_agent import DoctrineAgent

__all__ = [
    "__version__",
    "__author__",
    "CoreLearner",
    "resolve_compute_dtype",
    "MetaController",
    "ValueCritic",
    "ContextPairedValueCritic",
    "EulerConsensusGate",
    "HiddenStateGate",
    "SplitLayerGateController",
    "DoctrineAgent",
]