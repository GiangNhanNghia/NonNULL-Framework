"""
NonNULL Framework V1.0 - Core Package Initialization
File: doctrine/__init__.py
"""

__version__ = "1.0.0"
__author__ = "Giang Nhan Nghia"

from .core_models import CoreLearner
from .meta_layers import MetaController, HarmonyLayer, HarmonyController
from .doctrine_trainer import HierarchicalTrainer, EulerTrainer
from .doctrine_agent import DoctrineAgent
from .ltm_services import LTM_Services, LTMService, GenesisEngine, GenesisEmbedder, VectorDB
from .doctrine_logger import HarmonyLogger, setup_logger, log_genesis, logger
from .utils import seed_everything, count_parameters, load_config

__all__ = [
    "CoreLearner",
    "HarmonyController",
    "HarmonyLayer",
    "MetaController",
    "EulerTrainer",
    "HierarchicalTrainer",
    "DoctrineAgent",
    "LTMService",
    "LTM_Services",
    "GenesisEngine",
    "GenesisEmbedder",
    "VectorDB",
    "HarmonyLogger",
    "setup_logger",
    "log_genesis",
    "logger",
    "seed_everything",
    "count_parameters",
    "load_config",
]