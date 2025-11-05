from .defaults import ApplyDefaults
from .train import (
    ApplyTrainingOptimization,
    ApplyDistributedTraining,
    ApplyFastKernelsOptimization,
    ApplyGradientCheckpointing,
    ApplyLoRAConfig,
    ApplyMoEOptimization,
    ApplyOptimalBatchSize,
)
from .data import ApplyChatFormat, ApplyQAFormat
from .compute import ApplyComputeConfig
from .actions import Action, IR

ACTIONS = [
    ApplyDefaults,
    ApplyComputeConfig,
    ApplyTrainingOptimization,
    ApplyDistributedTraining,
    ApplyFastKernelsOptimization,
    ApplyGradientCheckpointing,
    ApplyLoRAConfig,
    ApplyMoEOptimization,
    ApplyOptimalBatchSize,
    ApplyChatFormat,
    ApplyQAFormat,
]
