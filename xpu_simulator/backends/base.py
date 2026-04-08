"""Backend abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.hardware import HardwareSpec
from ..core.cost_model import CostModel, OpCost
from ..core.operator import OpSpec, TensorSpec


class Backend(ABC):
    """Abstract base for device backends."""

    def __init__(self, hw: HardwareSpec, cost_model: CostModel):
        self.hw = hw
        self.cost_model = cost_model

    @abstractmethod
    def get_memory_layout(self, op: OpSpec, tensor: TensorSpec) -> str:
        """Return the preferred memory layout for a tensor in a given op context."""
        ...

    def estimate(self, op: OpSpec) -> OpCost:
        """Estimate cost for an operation on this backend."""
        return self.cost_model.estimate(op)
