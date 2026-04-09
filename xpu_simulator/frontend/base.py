"""Base class for all graph extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.graph import ComputeGraph
from ..core.operator import Dtype
from .op_registry import OpRegistry


class GraphExtractor(ABC):
    """Abstract base for all compute graph extractors.

    Subclasses implement extract() to produce a ComputeGraph from
    their respective source format (PyTorch model, ONNX file, profiler trace, etc.).
    """

    def __init__(self, dtype: Dtype = Dtype.FP16):
        self.dtype = dtype
        self.registry = OpRegistry()

    @abstractmethod
    def extract(self, *args, **kwargs) -> ComputeGraph:
        """Extract a ComputeGraph from a model source."""
        ...
