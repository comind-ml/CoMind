# https://github.com/WecoAI/aideml/blob/main/aide/utils/metric.py
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Dict

import numpy as np
from dataclasses_json import DataClassJsonMixin

@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    Represents the value of a metric to be optimized, which can be compared to other metric values.
    Comparisons (and max, min) are based on which value is better, not which is larger.
    """

    value: float | int | np.number | np.floating | np.ndarray | None
    maximize: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:
            # The isinstance check for np.floating is causing issues with certain numpy versions
            # where np.floating is not a class. Using a broader check.
            if not isinstance(self.value, (float, int, np.number)):
                 raise TypeError(f"Metric value must be a number, but got {type(self.value)}")
            self.value = float(self.value)
        
    def __gt__(self, other) -> bool:
        """True if self is a _better_ (not necessarily larger) metric value than other"""
        if self.value is None:
            return False
        if other.value is None:
            return True

        if not isinstance(other, MetricValue):
            return NotImplemented

        assert self.maximize == other.maximize, "Cannot compare metrics with different optimization directions"

        if self.value == other.value:
            return False

        comp = self.value > other.value
        return comp if self.maximize else not comp

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return self.value == other.value and self.maximize == other.maximize

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        
        value_str = f"{self.value:.4f}" if self.value is not None else "N/A"
        return f"Metric{opt_dir}({value_str})"

    @property
    def is_worst(self):
        """True if the metric value is the worst possible value."""
        return self.value is None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for JSON serialization."""
        return {"value": self.value, "maximize": self.maximize}


@dataclass
class WorstMetricValue(MetricValue):
    """
    Represents an invalid metric value, e.g. when the agent creates a buggy solution.
    Always compares worse than any valid metric value.
    """

    value: None = None
    maximize: bool | None = field(default=None, kw_only=True)

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for JSON serialization."""
        return {"value": self.value, "maximize": self.maximize}