"""ab_design — experiment design: power / MDE / sample-size / duration."""
from .design import (
    sample_size, mde, power, power_curve, duration,
    cluster_design_effect, allocate_alpha, ratio_variance, switchback_design,
    format_design_report,
)

__all__ = ["sample_size", "mde", "power", "power_curve", "duration",
           "cluster_design_effect", "allocate_alpha", "ratio_variance",
           "switchback_design", "format_design_report"]
__version__ = "0.1.0"
