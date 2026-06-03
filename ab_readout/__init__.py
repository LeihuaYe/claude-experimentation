"""ab_readout — a trustworthy A/B test readout pipeline."""
from .readout import (
    srm_check, cuped, mean_effect, proportion_effect,
    benjamini_hochberg, run_readout, format_report,
)

__all__ = ["srm_check", "cuped", "mean_effect", "proportion_effect",
           "benjamini_hochberg", "run_readout", "format_report"]
__version__ = "0.1.0"
