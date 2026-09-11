"""Research modules for hypothesis generation, validation, and lifecycle management."""

from research.overnight_edge_scanner import (
    OvernightEdgeConfig,
    OvernightEdgeScanner,
    OvernightEdgeSignal,
    classify_edge_type,
    compute_overnight_returns,
    compute_intraday_returns,
    validate_overnight_inputs,
)
from research.opportunity_framework import (
    HoldoutManager,
    MarketAnomaly,
    MeanReversionDetector,
    MomentumDetector,
    OpportunityDetector,
    OpportunityScanner,
    TemporalAnomalyDetector,
    VolumeGapDetector,
    benjamini_hochberg_qvalues,
    compute_opportunity_score,
)
from research.opportunity_lifecycle import (
    EdgeMonitor,
    OpportunityBoard,
    PaperTradingExperiment,
    StrategyGraveyard,
    StrategyLifecycleManager,
)

__all__ = [
    "OvernightEdgeConfig",
    "OvernightEdgeScanner",
    "OvernightEdgeSignal",
    "classify_edge_type",
    "compute_overnight_returns",
    "compute_intraday_returns",
    "validate_overnight_inputs",
    "MarketAnomaly",
    "OpportunityDetector",
    "OpportunityScanner",
    "TemporalAnomalyDetector",
    "MomentumDetector",
    "MeanReversionDetector",
    "VolumeGapDetector",
    "compute_opportunity_score",
    "benjamini_hochberg_qvalues",
    "HoldoutManager",
    "PaperTradingExperiment",
    "EdgeMonitor",
    "StrategyLifecycleManager",
    "StrategyGraveyard",
    "OpportunityBoard",
]
