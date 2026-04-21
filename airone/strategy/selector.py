"""
AirOne Strategy Selector — Phase 2
Rule-based decision engine that selects and ranks compression strategies
based on the analysis report.

Phase 3 will replace the rule engine with a trained ML ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airone.analysis.engine import AnalysisReport


@dataclass
class StrategyCandidate:
    """A strategy with its estimated compression ratio and priority score."""
    strategy_name: str
    estimated_ratio: float
    priority: int          # lower = higher priority
    reason: str


class StrategySelector:
    """
    Selects and ranks compression strategies for a given :class:`AnalysisReport`.

    Usage::

        selector = StrategySelector(registry)
        candidates = selector.select(analysis_report)
        # candidates[0] is the most promising strategy
    """

    def __init__(self, registry) -> None:
        self._registry = registry   # StrategyRegistry instance

    def select(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        """
        Returns a ranked list of :class:`StrategyCandidate`.
        The orchestrator should try them in order and use the best outcome.
        """
        candidates: list[StrategyCandidate] = []

        # --- 1. Apply each rule ---
        candidates.extend(self._rule_procedural_gradient(report))
        candidates.extend(self._rule_procedural_cad(report))
        candidates.extend(self._rule_semantic_json(report))
        candidates.extend(self._rule_semantic_pdf(report))
        candidates.extend(self._rule_semantic_screenshot(report))
        candidates.extend(self._rule_neural_medical(report))
        candidates.extend(self._rule_traditional(report))

        # --- 2. Filter to only registered strategies ---
        registered = self._registry.list_names()
        candidates = [c for c in candidates if c.strategy_name in registered]

        # --- 3. Deduplicate (keep highest priority per strategy) ---
        seen: dict[str, StrategyCandidate] = {}
        for c in candidates:
            if c.strategy_name not in seen or c.priority < seen[c.strategy_name].priority:
                seen[c.strategy_name] = c

        # --- 4. Sort: priority ASC, then estimated_ratio DESC ---
        ranked = sorted(
            seen.values(),
            key=lambda c: (c.priority, -c.estimated_ratio),
        )

        # --- 5. Always append zstd as final fallback if not present ---
        if "traditional_zstd" not in {c.strategy_name for c in ranked}:
            ranked.append(StrategyCandidate(
                strategy_name="traditional_zstd",
                estimated_ratio=report.entropy.compressibility_estimate,
                priority=999,
                reason="Universal fallback",
            ))

        return ranked

    # ------------------------------------------------------------------
    # Rule methods
    # ------------------------------------------------------------------

    def _rule_procedural_gradient(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        if not report.is_image:
            return []
        ic = report.image_classification
        if ic is None:
            return []

        from airone.analysis.image_classifier import ContentType, GenerationMethod
        is_gradient = (
            ic.content_type == ContentType.GRADIENT
            or ic.generation_method == GenerationMethod.GRADIENT
        )
        if is_gradient and ic.content_confidence > 0.70:
            estimated = report.file_size / 200   # ~200 bytes for params
            return [StrategyCandidate(
                strategy_name="procedural_gradient",
                estimated_ratio=min(estimated, 50_000),
                priority=1,
                reason=f"Gradient detected ({ic.content_confidence:.0%} confidence).",
            )]
        return []

    def _rule_procedural_cad(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        if report.format.type in ("DWG", "DXF"):
            return [StrategyCandidate(
                strategy_name="procedural_cad",
                estimated_ratio=120.0,
                priority=2,
                reason="CAD file — parametric extraction applicable.",
            )]
        return []

    def _rule_semantic_json(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        fmt = report.format
        if fmt.type in ("JSON", "CSV") or fmt.mime_type in (
            "application/json", "text/csv"
        ):
            # Larger files benefit more from columnar encoding
            ratio = 80.0 if report.file_size > 50_000 else 30.0
            return [StrategyCandidate(
                strategy_name="semantic_json",
                estimated_ratio=ratio,
                priority=2,
                reason=f"Structured data ({fmt.type}) — columnar semantic encoding.",
            )]
        return []

    def _rule_semantic_pdf(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        if report.format.type == "PDF":
            return [StrategyCandidate(
                strategy_name="semantic_pdf",
                estimated_ratio=10.0,
                priority=3,
                reason="PDF — semantic decomposition strategy.",
            )]
        return []

    def _rule_semantic_screenshot(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        if not report.is_image:
            return []
        ic = report.image_classification
        if ic is None:
            return []

        from airone.analysis.image_classifier import ContentType
        if ic.content_type == ContentType.SCREENSHOT and ic.content_confidence > 0.60:
            return [StrategyCandidate(
                strategy_name="semantic_screenshot",
                estimated_ratio=28.0,
                priority=4,
                reason=f"Screenshot detected ({ic.content_confidence:.0%}).",
            )]
        return []

    def _rule_neural_medical(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        if report.format.is_medical:
            return [StrategyCandidate(
                strategy_name="neural_medical",
                estimated_ratio=35.0,
                priority=5,
                reason="Medical image — domain-specific neural codec.",
            )]
        return []

    def _rule_traditional(self, report: "AnalysisReport") -> list[StrategyCandidate]:
        ratio = report.entropy.compressibility_estimate
        return [StrategyCandidate(
            strategy_name="traditional_zstd",
            estimated_ratio=ratio,
            priority=50,
            reason=f"Entropy-based estimate: {ratio:.1f}x.",
        )]
