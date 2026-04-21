"""
AirOne Analysis Engine — Phase 2
Coordinates all analysers into a single unified report.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from airone.analysis.entropy import EntropyAnalyser, EntropyReport
from airone.analysis.format_detector import FileFormat, FormatDetector
from airone.analysis.image_classifier import (
    ImageClassification,
    ImageClassifier,
)
from airone.exceptions import FormatError


@dataclass
class AnalysisReport:
    """
    Unified analysis report produced by :class:`AnalysisEngine`.
    This is the primary input to the StrategySelector.
    """
    # Core identity
    file_path: str
    file_name: str
    file_size: int               # bytes
    analysis_time: float         # seconds

    # Format
    format: FileFormat

    # Entropy
    entropy: EntropyReport

    # Image-specific (None for non-images)
    image_classification: Optional[ImageClassification] = None

    # Strategy hints accumulated during analysis
    strategy_hints: list[str] = field(default_factory=list)

    # Human-readable notes
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers

    @property
    def is_image(self) -> bool:
        return self.format.is_image

    @property
    def is_document(self) -> bool:
        return self.format.is_document

    @property
    def recommended_strategy(self) -> str:
        """
        Quick strategy hint derived from hints list.
        StrategySelector will do the proper ranking.
        """
        return self.strategy_hints[0] if self.strategy_hints else "traditional_zstd"

    def summary(self) -> str:
        lines = [
            f"File       : {self.file_name}",
            f"Size       : {self.file_size / 1024:.1f} KB",
            f"Format     : {self.format.type} ({self.format.category})",
            f"Entropy    : {self.entropy.global_entropy:.2f} bits/byte",
            f"Est. ratio : {self.entropy.compressibility_estimate:.1f}x",
        ]
        if self.image_classification:
            ic = self.image_classification
            lines += [
                f"Content    : {ic.content_type:.0} ({ic.content_confidence:.0%})",
                f"Domain     : {ic.domain}",
                f"Generation : {ic.generation_method}",
            ]
        lines.append(f"Strategy   : {self.recommended_strategy}")
        return "\n".join(lines)


class AnalysisEngine:
    """
    Facade that runs all analysis sub-systems and returns a unified report.

    Usage::

        engine = AnalysisEngine()
        report = engine.analyse("invoice.pdf")
        print(report.summary())
    """

    def __init__(self) -> None:
        self._format_detector    = FormatDetector()
        self._entropy_analyser   = EntropyAnalyser()
        self._image_classifier   = ImageClassifier()

    def analyse(self, file_path: str) -> AnalysisReport:
        """
        Run the full analysis pipeline on *file_path*.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        start = time.perf_counter()

        # 1. Format detection
        fmt = self._format_detector.detect(file_path)

        # 2. Entropy analysis
        entropy_report = self._entropy_analyser.analyse(file_path)

        # 3. Image classification (images only)
        image_classification = None
        if fmt.is_image:
            try:
                image_classification = self._image_classifier.classify(file_path)
            except Exception as exc:
                # Non-fatal: log and continue without image classification
                pass

        elapsed = time.perf_counter() - start

        # 4. Build report
        report = AnalysisReport(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            file_size=os.path.getsize(file_path),
            analysis_time=elapsed,
            format=fmt,
            entropy=entropy_report,
            image_classification=image_classification,
        )

        # 5. Derive strategy hints
        self._derive_hints(report)

        return report

    # ------------------------------------------------------------------

    def _derive_hints(self, report: AnalysisReport) -> None:
        """
        Populate report.strategy_hints with ordered strategy preferences.
        This is a fast, rule-based pre-filter.
        The StrategySelector will do the final ranking.
        """
        hints: list[str] = []
        notes: list[str] = []

        ic = report.image_classification

        # --- Procedural content ---
        if ic:
            from airone.analysis.image_classifier import GenerationMethod, ContentType
            if ic.generation_method == GenerationMethod.GRADIENT and ic.generation_confidence > 0.80:
                hints.append("procedural_gradient")
                notes.append("Image appears to be a gradient — parametric compression possible.")

            if ic.content_type == ContentType.SCREENSHOT:
                hints.append("semantic_screenshot")
                notes.append("Screenshot detected — UI-aware compression applicable.")

            if ic.content_type == ContentType.LOGO:
                hints.append("procedural_vector")
                notes.append("Logo detected — vector optimization possible.")

        # --- Structured data (JSON / CSV) ---
        if report.format.type in ("JSON", "CSV") or report.format.mime_type in (
            "application/json", "text/csv"
        ):
            hints.append("semantic_json")
            notes.append("Structured data detected — columnar semantic compression applicable.")

        # --- Document-specific ---
        if report.format.type == "PDF":
            hints.append("semantic_pdf")
            notes.append("PDF: semantic decomposition will be attempted.")

        if report.format.type in ("DOCX", "XLSX", "PPTX"):
            hints.append("semantic_office")
            notes.append("Office document: structural optimisation possible.")

        if report.format.type in ("DWG", "DXF"):
            hints.append("procedural_cad")
            notes.append("CAD file: parametric extraction possible.")

        # --- Medical ---
        if report.format.is_medical:
            hints.append("neural_medical")
            notes.append("Medical image: domain-specific codec available.")

        # --- High-entropy fallback ---
        if report.entropy.global_entropy > 7.5:
            notes.append("Very high entropy — limited compression expected.")
            if "traditional_zstd" not in hints:
                hints.append("traditional_zstd")

        # --- Default fallback always present ---
        if "traditional_zstd" not in hints:
            hints.append("traditional_zstd")

        report.strategy_hints = hints
        report.notes = notes
