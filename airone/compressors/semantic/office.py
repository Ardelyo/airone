"""
AirOne Office Document Compressor - Phase 5
Semantic compression for DOCX, XLSX, and PPTX.

Office Open XML documents are ZIP archives containing XML.
This compressor understands their internal structure:

DOCX:
    word/document.xml        ← Main text content
    word/media/image*.png    ← Embedded images
    word/styles.xml          ← Style definitions
    word/fonts/              ← Embedded fonts

XLSX:
    xl/worksheets/sheet*.xml ← Cell data (often repetitive)
    xl/sharedStrings.xml     ← Deduplicated string table
    xl/styles.xml            ← Cell formatting

PPTX:
    ppt/slides/slide*.xml    ← Per-slide content
    ppt/media/               ← Embedded media

Compression strategy:
    1. Unzip the archive
    2. Identify and deduplicate media (images, fonts)
    3. Compress XML content (text) with Brotli (excellent for XML)
    4. Compress binary content (images) with domain-appropriate codec
    5. Repack as optimised ZIP with ZSTD store
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import msgpack

from airone.compressors.base import BaseCompressor, CompressionResult
from airone.compressors.procedural.gradient import GradientCompressor
from airone.compressors.traditional.brotli import BrotliCompressor
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.exceptions import CompressionError, DecompressionError


# ---------------------------------------------------------------------------
# Office format classifier
# ---------------------------------------------------------------------------

class OfficeFormat:
    DOCX = "DOCX"
    XLSX = "XLSX"
    PPTX = "PPTX"

    # Marker files that identify each format
    _MARKERS = {
        "word/document.xml":       DOCX,
        "xl/workbook.xml":         XLSX,
        "ppt/presentation.xml":    PPTX,
    }

    @classmethod
    def detect(cls, zip_file: zipfile.ZipFile) -> Optional[str]:
        names = set(zip_file.namelist())
        for marker, fmt in cls._MARKERS.items():
            if marker in names:
                return fmt
        return None

    @classmethod
    def is_xml(cls, path: str) -> bool:
        return path.endswith(".xml") or path.endswith(".rels")

    @classmethod
    def is_media(cls, path: str) -> bool:
        media_exts = (
            ".png", ".jpg", ".jpeg", ".gif", ".bmp",
            ".tiff", ".emf", ".wmf", ".svg",
        )
        return any(path.lower().endswith(ext) for ext in media_exts)

    @classmethod
    def is_font(cls, path: str) -> bool:
        font_exts = (".ttf", ".otf", ".fntdata", ".odttf")
        return any(path.lower().endswith(ext) for ext in font_exts)


# ---------------------------------------------------------------------------
# Component record
# ---------------------------------------------------------------------------

@dataclass
class OfficeComponent:
    """One file extracted from the Office ZIP archive."""
    path:         str      # path within the ZIP
    raw_data:     bytes
    component_type: str   # "xml" | "media" | "font" | "other"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.raw_data and not self.content_hash:
            self.content_hash = hashlib.sha256(self.raw_data).hexdigest()


# ---------------------------------------------------------------------------
# Office Semantic Compressor
# ---------------------------------------------------------------------------

class OfficeSemanticCompressor(BaseCompressor):
    """
    Semantic compression for DOCX, XLSX, PPTX files.

    These formats are ZIP archives — we compress each internal
    component with its optimal strategy rather than treating
    the whole file as an opaque blob.
    """

    name = "semantic_office"

    def __init__(self) -> None:
        self._zstd     = ZstdCompressor()
        self._brotli   = BrotliCompressor(quality=9)
        self._gradient = GradientCompressor()

    # ------------------------------------------------------------------
    # BaseCompressor interface
    # ------------------------------------------------------------------

    def can_handle(self, analysis) -> bool:
        if analysis is None:
            return False
        fmt = getattr(analysis, "format", None)
        if fmt is None:
            return False
        return fmt.type in ("DOCX", "XLSX", "PPTX")

    def estimate_ratio(self, analysis) -> float:
        fmt_type = getattr(
            getattr(analysis, "format", None), "type", ""
        )
        estimates = {
            "DOCX": 6.0,
            "XLSX": 8.0,   # Spreadsheets often have highly repetitive XML
            "PPTX": 4.0,
        }
        return estimates.get(fmt_type, 4.0)

    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        start = time.perf_counter()

        # 1. Open ZIP
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise CompressionError(
                f"OfficeSemanticCompressor: not a valid ZIP: {exc}"
            ) from exc

        # 2. Detect format
        fmt = OfficeFormat.detect(archive)
        if fmt is None:
            raise CompressionError(
                "OfficeSemanticCompressor: unrecognised Office format."
            )

        # 3. Extract components
        components = self._extract_components(archive)

        # 4. Compress component bundle
        bundle = self._compress_bundle(components, fmt)

        # 5. Serialise
        raw_bundle = msgpack.packb(bundle, use_bin_type=True)

        # 6. Final ZSTD pass
        final = self._zstd.compress(raw_bundle)

        return CompressionResult(
            compressed_data=final.compressed_data,
            original_size=len(data),
            compressed_size=final.compressed_size,
            strategy_name=self.name,
            execution_time=time.perf_counter() - start,
            metadata={
                "office_format":   fmt,
                "component_count": len(components),
            },
        )

    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        # Undo ZSTD pass
        raw_bundle = self._zstd.decompress(compressed_data, {})
        bundle     = msgpack.unpackb(raw_bundle, raw=False)

        # Reconstruct ZIP archive in memory
        out_buf = io.BytesIO()
        with zipfile.ZipFile(
            out_buf, "w", compression=zipfile.ZIP_DEFLATED
        ) as out_zip:
            self._reconstruct_archive(bundle, out_zip)

        return out_buf.getvalue()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_components(
        self, archive: zipfile.ZipFile
    ) -> list[OfficeComponent]:
        components = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            try:
                raw = archive.read(info.filename)
                c_type = self._classify_component(info.filename)
                components.append(
                    OfficeComponent(
                        path=info.filename,
                        raw_data=raw,
                        component_type=c_type,
                    )
                )
            except Exception:
                continue
        return components

    @staticmethod
    def _classify_component(path: str) -> str:
        if OfficeFormat.is_xml(path):
            return "xml"
        if OfficeFormat.is_media(path):
            return "media"
        if OfficeFormat.is_font(path):
            return "font"
        return "other"

    # ------------------------------------------------------------------
    # Bundle construction
    # ------------------------------------------------------------------

    def _compress_bundle(
        self,
        components: list[OfficeComponent],
        fmt: str,
    ) -> dict:
        bundle: dict = {
            "format":     fmt,
            "components": [],
            "dedup_map":  {},
        }

        seen_hashes: dict[str, bytes] = {}

        for comp in components:
            h = comp.content_hash

            if h in seen_hashes:
                # Store reference only
                bundle["components"].append({
                    "path":   comp.path,
                    "type":   "reference",
                    "hash":   h,
                })
                bundle["dedup_map"].setdefault(h, []).append(comp.path)
                continue

            # Compress by component type
            compressed, codec = self._compress_component(comp)
            seen_hashes[h] = compressed

            bundle["components"].append({
                "path":  comp.path,
                "type":  comp.component_type,
                "codec": codec,
                "hash":  h,
                "data":  compressed,
            })

        return bundle

    def _compress_component(
        self, comp: OfficeComponent
    ) -> tuple[bytes, str]:
        """
        Choose optimal compressor for each component type.
        """
        if comp.component_type == "xml":
            # Brotli is excellent for XML (structured, repetitive)
            try:
                result = self._brotli.compress(comp.raw_data)
                return result.compressed_data, "brotli"
            except Exception:
                pass

        elif comp.component_type == "media":
            # Try gradient detection for image components
            try:
                result = self._gradient.compress(comp.raw_data)
                return result.compressed_data, "procedural_gradient"
            except Exception:
                pass

        # Universal fallback
        result = self._zstd.compress(comp.raw_data)
        return result.compressed_data, "zstd"

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_archive(
        self, bundle: dict, out_zip: zipfile.ZipFile
    ) -> None:
        # Build content store from non-reference components
        content_store: dict[str, bytes] = {}
        for item in bundle["components"]:
            if item["type"] != "reference":
                decompressed = self._decompress_item(item)
                content_store[item["hash"]] = decompressed

        # Write all files to output ZIP
        for item in bundle["components"]:
            path = item["path"]
            if item["type"] == "reference":
                data = content_store.get(item["hash"], b"")
            else:
                data = content_store.get(item["hash"], b"")
            out_zip.writestr(path, data)

    def _decompress_item(self, item: dict) -> bytes:
        codec = item.get("codec", "zstd")
        data  = item.get("data", b"")
        if not data:
            return b""
        if codec == "brotli":
            return self._brotli.decompress(data, {})
        if codec == "procedural_gradient":
            return self._gradient.decompress(data, {})
        return self._zstd.decompress(data, {})
