"""
AirOne PDF Semantic Compressor - Phase 3

Compresses PDFs component-by-component:
    - Text    → ZSTD (high redundancy in prose text)
    - Images  → Best of: GradientCompressor / ZSTD (future: NeuralCodec)
    - Fonts   → ZSTD (binary, but often repetitive)
    - Layout  → ZSTD (tiny JSON)
    - Metadata→ ZSTD (tiny JSON)

Deduplication:
    Repeated images (same content_hash) are stored once.
    Subsequent references store only the hash + page metadata.

The final payload is a msgpack-encoded bundle containing
all compressed components plus a reconstruction manifest.

This compressor is intentionally extensible:
    - swap in NeuralCodec for images in Phase 4
    - add vector re-encoding in Phase 4
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict
from typing import Optional

import msgpack

from airone.analysis.document_decomposer import (
    ComponentType,
    DecomposedDocument,
    DocumentComponent,
    DocumentDecomposer,
)
from airone.compressors.base import BaseCompressor, CompressionResult
from airone.compressors.procedural.gradient import GradientCompressor
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.exceptions import CompressionError, DecompressionError


class PDFSemanticCompressor(BaseCompressor):
    """
    Semantic compression for PDF documents.

    Compression pipeline
    --------------------
    1. Decompose PDF into semantic components
    2. Deduplicate repeated images
    3. Compress each unique component with optimal strategy
    4. Pack into a single msgpack bundle
    5. Apply a final ZSTD pass over the bundle

    Decompression pipeline
    ----------------------
    1. Unpack msgpack bundle
    2. Reconstruct components from their compressed forms
    3. Re-assemble PDF (Phase 4; currently returns component bundle)
    """

    name = "semantic_pdf"

    def __init__(self) -> None:
        self._decomposer  = DocumentDecomposer()
        self._zstd        = ZstdCompressor()
        self._gradient    = GradientCompressor()

    # ------------------------------------------------------------------
    # BaseCompressor interface
    # ------------------------------------------------------------------

    def can_handle(self, analysis) -> bool:
        return (
            analysis is not None
            and hasattr(analysis, "format")
            and analysis.format.type == "PDF"
        )

    def estimate_ratio(self, analysis) -> float:
        # Conservative estimate based on typical PDF composition
        if analysis and hasattr(analysis, "entropy"):
            if analysis.entropy.global_entropy < 5.0:
                return 8.0
        return 5.0

    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        start = time.perf_counter()

        # We need the file path to use pypdf
        source_path = (
            analysis.file_path
            if analysis and hasattr(analysis, "file_path")
            else self._write_temp(data)
        )

        # 1. Decompose
        try:
            doc = self._decomposer.decompose(source_path)
        except Exception as exc:
            raise CompressionError(
                f"PDFSemanticCompressor: decomposition failed: {exc}"
            ) from exc

        # 2. Build compressed bundle
        bundle = self._build_bundle(doc)

        # 3. Serialise with msgpack
        raw_bundle = msgpack.packb(bundle, use_bin_type=True)

        # 4. Final ZSTD pass over the whole bundle
        final_result = self._zstd.compress(raw_bundle)
        compressed_bytes = final_result.compressed_data

        return CompressionResult(
            compressed_data=compressed_bytes,
            original_size=len(data),
            compressed_size=len(compressed_bytes),
            strategy_name=self.name,
            execution_time=time.perf_counter() - start,
            metadata={
                "format": "PDF",
                "page_count": doc.page_count,
                "unique_images": doc.unique_image_count,
                "duplicate_images": doc.duplicate_image_count,
                "duplicate_map": doc.duplicate_map,
            },
        )

    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        """
        Decompress a semantically-compressed PDF bundle.

        Phase 3 returns the reconstructed component bundle as msgpack.
        Full PDF reconstruction (re-building the PDF byte stream) is
        a Phase 4 deliverable that requires a PDF writer integration.
        """
        try:
            # Undo final ZSTD pass
            raw_bundle = self._zstd.decompress(compressed_data, {})

            # Deserialise
            bundle = msgpack.unpackb(raw_bundle, raw=False)

            # Reconstruct individual components
            components = self._reconstruct_components(bundle)

            # Return re-serialised bundle (Phase 3 intermediate format)
            return msgpack.packb(
                {"components": components, "meta": metadata},
                use_bin_type=True,
            )
        except Exception as exc:
            raise DecompressionError(
                f"PDFSemanticCompressor: decompression failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Bundle construction
    # ------------------------------------------------------------------

    def _build_bundle(self, doc: DecomposedDocument) -> dict:
        """
        Compress each component group and gather into a serialisable dict.
        """
        bundle: dict = {
            "format":   "PDF",
            "pages":    doc.page_count,
            "dup_map":  doc.duplicate_map,
            "text":     [],
            "images":   [],
            "fonts":    [],
            "metadata": None,
            "layout":   None,
        }

        # Text blocks
        for component in doc.text_blocks:
            bundle["text"].append(
                self._compress_component(component, preferred="zstd")
            )

        # Images (with deduplication)
        seen_hashes: dict[str, bytes] = {}  # hash → compressed bytes

        for component in doc.images:
            h = component.content_hash
            if h in seen_hashes:
                # Store a reference instead of the full image
                bundle["images"].append({
                    "type":   "reference",
                    "hash":   h,
                    "pages":  component.pages,
                    "meta":   component.meta,
                })
            else:
                compressed = self._compress_image_component(component)
                seen_hashes[h] = compressed["data"]
                bundle["images"].append(compressed)

        # Fonts
        for component in doc.fonts:
            bundle["fonts"].append(
                self._compress_component(component, preferred="zstd")
            )

        # Metadata & Layout (both tiny — just ZSTD them)
        if doc.metadata_block:
            bundle["metadata"] = self._compress_component(
                doc.metadata_block, preferred="zstd"
            )
        if doc.layout_block:
            bundle["layout"] = self._compress_component(
                doc.layout_block, preferred="zstd"
            )

        return bundle

    def _compress_component(
        self,
        component: DocumentComponent,
        preferred: str = "zstd",
    ) -> dict:
        """
        Compress a single component using the preferred codec.
        Returns a serialisable dict.
        """
        result = self._zstd.compress(component.data)
        return {
            "type":   component.component_type.value,
            "codec":  "zstd",
            "pages":  component.pages,
            "meta":   component.meta,
            "hash":   component.content_hash,
            "data":   result.compressed_data,
        }

    def _compress_image_component(self, component: DocumentComponent) -> dict:
        """
        Compress an image component.
        Tries gradient compressor first; falls back to ZSTD.
        Future phases will add NeuralCodec here.
        """
        codec_used = "zstd"
        compressed_data: Optional[bytes] = None

        # Attempt gradient compression
        try:
            result = self._gradient.compress(component.data)
            compressed_data = result.compressed_data
            codec_used = "procedural_gradient"
        except Exception:
            pass

        # Fallback to ZSTD
        if compressed_data is None:
            result = self._zstd.compress(component.data)
            compressed_data = result.compressed_data

        return {
            "type":  ComponentType.IMAGE.value,
            "codec": codec_used,
            "pages": component.pages,
            "meta":  component.meta,
            "hash":  component.content_hash,
            "data":  compressed_data,
        }

    # ------------------------------------------------------------------
    # Bundle reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_components(self, bundle: dict) -> list[dict]:
        """
        Decompress each component in the bundle.
        Returns a list of {type, pages, meta, data} dicts.
        """
        reconstructed = []

        # Text
        for item in bundle.get("text", []):
            reconstructed.append(self._decompress_item(item))

        # Images
        img_store: dict[str, bytes] = {}
        for item in bundle.get("images", []):
            if item.get("type") == "reference":
                h = item["hash"]
                # Data was stored in the first occurrence
                data = img_store.get(h, b"")
                reconstructed.append({
                    "type":  "image",
                    "pages": item["pages"],
                    "meta":  item["meta"],
                    "data":  data,
                })
            else:
                dec = self._decompress_item(item)
                img_store[item["hash"]] = dec["data"]
                reconstructed.append(dec)

        # Fonts
        for item in bundle.get("fonts", []):
            reconstructed.append(self._decompress_item(item))

        # Metadata & Layout
        for key in ("metadata", "layout"):
            item = bundle.get(key)
            if item:
                reconstructed.append(self._decompress_item(item))

        return reconstructed

    def _decompress_item(self, item: dict) -> dict:
        codec = item.get("codec", "zstd")

        if codec == "zstd":
            data = self._zstd.decompress(item["data"], {})
        elif codec == "procedural_gradient":
            data = self._gradient.decompress(item["data"], {})
        else:
            data = item["data"]   # Unknown codec — return as-is

        return {
            "type":  item.get("type"),
            "pages": item.get("pages", []),
            "meta":  item.get("meta", {}),
            "data":  data,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _write_temp(data: bytes) -> str:
        """
        Write raw bytes to a temporary file so pypdf can read it.
        Used when only bytes (not a path) are provided.
        """
        import tempfile

        # Windows-safe temp file creation
        fd, path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(data)
        return path
