"""
AirOne Document Decomposer - Phase 3
Breaks structured documents into semantic components.

PDFs are treated not as a stream of bytes but as a composition of:
    - Text blocks  (font, layout, content)
    - Images       (classified individually)
    - Vector paths (lines, shapes)
    - Fonts        (embedded vs. referenceable)
    - Metadata     (title, author, creation date)
    - Repeated elements (headers, footers, logos)

Each component is compressed by the most appropriate strategy,
then the results are reassembled into a single .air container.

Supported formats this phase:
    PDF   — full decomposition
    DOCX  — stub (Phase 4)
    XLSX  — stub (Phase 4)
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from airone.exceptions import FormatError


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ComponentType(str, Enum):
    TEXT        = "text"
    IMAGE       = "image"
    VECTOR      = "vector"
    FONT        = "font"
    METADATA    = "metadata"
    LAYOUT      = "layout"


@dataclass
class DocumentComponent:
    """
    One semantic unit extracted from a document.
    Carries both its raw data and enough context
    for the compressor to make good decisions.
    """
    component_type: ComponentType

    # Raw bytes of this component (PNG for images, UTF-8 for text, etc.)
    data: bytes

    # SHA-256 of data — used for cross-page deduplication
    content_hash: str = ""

    # Page number(s) where this component appears
    pages: list[int] = field(default_factory=list)

    # Bounding box on its page [x0, y0, x1, y1] in PDF units
    bbox: Optional[list[float]] = None

    # Type-specific metadata
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data and not self.content_hash:
            self.content_hash = hashlib.sha256(self.data).hexdigest()

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def size_kb(self) -> float:
        return self.size / 1024


@dataclass
class DecomposedDocument:
    """
    Full decomposition result for a document.
    """
    source_path: str
    format_type: str                         # "PDF", "DOCX", etc.
    page_count: int

    text_blocks:    list[DocumentComponent] = field(default_factory=list)
    images:         list[DocumentComponent] = field(default_factory=list)
    vectors:        list[DocumentComponent] = field(default_factory=list)
    fonts:          list[DocumentComponent] = field(default_factory=list)
    metadata_block: Optional[DocumentComponent] = None
    layout_block:   Optional[DocumentComponent] = None

    # Components identified as duplicates of an earlier component
    # Maps: content_hash → list of page numbers where it recurs
    duplicate_map:  dict[str, list[int]] = field(default_factory=dict)

    @property
    def all_components(self) -> list[DocumentComponent]:
        parts: list[DocumentComponent] = []
        parts.extend(self.text_blocks)
        parts.extend(self.images)
        parts.extend(self.vectors)
        parts.extend(self.fonts)
        if self.metadata_block:
            parts.append(self.metadata_block)
        if self.layout_block:
            parts.append(self.layout_block)
        return parts

    @property
    def total_raw_size(self) -> int:
        return sum(c.size for c in self.all_components)

    @property
    def unique_image_count(self) -> int:
        hashes = {c.content_hash for c in self.images}
        return len(hashes)

    @property
    def duplicate_image_count(self) -> int:
        return len(self.images) - self.unique_image_count

    def summary(self) -> str:
        lines = [
            f"Document    : {os.path.basename(self.source_path)}",
            f"Format      : {self.format_type}",
            f"Pages       : {self.page_count}",
            f"Text blocks : {len(self.text_blocks)}",
            f"Images      : {len(self.images)} "
            f"({self.unique_image_count} unique, "
            f"{self.duplicate_image_count} duplicates)",
            f"Fonts       : {len(self.fonts)}",
            f"Raw size    : {self.total_raw_size / 1024 / 1024:.2f} MB",
        ]
        if self.duplicate_map:
            saved = sum(
                len(pages) - 1
                for pages in self.duplicate_map.values()
                if len(pages) > 1
            )
            lines.append(f"Dedup saves : {saved} redundant copies detected")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF Decomposer
# ---------------------------------------------------------------------------

class PDFDecomposer:
    """
    Extracts semantic components from a PDF file.

    Requires: pypdf  (already in requirements/base.txt)
    Optional: Pillow (already in requirements/base.txt) for image export
    """

    def decompose(self, pdf_path: str) -> DecomposedDocument:
        """
        Full decomposition pipeline.
        """
        self._check_path(pdf_path)

        try:
            import pypdf
        except ImportError as exc:
            raise FormatError(
                "pypdf is required for PDF decomposition. "
                "Install it with: pip install pypdf"
            ) from exc

        reader = pypdf.PdfReader(pdf_path)
        page_count = len(reader.pages)

        doc = DecomposedDocument(
            source_path=pdf_path,
            format_type="PDF",
            page_count=page_count,
        )

        # Run extraction passes
        self._extract_metadata(reader, doc)
        self._extract_text(reader, doc)
        self._extract_images(reader, doc)
        self._extract_fonts(reader, doc)
        self._extract_layout(reader, doc)
        self._find_duplicates(doc)

        return doc

    # ------------------------------------------------------------------
    # Extraction passes
    # ------------------------------------------------------------------

    def _extract_metadata(
        self, reader, doc: DecomposedDocument
    ) -> None:
        """
        Pulls document-level metadata: title, author, dates, etc.
        Stored as a JSON-encoded text component.
        """
        import json

        meta = {}
        if reader.metadata:
            for key, value in reader.metadata.items():
                clean_key = key.lstrip("/")
                try:
                    meta[clean_key] = str(value)
                except Exception:
                    pass

        # Page geometry info
        if reader.pages:
            first_page = reader.pages[0]
            try:
                mb = first_page.mediabox
                meta["page_width"]  = float(mb.width)
                meta["page_height"] = float(mb.height)
            except Exception:
                pass

        meta["page_count"] = doc.page_count

        payload = json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8")
        doc.metadata_block = DocumentComponent(
            component_type=ComponentType.METADATA,
            data=payload,
            meta={"fields": list(meta.keys())},
        )

    def _extract_text(
        self, reader, doc: DecomposedDocument
    ) -> None:
        """
        Extracts text content from every page.
        Each page becomes one TEXT component.
        Text is stored as UTF-8.
        """
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            if not text.strip():
                continue

            payload = text.encode("utf-8")
            doc.text_blocks.append(
                DocumentComponent(
                    component_type=ComponentType.TEXT,
                    data=payload,
                    pages=[page_num],
                    meta={
                        "char_count": len(text),
                        "line_count": text.count("\n"),
                    },
                )
            )

    def _extract_images(
        self, reader, doc: DecomposedDocument
    ) -> None:
        """
        Extracts all embedded images from every page.
        Each image is exported as PNG bytes for uniform handling.

        Uses pypdf's page.images API (pypdf >= 3.x).
        Falls back to xObject traversal for older structures.
        """
        for page_num, page in enumerate(reader.pages, start=1):
            self._extract_images_from_page(page, page_num, doc)

    def _extract_images_from_page(
        self,
        page,
        page_num: int,
        doc: DecomposedDocument,
    ) -> None:
        """
        Extract images from a single page using pypdf's image API.
        """
        try:
            # pypdf >= 3.4 exposes page.images as a list-like
            for img_obj in page.images:
                self._process_image_object(img_obj, page_num, doc)
        except AttributeError:
            # Older pypdf: manual xObject traversal
            self._extract_images_via_xobjects(page, page_num, doc)
        except Exception:
            pass  # Non-fatal: skip unreadable images

    def _process_image_object(
        self, img_obj, page_num: int, doc: DecomposedDocument
    ) -> None:
        """
        Convert a pypdf image object to a PNG-encoded DocumentComponent.
        """
        try:
            from PIL import Image as PILImage

            # pypdf image objects expose .data (raw bytes) and .name
            raw = img_obj.data
            name = getattr(img_obj, "name", f"img_{page_num}")

            # Attempt to open and re-encode as PNG for uniformity
            try:
                pil_img = PILImage.open(io.BytesIO(raw))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                width, height = pil_img.size
                mode = pil_img.mode
            except Exception:
                # Could not decode — store raw bytes as-is
                png_bytes = raw
                width = height = 0
                mode = "unknown"

            doc.images.append(
                DocumentComponent(
                    component_type=ComponentType.IMAGE,
                    data=png_bytes,
                    pages=[page_num],
                    meta={
                        "name": name,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "original_size": len(raw),
                    },
                )
            )
        except Exception:
            pass

    def _extract_images_via_xobjects(
        self, page, page_num: int, doc: DecomposedDocument
    ) -> None:
        """
        Fallback image extraction via raw PDF xObjects.
        """
        try:
            resources = page.get("/Resources")
            if not resources:
                return
            xobjects = resources.get("/XObject")
            if not xobjects:
                return

            for name, obj_ref in xobjects.items():
                try:
                    obj = obj_ref.get_object()
                    if obj.get("/Subtype") == "/Image":
                        raw = obj.get_data()
                        doc.images.append(
                            DocumentComponent(
                                component_type=ComponentType.IMAGE,
                                data=raw,
                                pages=[page_num],
                                meta={
                                    "name": str(name),
                                    "width": int(obj.get("/Width", 0)),
                                    "height": int(obj.get("/Height", 0)),
                                    "color_space": str(obj.get("/ColorSpace", "")),
                                },
                            )
                        )
                except Exception:
                    continue
        except Exception:
            pass

    def _extract_fonts(
        self, reader, doc: DecomposedDocument
    ) -> None:
        """
        Collects all font references across the document.
        Identifies embedded vs. non-embedded fonts.
        Embedded font data is stored as binary blobs for
        potential subset optimisation in a later phase.
        """
        seen_fonts: set[str] = set()

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                resources = page.get("/Resources")
                if not resources:
                    continue
                fonts_dict = resources.get("/Font")
                if not fonts_dict:
                    continue

                for font_name, font_ref in fonts_dict.items():
                    try:
                        font_obj = font_ref.get_object()
                        base_font = str(font_obj.get("/BaseFont", font_name))

                        if base_font in seen_fonts:
                            continue
                        seen_fonts.add(base_font)

                        # Check if font is embedded
                        descriptor = font_obj.get("/FontDescriptor")
                        is_embedded = False
                        font_bytes  = b""

                        if descriptor:
                            desc_obj = descriptor.get_object()
                            for stream_key in ("/FontFile", "/FontFile2", "/FontFile3"):
                                ff = desc_obj.get(stream_key)
                                if ff:
                                    is_embedded = True
                                    try:
                                        font_bytes = ff.get_object().get_data()
                                    except Exception:
                                        pass
                                    break

                        doc.fonts.append(
                            DocumentComponent(
                                component_type=ComponentType.FONT,
                                data=font_bytes,
                                pages=[page_num],
                                meta={
                                    "base_font": base_font,
                                    "is_embedded": is_embedded,
                                    "font_type": str(font_obj.get("/Subtype", "Unknown")),
                                },
                            )
                        )
                    except Exception:
                        continue
            except Exception:
                continue

    def _extract_layout(
        self, reader, doc: DecomposedDocument
    ) -> None:
        """
        Captures high-level page layout structure as JSON.
        Records page dimensions and text block positions.
        Minimal implementation for Phase 3 —
        full spatial analysis is a Phase 4 concern.
        """
        import json

        pages_layout = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                mb = page.mediabox
                pages_layout.append({
                    "page": page_num,
                    "width":  float(mb.width),
                    "height": float(mb.height),
                })
            except Exception:
                pages_layout.append({"page": page_num})

        payload = json.dumps(
            {"pages": pages_layout}, indent=2
        ).encode("utf-8")

        doc.layout_block = DocumentComponent(
            component_type=ComponentType.LAYOUT,
            data=payload,
            meta={"page_count": doc.page_count},
        )

    def _find_duplicates(self, doc: DecomposedDocument) -> None:
        """
        Scans all image components and builds a map of
        hash → [page numbers] for repeated images.

        This powers the deduplication step in the compressor:
        each unique image is stored once; duplicate occurrences
        store only a reference (hash + page position).
        """
        hash_to_pages: dict[str, list[int]] = {}

        for component in doc.images:
            h = component.content_hash
            pages = component.pages
            if h not in hash_to_pages:
                hash_to_pages[h] = []
            hash_to_pages[h].extend(pages)

        # Only keep hashes with more than one occurrence
        doc.duplicate_map = {
            h: pages
            for h, pages in hash_to_pages.items()
            if len(pages) > 1
        }


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------

class DocumentDecomposer:
    """
    Public facade. Dispatches to format-specific decomposers.

    Usage::

        decomposer = DocumentDecomposer()
        result = decomposer.decompose("report.pdf")
        print(result.summary())
    """

    def __init__(self) -> None:
        self._pdf = PDFDecomposer()

    def decompose(self, document_path: str) -> DecomposedDocument:
        ext = os.path.splitext(document_path)[1].lower()

        if ext == ".pdf":
            return self._pdf.decompose(document_path)

        raise FormatError(
            f"DocumentDecomposer: unsupported format '{ext}'. "
            f"Supported: .pdf  (DOCX/XLSX coming in Phase 4)"
        )

    @staticmethod
    def _check_path(path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

# Assign the method correctly to the instances or class after definition
PDFDecomposer._check_path = staticmethod(DocumentDecomposer._check_path)
