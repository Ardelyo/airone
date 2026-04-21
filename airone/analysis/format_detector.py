"""
AirOne Format Detector
Identifies file types beyond simple extension checking.
Uses magic bytes, structural probing, and content inspection.
"""

import os
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from airone.exceptions import FormatError


class FileCategory(str, Enum):
    IMAGE       = "image"
    DOCUMENT    = "document"
    CAD         = "cad"
    MEDICAL     = "medical"
    AUDIO       = "audio"
    VIDEO       = "video"
    ARCHIVE     = "archive"
    DATA        = "data"
    TEXT        = "text"
    UNKNOWN     = "unknown"


@dataclass
class FileFormat:
    """
    Represents a fully resolved file format.
    """
    # e.g. "PNG", "PDF", "DWG"
    type: str

    # e.g. "image/png", "application/pdf"
    mime_type: str

    # High-level category
    category: FileCategory

    # e.g. "1.4" for PDF 1.4
    version: Optional[str] = None

    # 0.0 – 1.0 confidence in the detection
    confidence: float = 1.0

    # Hints that inform strategy selection
    hints: dict = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.category == FileCategory.IMAGE

    @property
    def is_document(self) -> bool:
        return self.category == FileCategory.DOCUMENT

    @property
    def is_cad(self) -> bool:
        return self.category == FileCategory.CAD

    @property
    def is_medical(self) -> bool:
        return self.category == FileCategory.MEDICAL


# ---------------------------------------------------------------------------
# Magic-byte signatures
# Each entry: (offset, bytes_to_match, format_type, mime, category, version)
# ---------------------------------------------------------------------------
_MAGIC_SIGNATURES: list[tuple] = [
    # Images
    (0, b'\x89PNG\r\n\x1a\n',  "PNG",  "image/png",        FileCategory.IMAGE,    None),
    (0, b'\xff\xd8\xff',        "JPEG", "image/jpeg",       FileCategory.IMAGE,    None),
    (0, b'GIF87a',              "GIF",  "image/gif",        FileCategory.IMAGE,    "87a"),
    (0, b'GIF89a',              "GIF",  "image/gif",        FileCategory.IMAGE,    "89a"),
    (0, b'RIFF',                "WEBP", "image/webp",       FileCategory.IMAGE,    None),  # refined below
    (0, b'BM',                  "BMP",  "image/bmp",        FileCategory.IMAGE,    None),
    (0, b'II\x2a\x00',         "TIFF", "image/tiff",       FileCategory.IMAGE,    None),
    (0, b'MM\x00\x2a',         "TIFF", "image/tiff",       FileCategory.IMAGE,    None),
    (0, b'\x00\x00\x00\x0cjP', "JPEG2000", "image/jp2",   FileCategory.IMAGE,    None),

    # Documents
    (0, b'%PDF-',               "PDF",  "application/pdf",  FileCategory.DOCUMENT, None),
    (0, b'PK\x03\x04',         "ZIP_OFFICE", "application/zip", FileCategory.DOCUMENT, None),  # DOCX/XLSX etc
    (0, b'\xd0\xcf\x11\xe0',   "OLE",  "application/msword", FileCategory.DOCUMENT, None),  # DOC/XLS/PPT

    # Medical
    (128, b'DICM',              "DICOM","application/dicom",FileCategory.MEDICAL,  None),

    # Archives
    (0, b'PK\x03\x04',         "ZIP",  "application/zip",  FileCategory.ARCHIVE,  None),
    (0, b'\x1f\x8b',           "GZIP", "application/gzip", FileCategory.ARCHIVE,  None),
    (0, b'BZh',                "BZ2",  "application/bzip2",FileCategory.ARCHIVE,  None),
    (0, b'7z\xbc\xaf\x27\x1c', "7ZIP", "application/7z",  FileCategory.ARCHIVE,  None),
    (0, b'Rar!\x1a\x07',       "RAR",  "application/rar",  FileCategory.ARCHIVE,  None),
]


class FormatDetector:
    """
    Identifies file formats using multi-strategy detection.

    Detection pipeline:
        1. Magic-byte matching (fast, reliable for known formats)
        2. Structural probing (for ambiguous or complex formats)
        3. Extension fallback (last resort)
    """

    # Minimum bytes needed to run magic detection
    _READ_AHEAD = 256

    def detect(self, file_path: str) -> FileFormat:
        """
        Detect the format of *file_path*.
        Returns a :class:`FileFormat` describing the result.
        Raises :class:`FormatError` if the file cannot be read.
        """
        if not os.path.isfile(file_path):
            raise FormatError(f"File not found: {file_path}")

        header = self._read_header(file_path)
        fmt = self._match_magic(header)

        if fmt is None:
            fmt = self._probe_structure(file_path, header)

        if fmt is None:
            fmt = self._fallback_extension(file_path)

        fmt = self._refine(fmt, file_path, header)
        return fmt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_header(self, file_path: str) -> bytes:
        try:
            with open(file_path, "rb") as fh:
                return fh.read(self._READ_AHEAD)
        except OSError as exc:
            raise FormatError(f"Cannot read file: {exc}") from exc

    def _match_magic(self, header: bytes) -> Optional[FileFormat]:
        for offset, magic, fmt_type, mime, category, version in _MAGIC_SIGNATURES:
            end = offset + len(magic)
            if len(header) >= end and header[offset:end] == magic:
                return FileFormat(
                    type=fmt_type,
                    mime_type=mime,
                    category=category,
                    version=version,
                    confidence=0.95,
                )
        return None

    def _probe_structure(self, file_path: str, header: bytes) -> Optional[FileFormat]:
        """
        Deeper structural inspection for formats that need it
        e.g. Office Open XML formats (DOCX, XLSX, PPTX) are all ZIP files
        """
        # Office Open XML — ZIP archive containing [Content_Types].xml
        if header[:4] == b'PK\x03\x04':
            return self._probe_office_xml(file_path)

        # DWG — AutoCAD
        if header[:6] == b'AC1015' or header[:6] == b'AC1024':
            return FileFormat(
                type="DWG",
                mime_type="image/vnd.dwg",
                category=FileCategory.CAD,
                version=header[:6].decode("ascii"),
                confidence=0.98,
            )

        return None

    def _probe_office_xml(self, file_path: str) -> FileFormat:
        """
        Distinguishes DOCX / XLSX / PPTX by inspecting the ZIP manifest.
        """
        import zipfile

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()

            if "word/document.xml" in names:
                return FileFormat("DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                  FileCategory.DOCUMENT, confidence=0.99)
            if "xl/workbook.xml" in names:
                return FileFormat("XLSX", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  FileCategory.DOCUMENT, confidence=0.99)
            if "ppt/presentation.xml" in names:
                return FileFormat("PPTX", "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                  FileCategory.DOCUMENT, confidence=0.99)
        except zipfile.BadZipFile:
            pass

        return FileFormat("ZIP", "application/zip", FileCategory.ARCHIVE, confidence=0.90)

    def _fallback_extension(self, file_path: str) -> FileFormat:
        ext_map = {
            ".txt": ("TXT",  "text/plain",       FileCategory.TEXT),
            ".csv": ("CSV",  "text/csv",         FileCategory.DATA),
            ".json":("JSON", "application/json", FileCategory.DATA),
            ".xml": ("XML",  "application/xml",  FileCategory.DATA),
            ".svg": ("SVG",  "image/svg+xml",    FileCategory.IMAGE),
            ".dwg": ("DWG",  "image/vnd.dwg",    FileCategory.CAD),
            ".dxf": ("DXF",  "image/vnd.dxf",    FileCategory.CAD),
            ".dcm": ("DICOM","application/dicom",FileCategory.MEDICAL),
        }
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ext_map:
            fmt_type, mime, category = ext_map[ext]
            return FileFormat(fmt_type, mime, category, confidence=0.60)

        return FileFormat("UNKNOWN", "application/octet-stream", FileCategory.UNKNOWN, confidence=0.0)

    def _refine(self, fmt: FileFormat, file_path: str, header: bytes) -> FileFormat:
        """
        Post-processing refinements — e.g. extract PDF version, JPEG sub-type.
        """
        if fmt.type == "PDF":
            # Extract version from header e.g. b'%PDF-1.7'
            try:
                version_str = header[5:8].decode("ascii")
                fmt.version = version_str
            except Exception:
                pass

        if fmt.type == "WEBP":
            # RIFF....WEBP
            if len(header) >= 12 and header[8:12] == b'WEBP':
                fmt.type = "WEBP"
                fmt.mime_type = "image/webp"
            else:
                fmt.type = "RIFF"
                fmt.mime_type = "audio/x-wav"
                fmt.category = FileCategory.AUDIO

        return fmt
