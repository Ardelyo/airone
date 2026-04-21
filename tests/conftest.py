"""
Global pytest configuration and shared fixtures.
"""
from __future__ import annotations

import io
import os
import zipfile
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Shared synthetic data fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def repetitive_text_bytes() -> bytes:
    """1 MB of highly compressible repetitive text."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "AirOne semantic compression platform. "
    ).encode() * 15000


@pytest.fixture(scope="session")
def random_bytes_small() -> bytes:
    """4 KB of pseudo-random bytes."""
    return os.urandom(4096)


@pytest.fixture(scope="session")
def random_bytes_medium() -> bytes:
    """128 KB of pseudo-random bytes."""
    return os.urandom(128 * 1024)


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Minimal syntactically valid PDF."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f\n"
        b"0000000009 00000 n\n"
        b"0000000068 00000 n\n"
        b"0000000125 00000 n\n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n210\n%%EOF\n"
    )


@pytest.fixture
def minimal_docx_bytes() -> bytes:
    """Minimal valid DOCX in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument'
            '.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>AirOne test document content repeated many times. </w:t></w:r></w:p>" * 30
            + "</w:body></w:document>",
        )
    return buf.getvalue()


@pytest.fixture
def tmp_text_file(tmp_path) -> str:
    """A temporary text file on disk."""
    path = tmp_path / "sample.txt"
    path.write_bytes(
        b"AirOne lossless compression. " * 500
    )
    return str(path)


@pytest.fixture
def tmp_binary_file(tmp_path) -> str:
    """A temporary binary file on disk."""
    path = tmp_path / "sample.bin"
    data = bytes(range(256)) * 200
    path.write_bytes(data)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Markers
# ──────────────────────────────────────────────────────────────────────────────

def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "ml: marks tests that require PyTorch/ONNX")
    config.addinivalue_line("markers", "integration: full end-to-end pipeline tests")
