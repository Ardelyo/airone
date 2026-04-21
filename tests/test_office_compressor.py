"""Tests for Office Document Semantic Compressor."""

from __future__ import annotations

import io
import zipfile
import pytest

from airone.compressors.semantic.office import OfficeSemanticCompressor
from airone.exceptions import CompressionError


def _make_docx(tmp_path, text: str = "Hello AirOne") -> tuple[bytes, str]:
    """Create a minimal valid DOCX file."""
    path = tmp_path / "test.docx"
    buf  = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Required content types
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        # Relationships
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        # Main document
        zf.writestr(
            "word/document.xml",
            f'<?xml version="1.0"?>'
            f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>"
            f"</w:document>",
        )

    data = buf.getvalue()
    path.write_bytes(data)
    return data, str(path)


def _make_xlsx(tmp_path) -> tuple[bytes, str]:
    """Create a minimal valid XLSX file."""
    path = tmp_path / "test.xlsx"
    buf  = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheets><sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
            "</workbook>",
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>"
            + "".join(
                f"<row r=\"{i}\"><c r=\"A{i}\" t=\"n\"><v>{i * 100}</v></c></row>"
                for i in range(1, 51)
            )
            + "</sheetData></worksheet>",
        )

    data = buf.getvalue()
    path.write_bytes(data)
    return data, str(path)


class TestOfficeFormatDetection:

    def test_detects_docx(self, tmp_path):
        from airone.compressors.semantic.office import OfficeFormat
        data, _ = _make_docx(tmp_path)
        zf = zipfile.ZipFile(io.BytesIO(data))
        assert OfficeFormat.detect(zf) == OfficeFormat.DOCX

    def test_detects_xlsx(self, tmp_path):
        from airone.compressors.semantic.office import OfficeFormat
        data, _ = _make_xlsx(tmp_path)
        zf = zipfile.ZipFile(io.BytesIO(data))
        assert OfficeFormat.detect(zf) == OfficeFormat.XLSX

    def test_xml_classification(self):
        from airone.compressors.semantic.office import OfficeFormat
        assert OfficeFormat.is_xml("word/document.xml") is True
        assert OfficeFormat.is_xml("_rels/.rels") is True
        assert OfficeFormat.is_xml("image.png") is False

    def test_media_classification(self):
        from airone.compressors.semantic.office import OfficeFormat
        assert OfficeFormat.is_media("media/image1.png") is True
        assert OfficeFormat.is_media("word/document.xml") is False


class TestOfficeSemanticCompressor:

    def test_can_handle_docx_analysis(self):
        from unittest.mock import MagicMock
        from airone.analysis.format_detector import FileFormat, FileCategory

        compressor = OfficeSemanticCompressor()
        analysis   = MagicMock()
        analysis.format = FileFormat(
            type="DOCX",
            mime_type="application/vnd.openxmlformats-officedocument",
            category=FileCategory.DOCUMENT,
        )
        assert compressor.can_handle(analysis) is True

    def test_cannot_handle_pdf(self):
        from unittest.mock import MagicMock
        from airone.analysis.format_detector import FileFormat, FileCategory

        compressor = OfficeSemanticCompressor()
        analysis   = MagicMock()
        analysis.format = FileFormat(
            type="PDF",
            mime_type="application/pdf",
            category=FileCategory.DOCUMENT,
        )
        assert compressor.can_handle(analysis) is False

    def test_compress_docx(self, tmp_path):
        data, _ = _make_docx(tmp_path, text="AirOne test document " * 50)
        compressor = OfficeSemanticCompressor()
        result = compressor.compress(data)

        assert result.compressed_size > 0
        assert result.strategy_name == "semantic_office"
        assert result.original_size == len(data)

    def test_compress_xlsx(self, tmp_path):
        data, _ = _make_xlsx(tmp_path)
        compressor = OfficeSemanticCompressor()
        result = compressor.compress(data)

        assert result.compressed_size > 0
        assert result.strategy_name == "semantic_office"

    def test_decompress_roundtrip_docx(self, tmp_path):
        data, _ = _make_docx(tmp_path)
        compressor = OfficeSemanticCompressor()
        result = compressor.compress(data)
        decompressed = compressor.decompress(result.compressed_data, result.metadata)

        # Verify result is a valid ZIP (DOCX)
        assert zipfile.is_zipfile(io.BytesIO(decompressed))

        # Verify word/document.xml is present
        with zipfile.ZipFile(io.BytesIO(decompressed)) as zf:
            assert "word/document.xml" in zf.namelist()

    def test_decompress_roundtrip_xlsx(self, tmp_path):
        data, _ = _make_xlsx(tmp_path)
        compressor = OfficeSemanticCompressor()
        result     = compressor.compress(data)
        decompressed = compressor.decompress(result.compressed_data, result.metadata)

        with zipfile.ZipFile(io.BytesIO(decompressed)) as zf:
            assert "xl/workbook.xml" in zf.namelist()

    def test_invalid_zip_raises(self):
        compressor = OfficeSemanticCompressor()
        with pytest.raises(CompressionError, match="valid ZIP"):
            compressor.compress(b"not a zip file at all")
