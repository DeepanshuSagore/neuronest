"""Loader behaviour, with the failure cases given equal weight to the happy path.

The fixtures are built in-process rather than committed as binary files. A
checked-in PDF is opaque in review, and a scanned one large; generating them
here keeps what each test is actually about visible in the test itself.
"""

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from neuronest.ingest.loaders import (
    Document,
    LoadError,
    LoadErrorCode,
    document_id,
    load_directory,
    load_document,
)


def build_pdf_with_text(text: str) -> bytes:
    """A minimal one-page PDF whose text layer contains ``text``.

    Hand-built because pypdf can create pages but cannot write text onto them,
    and pulling in a PDF *writer* to test a PDF *reader* would be a heavier
    dependency than the thing under test.
    """
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return bytes(out)


def write_pdf_with_text(path: Path, text: str) -> Path:
    path.write_bytes(build_pdf_with_text(text))
    return path


def write_scanned_pdf(path: Path, pages: int = 3) -> Path:
    """A PDF of blank pages, which is what a scan looks like to a text extractor."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    writer.write(path)
    return path


def write_encrypted_pdf(path: Path, password: str = "hunter2") -> Path:
    """Encrypt an in-memory PDF rather than an intermediate file.

    Writing the unencrypted source next to the encrypted one would leave a
    stray document in the folder, which any test that later walks that folder
    would pick up and count.
    """
    writer = PdfWriter()
    writer.append(BytesIO(build_pdf_with_text("Secret contents")))
    writer.encrypt(password)
    writer.write(path)
    return path


def as_document(result: Document | LoadError) -> Document:
    assert isinstance(result, Document), f"expected a Document, got {result!r}"
    return result


def as_error(result: Document | LoadError) -> LoadError:
    assert isinstance(result, LoadError), f"expected a LoadError, got {result!r}"
    return result


# --- the happy path ---------------------------------------------------------


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_loads_plain_text_and_markdown(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"notes{suffix}"
    path.write_text("Retrieval is not generation.", encoding="utf-8")

    document = as_document(load_document(path))

    assert document.text == "Retrieval is not generation."
    assert document.source_path == path
    assert document.metadata["filename"] == f"notes{suffix}"
    assert document.metadata["suffix"] == suffix


def test_loads_pdf_text_layer(tmp_path: Path) -> None:
    path = write_pdf_with_text(tmp_path / "paper.pdf", "Loss scales as a power law")

    document = as_document(load_document(path))

    assert "Loss scales as a power law" in document.text


# --- the three failure cases the spec calls out -----------------------------


def test_zero_byte_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.touch()

    assert as_error(load_document(path)).code is LoadErrorCode.EMPTY_FILE


def test_whitespace_only_file_is_rejected(tmp_path: Path) -> None:
    """Not zero bytes, but indexes to exactly as much as a zero-byte file does."""
    path = tmp_path / "blank.md"
    path.write_text("\n\n   \t\n", encoding="utf-8")

    assert as_error(load_document(path)).code is LoadErrorCode.EMPTY_FILE


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    path = write_encrypted_pdf(tmp_path / "locked.pdf")

    error = as_error(load_document(path))

    assert error.code is LoadErrorCode.ENCRYPTED
    assert "password" in error.message.lower()


def test_scanned_pdf_is_rejected_rather_than_indexed_empty(tmp_path: Path) -> None:
    """The trap: extraction succeeds, returns nothing, and raises nothing."""
    path = write_scanned_pdf(tmp_path / "scan.pdf", pages=3)

    error = as_error(load_document(path))

    assert error.code is LoadErrorCode.NO_TEXT_LAYER
    assert "ocr" in error.message.lower()


# --- the rest of the surface ------------------------------------------------


def test_unsupported_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sheet.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    error = as_error(load_document(path))

    assert error.code is LoadErrorCode.UNSUPPORTED_TYPE
    assert ".pdf" in error.message


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    assert as_error(load_document(tmp_path / "nope.txt")).code is LoadErrorCode.NOT_FOUND


def test_directory_passed_as_a_file_is_rejected(tmp_path: Path) -> None:
    assert as_error(load_document(tmp_path)).code is LoadErrorCode.NOT_A_FILE


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "latin.txt"
    path.write_bytes(b"caf\xe9 society")

    assert as_error(load_document(path)).code is LoadErrorCode.UNREADABLE


def test_malformed_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a pdf body")

    assert as_error(load_document(path)).code is LoadErrorCode.UNREADABLE


# --- identity ---------------------------------------------------------------


def test_document_id_is_stable_across_runs() -> None:
    assert document_id("papers/scaling.pdf") == document_id("papers/scaling.pdf")


def test_document_id_is_independent_of_path_separator() -> None:
    """Windows and POSIX must agree, or a corpus labelled on one breaks on the other."""
    assert document_id("papers\\scaling.pdf") == document_id("papers/scaling.pdf")


def test_document_id_distinguishes_same_name_in_different_folders() -> None:
    assert document_id("a/notes.md") != document_id("b/notes.md")


def test_document_id_survives_an_edit(tmp_path: Path) -> None:
    """Ids are path-derived, so re-ingesting an edited file updates instead of duplicating."""
    path = tmp_path / "notes.md"

    path.write_text("first draft", encoding="utf-8")
    before = as_document(load_document(path)).doc_id

    path.write_text("second draft, entirely rewritten", encoding="utf-8")
    after = as_document(load_document(path)).doc_id

    assert before == after


# --- folders ----------------------------------------------------------------


def test_load_directory_separates_documents_from_errors(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "good.txt").write_text("indexable", encoding="utf-8")
    (tmp_path / "nested" / "also-good.md").write_text("also indexable", encoding="utf-8")
    write_pdf_with_text(tmp_path / "paper.pdf", "Readable text layer")
    (tmp_path / "empty.txt").touch()
    write_scanned_pdf(tmp_path / "scan.pdf")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")

    report = load_directory(tmp_path)

    assert len(report.documents) == 3
    assert {error.code for error in report.errors} == {
        LoadErrorCode.EMPTY_FILE,
        LoadErrorCode.NO_TEXT_LAYER,
    }
    # The csv is skipped by the walk rather than reported: a corpus folder with
    # unrelated files in it should not produce noise for every one of them.
    assert all(error.source_path.suffix != ".csv" for error in report.errors)


def test_load_directory_ids_are_relative_to_the_root(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "notes.md").write_text("content", encoding="utf-8")

    report = load_directory(tmp_path)

    assert [document.doc_id for document in report.documents] == [document_id("papers/notes.md")]


def test_load_directory_is_deterministic(tmp_path: Path) -> None:
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    first = load_directory(tmp_path)
    second = load_directory(tmp_path)

    assert [document.doc_id for document in first.documents] == [
        document.doc_id for document in second.documents
    ]


def test_load_directory_reports_a_missing_root(tmp_path: Path) -> None:
    report = load_directory(tmp_path / "absent")

    assert report.documents == []
    assert [error.code for error in report.errors] == [LoadErrorCode.NOT_FOUND]
