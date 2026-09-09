"""Loading PDF, Markdown and plain text into one ``Document`` shape.

Loading is the only stage that touches the outside world, so it is the stage
where things are actually broken: a PDF nobody can open, a file someone
truncated to nothing, a password-protected export. None of those raise here.
A caller ingesting a folder wants the twenty documents that loaded *and* an
account of the three that did not, which means failure has to be a value rather
than an exception.

Hence ``LoadResult``: every load returns either a ``Document`` or a
``LoadError`` carrying a machine-readable code and a sentence a human can act
on. The API layer in phase 8 turns those codes into per-file status without
re-deriving anything.
"""

import hashlib
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict
from pypdf import PasswordType, PdfReader
from pypdf.errors import PdfReadError, PyPdfError

TEXT_SUFFIXES = frozenset({".txt", ".md"})
PDF_SUFFIX = ".pdf"
SUPPORTED_SUFFIXES = frozenset({PDF_SUFFIX}) | TEXT_SUFFIXES

# Below this many characters per page, a PDF is treated as having no text layer.
# Deliberately tiny: a scan yields zero, so the threshold only has to separate
# "nothing" from "something", and anything higher would start rejecting real
# documents that happen to have sparse pages.
MIN_CHARS_PER_PAGE = 8


class LoadErrorCode(StrEnum):
    """Why a file did not become a document.

    Codes are stable strings because they cross a process boundary in phase 8
    and end up in ingest reports; the human message beside them is not stable
    and should never be parsed.
    """

    NOT_FOUND = "not_found"
    NOT_A_FILE = "not_a_file"
    UNSUPPORTED_TYPE = "unsupported_type"
    EMPTY_FILE = "empty_file"
    ENCRYPTED = "encrypted"
    NO_TEXT_LAYER = "no_text_layer"
    UNREADABLE = "unreadable"


class Document(BaseModel):
    """One source file, normalised to text.

    Frozen because everything downstream — chunk ids, embeddings, the vector
    store — is derived from these fields, and a document mutated after loading
    would silently invalidate all of it.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: Path
    text: str
    metadata: dict[str, str | int]


class LoadError(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: Path
    code: LoadErrorCode
    message: str


LoadResult = Document | LoadError


def document_id(identity: str) -> str:
    """Derive a stable document id from a corpus-relative identity.

    Keyed on the path *inside the corpus*, never the absolute path: an
    absolute path differs between a laptop and a container, and phase 11 labels
    evaluation answers by chunk id, which is derived from this. Ids that move
    when the corpus does would invalidate every label in the set.

    Deriving from the path rather than the content is equally deliberate.
    Editing a document has to keep its id so that re-ingesting *updates* the
    existing chunks, which is exactly what phase 6 asserts.
    """
    normalised = str(PurePosixPath(identity.replace("\\", "/")))
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _error(path: Path, code: LoadErrorCode, message: str) -> LoadError:
    return LoadError(source_path=path, code=code, message=message)


def _load_text_file(path: Path) -> str | LoadError:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _error(
            path,
            LoadErrorCode.UNREADABLE,
            "Not valid UTF-8. Re-save the file as UTF-8 before ingesting it.",
        )
    except OSError as exc:
        return _error(path, LoadErrorCode.UNREADABLE, f"Could not be read: {exc.strerror or exc}.")


def _load_pdf(path: Path) -> str | LoadError:
    try:
        reader = PdfReader(path)
    except (PyPdfError, OSError, ValueError) as exc:
        return _error(path, LoadErrorCode.UNREADABLE, f"Not a readable PDF: {exc}.")

    if reader.is_encrypted:
        # An empty user password is common in exports that are technically
        # encrypted but not actually protected, so try it before giving up.
        try:
            unlocked = reader.decrypt("")
        except (PyPdfError, NotImplementedError) as exc:
            return _error(
                path, LoadErrorCode.ENCRYPTED, f"Encrypted and could not be opened: {exc}."
            )
        if unlocked == PasswordType.NOT_DECRYPTED:
            return _error(
                path,
                LoadErrorCode.ENCRYPTED,
                "Password protected. Remove the password and ingest it again.",
            )

    try:
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as exc:
        return _error(path, LoadErrorCode.UNREADABLE, f"Text extraction failed: {exc}.")

    text = "\n\n".join(pages)

    # The trap this whole function exists to avoid. A scanned PDF is a stack of
    # images: extraction succeeds, returns empty strings, and raises nothing. It
    # would index as a document with no content, and then read as a retrieval
    # bug three phases later rather than as an ingest problem now.
    page_count = len(pages)
    if page_count > 0 and len(text.strip()) < MIN_CHARS_PER_PAGE * page_count:
        return _error(
            path,
            LoadErrorCode.NO_TEXT_LAYER,
            f"No extractable text across {page_count} page(s) — it is almost certainly a scan. "
            "Run OCR over it and ingest the result.",
        )

    return text


def load_document(path: Path, *, identity: str | None = None) -> LoadResult:
    """Load one file, returning either a ``Document`` or a ``LoadError``.

    ``identity`` is the corpus-relative name the id is derived from; it
    defaults to the file name. ``load_directory`` passes the path relative to
    the corpus root so that two files sharing a name in different folders do
    not collide.
    """
    if not path.exists():
        return _error(path, LoadErrorCode.NOT_FOUND, "No such file.")
    if not path.is_file():
        return _error(path, LoadErrorCode.NOT_A_FILE, "Not a file.")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        return _error(
            path,
            LoadErrorCode.UNSUPPORTED_TYPE,
            f"Unsupported file type '{suffix or path.name}'. Supported: {supported}.",
        )

    size = path.stat().st_size
    if size == 0:
        return _error(path, LoadErrorCode.EMPTY_FILE, "File is empty.")

    extracted = _load_pdf(path) if suffix == PDF_SUFFIX else _load_text_file(path)
    if isinstance(extracted, LoadError):
        return extracted

    # A file of nothing but whitespace indexes as nothing and then looks like a
    # retrieval bug later, so it is rejected here alongside the zero-byte case.
    if not extracted.strip():
        return _error(path, LoadErrorCode.EMPTY_FILE, "No text content.")

    metadata: dict[str, str | int] = {
        "filename": path.name,
        "suffix": suffix,
        "bytes": size,
    }

    return Document(
        doc_id=document_id(identity if identity is not None else path.name),
        source_path=path,
        text=extracted,
        metadata=metadata,
    )


class LoadReport(BaseModel):
    """The outcome of loading a folder: what worked and what did not.

    Both halves are returned together because a caller that only sees the
    documents cannot tell an empty corpus from a corpus where every file
    failed.
    """

    model_config = ConfigDict(frozen=True)

    documents: list[Document]
    errors: list[LoadError]


def load_directory(root: Path) -> LoadReport:
    """Load every supported file under ``root``, recursively.

    Files are visited in sorted order so that a run over the same folder
    produces the same documents in the same sequence — the first of several
    places this project has to be deterministic to make its numbers reproducible.
    """
    documents: list[Document] = []
    errors: list[LoadError] = []

    if not root.exists():
        return LoadReport(
            documents=[],
            errors=[_error(root, LoadErrorCode.NOT_FOUND, "No such directory.")],
        )

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        result = load_document(path, identity=str(path.relative_to(root)))
        if isinstance(result, Document):
            documents.append(result)
        else:
            errors.append(result)

    return LoadReport(documents=documents, errors=errors)
