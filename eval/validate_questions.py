"""Check that every label in questions.json is still true of the corpus.

The evaluation set is only worth what its labels are worth, and a label is a
claim about a character range in a document: that the text there answers the
question. Claims rot. A corpus rebuild, an edit, a change of chunker - any of
them can move an offset and leave a label pointing at the wrong sentence, and
nothing downstream would notice. Recall@5 would simply be wrong, in a direction
nobody could see.

So the labels are checkable, and this checks them:

* every span round-trips - the text at those offsets is the quote recorded
  beside it, so a label cannot drift without failing here;
* every chunk id listed is exactly the set of reference-configuration chunks
  the span overlaps, so the ids a retrieval run is scored against are real;
* the two halves of a multi-chunk question land in different chunks, which is
  what makes it a multi-chunk question rather than a long sentence;
* every unanswerable question's terms are absent from all forty documents,
  which is the only evidence that "unanswerable" means what it says.

Run it after touching the corpus, the chunker or the set:

    uv run python eval/validate_questions.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

from neuronest.ingest.chunking import Chunk, FixedSizeChunker
from neuronest.ingest.loaders import Document, LoadError, load_document

EVAL = Path(__file__).parent
CORPUS = EVAL / "corpus"
QUESTIONS = EVAL / "questions.json"


def normalise(text: str) -> str:
    return " ".join(text.split())


def load_corpus(chunk_size: int, chunk_overlap: int) -> dict[str, tuple[Document, list[Chunk]]]:
    chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    corpus: dict[str, tuple[Document, list[Chunk]]] = {}
    for path in sorted(CORPUS.glob("*.txt")):
        # Identity is the corpus-relative name, matching what load_directory
        # passes, so the doc ids here are the ones the service would produce.
        document = load_document(path, identity=path.name)
        if isinstance(document, LoadError):
            msg = f"{path.name}: {document.message}"
            raise SystemExit(msg)
        corpus[path.name] = (document, chunker.chunk(document))
    return corpus


def check_answer(
    answer: dict[str, Any], corpus: dict[str, tuple[Document, list[Chunk]]]
) -> list[str]:
    problems: list[str] = []
    source = str(answer["source"])
    if source not in corpus:
        return [f"names {source}, which is not in the corpus"]

    document, chunks = corpus[source]
    start, end = int(answer["char_start"]), int(answer["char_end"])

    if not 0 <= start < end <= len(document.text):
        return [f"span {start}-{end} is outside {source} (length {len(document.text)})"]

    if normalise(document.text[start:end]) != normalise(str(answer["quote"])):
        problems.append(f"the text at {start}-{end} of {source} is not the quote recorded")

    if str(answer["doc_id"]) != document.doc_id:
        problems.append(f"doc_id {answer['doc_id']} does not match {source} ({document.doc_id})")

    overlapping = [c.chunk_id for c in chunks if c.char_start < end and start < c.char_end]
    if list(answer["chunk_ids"]) != overlapping:
        problems.append(
            f"chunk_ids for {source} {start}-{end} are stale: "
            f"recorded {answer['chunk_ids']}, recomputed {overlapping}"
        )
    return problems


def main() -> int:
    data: dict[str, Any] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    reference = data["reference_chunking"]
    corpus = load_corpus(int(reference["chunk_size"]), int(reference["chunk_overlap"]))
    flattened = {name: normalise(doc.text) for name, (doc, _) in corpus.items()}

    failures: list[str] = []
    counts: dict[str, int] = {"answerable": 0, "multi_chunk": 0, "unanswerable": 0}

    for question in data["questions"]:
        qid, kind = question["id"], question["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        answers = question["answers"]

        if kind == "unanswerable":
            if answers:
                failures.append(f"{qid}: unanswerable but carries {len(answers)} label(s)")
            for term in question["absent_terms"]:
                for name, flat in flattened.items():
                    if re.search(re.escape(str(term)), flat, re.IGNORECASE):
                        failures.append(f"{qid}: '{term}' occurs in {name}, so it is answerable")
            continue

        if not answers:
            failures.append(f"{qid}: {kind} but carries no label")
        for answer in answers:
            failures.extend(f"{qid}: {problem}" for problem in check_answer(answer, corpus))

        if kind == "multi_chunk":
            if len(answers) < 2:
                failures.append(f"{qid}: multi_chunk needs two labels, has {len(answers)}")
            else:
                first, second = set(answers[0]["chunk_ids"]), set(answers[1]["chunk_ids"])
                if first & second:
                    failures.append(f"{qid}: both labels fall in the same chunk")

    print(f"corpus     {len(corpus)} documents, {sum(len(c) for _, c in corpus.values())} chunks")
    print(f"questions  {counts} (total {len(data['questions'])})")

    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("every label round-trips, every chunk id is current, every absent term is absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
