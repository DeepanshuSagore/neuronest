# The evaluation set

One hundred questions over the forty documents described in [`CORPUS.md`](CORPUS.md), each one
labelled by hand with the text that answers it. They live in
[`questions.json`](questions.json) and they are what every number this project publishes is measured
against.

This page is the labelling protocol: what a label means, how each of the hundred was made, how the
set is checked, and how to extend it without weakening it.

## The rule that decides everything else: label before you tune

The set was written and committed before any retrieval parameter was measured or changed. Nothing in
it was chosen because the current configuration happened to answer it.

That order is not a nicety. A set built after tuning is a set of questions the system already gets
right — you write them without noticing, because you know what it can do — and every score after
that is decoration. The corpus commit and this commit both land before phase 12 computes a single
metric, and the git history is the evidence.

The corollary: **when a number later disappoints, fix the system, not the set.** A question may be
corrected if it is *wrong* — mislabelled, ambiguous, or asking about text that does not exist — and
that correction is a commit that says so. It may not be corrected for being hard.

## Shape

| Kind | Count | What it tests |
|---|---|---|
| `answerable` | 70 | Retrieval finds the passage; generation answers from it |
| `multi_chunk` | 10 | Both halves of an answer are retrieved, not just the nearer one |
| `unanswerable` | 20 | The service declines instead of answering from adjacent text |

All forty documents carry at least one label, so no document is in the corpus purely as noise.

The twenty unanswerable questions are the reason the corpus has a deliberate edge. Each one asks
about a subject that plainly belongs to the same world and is simply not in these forty documents —
QUIC, HTTP/3, WebSockets, IPv6 autoconfiguration, routing, mail, SSH. A question about the boiling
point of mercury would test nothing, because every system on earth declines that one.

## What a label is

A label is a claim about a **character range in a document**, plus the verbatim quote found there:

```json
{
  "id": "q001",
  "question": "What does a 408 status code tell the client?",
  "kind": "answerable",
  "topic": "HTTP",
  "answers": [
    {
      "source": "rfc9110.txt",
      "doc_id": "3f9054029f235d46",
      "char_start": 359180,
      "char_end": 359362,
      "quote": "408 Request Timeout The 408 (Request Timeout) status code indicates that the server did not receive a complete request message within the time that it was prepared to wait.",
      "chunk_ids": ["3f9054029f235d46-0473-103643e9", "3f9054029f235d46-0474-3c53a38c"]
    }
  ]
}
```

An `unanswerable` question carries `"answers": []` and an `absent_terms` list instead — the terms
whose absence from all forty documents is what makes the question unanswerable:

```json
{
  "id": "q081",
  "question": "How does QUIC handle connection migration when a client's address changes?",
  "kind": "unanswerable",
  "topic": "QUIC",
  "answers": [],
  "absent_terms": ["connection migration", "preferred_address"]
}
```

### Why the span is the label and the chunk id is derived

Chunk ids fold in the chunker's fingerprint — `{doc_id}-{index}-{digest}`, where the digest covers
the strategy and its parameters. That is deliberate and load-bearing elsewhere, but it means a chunk
id is only valid for one chunking configuration. Phase 14 sweeps three chunk sizes against two
overlaps; if the labels were chunk ids alone, the first sweep would invalidate all one hundred of
them and the experiment would have nothing to score against.

So ground truth is the span. `chunk_ids` records which chunks that span falls in **under the
reference configuration**, stated at the top of the file:

```json
"reference_chunking": {
  "strategy": "fixed", "chunk_size": 1000, "chunk_overlap": 200, "fingerprint": "fixed-1000-200"
}
```

which produces 5,810 chunks over the forty documents. Any other configuration recomputes its own
chunk ids from the same spans. A retrieval run is scored by asking whether the chunks it returned
overlap the labelled span — the ids are a convenience for reading and for the common case, not the
source of truth.

## How the hundred were made

### Answerable (70)

1. **Write the question first, from the subject — not from a sentence.** Reading for quotable
   sentences and reverse-engineering questions out of them produces questions shaped like the text,
   which flatters any retriever.
2. **Find the supporting text in the corpus.** Search the flattened document (line breaks collapsed
   to single spaces — these documents are hard-wrapped at ~72 characters, so `best-effort` appears in
   the file as `best- effort` and a naive search misses it).
3. **Require exactly one match.** An anchor phrase that matches in several places means the question
   has several correct answers and Recall@k would mark a correct retrieval wrong. Narrow the phrase
   or drop the question.
4. **Require that the matched text actually answers the question**, not merely that it mentions the
   subject. This is where questions written from memory die: several drafts named requirements these
   documents do not state in the form remembered.
5. **Record the span, the quote and the derived chunk ids**, then run the validator.

### Multi-chunk (10)

Same procedure, twice, with one extra rule: **the two spans must fall in different chunks under the
reference configuration**, and the validator enforces it. Otherwise it is one long sentence wearing a
second label, and it tests nothing that the answerable set does not already test.

The two halves may come from the same document or from two — both appear in the set. A question
spanning two documents (q076: DNS-over-TLS's port and DNS-over-HTTPS's media type) is the harder
case, because nothing in the index links them.

### Unanswerable (20)

1. **Pick a subject adjacent to the corpus, not distant from it.** The test is whether the service
   notices that forty documents about HTTP over TCP do not cover the protocol that replaced TCP.
2. **Choose the terms an answer would have to contain.** For "how does QUIC handle connection
   migration", an answer would have to say `connection migration` or `preferred_address`.
3. **Grep the whole corpus for every one of those terms, in flattened text, case-insensitively.**
   If any appears anywhere, the question is not unanswerable — fix the terms or drop it.
4. **Record the terms in `absent_terms`.** They are not decoration: the validator re-checks them on
   every run, so a future corpus change that introduces the subject fails the build instead of
   silently turning a refusal question into an answerable one.

Note that a subject being *mentioned* is fine, and is in fact what makes the best questions. RFC 9110
names QUIC and HTTP/3 in passing without explaining either. What must be absent is the **answer**,
not the word.

## Checking the set

```bash
uv run python eval/validate_questions.py
```

It loads the corpus through the same loader and chunker the service uses, and asserts, for all 100:

- every span is inside its document, and the text there normalises to the recorded quote;
- every `doc_id` matches the id the loader derives for that file;
- every `chunk_ids` list is exactly the set of reference-configuration chunks the span overlaps;
- every `multi_chunk` question has two labels landing in different chunks;
- every `unanswerable` question carries no labels and no `absent_terms` term occurs in any document.

Passing output:

```
corpus     40 documents, 5810 chunks
questions  {'answerable': 70, 'multi_chunk': 10, 'unanswerable': 20} (total 100)
every label round-trips, every chunk id is current, every absent term is absent
```

It bites. Shifting one offset by forty characters produces:

```
2 problem(s):
  q001: the text at 359220-359362 of rfc9110.txt is not the quote recorded
  q001: chunk_ids for rfc9110.txt 359220-359362 are stale: recorded [...], recomputed [...]
```

Run it after touching the corpus, the loader, the chunker or the set. A label is a claim, and an
unchecked claim about a character offset rots quietly: nothing downstream would notice, and the only
symptom would be a recall number that is wrong in a direction nobody can see.

## What verification threw away

Four drafted questions were removed because the corpus disagreed with them, and they are worth
recording because each was plausible:

- *"What is IPv6's minimum MTU?"*, drafted as unanswerable. RFC 9293 leaks it in passing as
  `1220 (1280 - 60) for IPv6`.
- *"How does SCTP avoid head-of-line blocking?"*, drafted as unanswerable. RFC 9113 discusses
  head-of-line blocking directly.
- *"How does SMTP negotiate STARTTLS?"*, drafted as unanswerable. RFC 7858 describes DNS-over-TLS as
  "essentially, STARTTLS for DNS".
- *"Which records does DNSSEC add?"*, drafted as unanswerable. RFC 8499 defines DNSKEY, RRSIG and DS
  outright.

Without step 3 of the unanswerable procedure, four of the twenty would have been labelled as
refusals the service could never correctly make.

One answerable question was also corrected before commit: q051 ("which characters count as
insignificant whitespace in JSON") originally pointed at the span listing JSON's six structural
characters, which says whitespace *is allowed* around them without saying what whitespace *is*. It
now points at the `ws` ABNF rule, which does.

## Known weaknesses, stated rather than discovered later

**The unanswerable twenty have a difficulty gradient.** Six of them (BGP, OSPF, CoAP, MQTT, gRPC,
WebRTC) name a subject that appears nowhere in the corpus at all — those refusals are easy, and a
system could pass them on vocabulary alone. The hard ones are q081 and q082 (QUIC and HTTP/3 are both
named in RFC 9110), q083 (RFC 9110 contains a WebSocket `Upgrade` example) and q099 (TLS is covered
in depth across four documents). If refusal accuracy comes out near 100%, that gradient is part of
the reason, and phase 13 should report the breakdown rather than the aggregate alone.

**The multi-chunk ten reuse spans from the answerable seventy.** q078 asks about the 410 and 411
status codes, whose spans are already labelled separately as q002 and q003. The question is new even
though the evidence is not — what is being tested is whether both passages are retrieved together,
which nothing in the answerable set tests. It does mean the two subsets are not statistically
independent, and a reader comparing their scores should know that.

**Four quotes read as fragments** because the supporting text in the source is a table row or a
numbered step rather than a sentence: q022 (Basic auth's construction step), q033 (a DNS message size
table row), q037 (the TTL entry in a class list) and q051 (an ABNF block). The spans are correct and
the answers are in them; they are simply less readable than the other sixty-six, and an LLM judge in
phase 13 may score them harder for reasons that are the document's formatting rather than the
service's retrieval.

**These are English-language technical standards, and nothing else.** No prose, no narrative, no
multilingual text, no scanned documents. A retrieval number measured here transfers to documents that
look like these and should not be quoted as though it transfers to anything else.

## Extending the set

The set is meant to grow. To add questions:

1. Follow the procedure above for the kind you are adding.
2. Append to `questions` in `questions.json`, keeping ids sequential.
3. Update `counts` to match.
4. Run `uv run python eval/validate_questions.py` and do not commit until it exits 0.
5. Re-run whatever phase-12 metrics already exist, and note in the commit that the denominator moved
   — a Recall@5 over 100 questions and one over 130 are not comparable numbers.

If you change the corpus, re-run `uv run python eval/build_corpus.py` first, then the validator: any
label whose document shifted will fail loudly rather than silently pointing at the wrong sentence.
