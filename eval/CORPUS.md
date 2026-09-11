# The evaluation corpus

Forty IETF RFCs, committed as text. They are the documents the hundred questions in
[`questions.json`](questions.json) are labelled against, and the documents every number this
project publishes is measured over.

Rebuild them from source with:

```bash
uv run python eval/build_corpus.py
```

That fetches each one from `https://www.rfc-editor.org/rfc/rfcNNNN.txt` and reproduces the files
there exactly. Nothing about this corpus is hand-edited.

`eval/corpus/` holds the forty `.txt` files and nothing else. This description lives outside it
deliberately: the loader accepts `.md` as well, so a file describing the corpus boundary sitting
inside the corpus would be ingested as a forty-first document — and the service could then "answer"
a question about QUIC by citing this page saying QUIC was excluded, which would quietly break the
twenty questions that matter most.

## Why RFCs

Four properties, and the project needs all four.

**They are plain text.** No extraction step stands between the document and the index, so a bad
retrieval score cannot be blamed on a PDF parser. When phase 14 compares chunking strategies, the
difference it measures is the chunker's, not the reader's.

**They are freely redistributable in full**, so the corpus ships with the repository and a published
number stays reproducible offline. See the licence section below.

**They are precise.** An RFC states requirements exactly, which makes labelling honest: the answer to
"what is the minimum MTU a host must support" is either in the text or it is not, and two people
labelling it would agree.

**They have an edge that can be aimed at.** This is the property the evaluation set is actually built
on, and the reason a pile of novels or a single project's manual would not do.

## The boundary, which is the point

The corpus covers **HTTP/1.1 and HTTP/2 over TLS, DNS, URIs, JSON, and the IPv4-era transport
basics**. It deliberately excludes **QUIC, HTTP/3, WebSockets, IPv6, mail, SSH, routing, NTP and
SCTP** — subjects that plainly belong to the same world and are simply not here.

Twenty of the hundred questions ask about the excluded half. That is the only kind of unanswerable
question that tests anything. A corpus of forty networking RFCs asked about the boiling point of
mercury proves nothing, because every system on earth declines that one. Asked *"how does QUIC handle
connection migration?"*, a system has to actually notice that forty documents about HTTP over TCP do
not cover the protocol that replaced TCP — and most will confidently answer from the adjacent text
instead.

## The documents

**HTTP** (8)

| Document | Title | Size |
|---|---|---|
| [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.txt) | HTTP Semantics | 491 KB |
| [RFC 9111](https://www.rfc-editor.org/rfc/rfc9111.txt) | HTTP Caching | 82 KB |
| [RFC 9112](https://www.rfc-editor.org/rfc/rfc9112.txt) | HTTP/1.1 | 107 KB |
| [RFC 9113](https://www.rfc-editor.org/rfc/rfc9113.txt) | HTTP/2 | 187 KB |
| [RFC 6265](https://www.rfc-editor.org/rfc/rfc6265.txt) | HTTP State Management Mechanism | 72 KB |
| [RFC 7239](https://www.rfc-editor.org/rfc/rfc7239.txt) | Forwarded HTTP Extension | 31 KB |
| [RFC 6266](https://www.rfc-editor.org/rfc/rfc6266.txt) | Content-Disposition in HTTP | 23 KB |
| [RFC 5789](https://www.rfc-editor.org/rfc/rfc5789.txt) | PATCH Method for HTTP | 19 KB |

**Authentication** (4)

| Document | Title | Size |
|---|---|---|
| [RFC 7617](https://www.rfc-editor.org/rfc/rfc7617.txt) | The 'Basic' HTTP Authentication Scheme | 27 KB |
| [RFC 7616](https://www.rfc-editor.org/rfc/rfc7616.txt) | HTTP Digest Access Authentication | 65 KB |
| [RFC 6749](https://www.rfc-editor.org/rfc/rfc6749.txt) | The OAuth 2.0 Authorization Framework | 148 KB |
| [RFC 6750](https://www.rfc-editor.org/rfc/rfc6750.txt) | OAuth 2.0 Bearer Token Usage | 35 KB |

**TLS and certificates** (7)

| Document | Title | Size |
|---|---|---|
| [RFC 8446](https://www.rfc-editor.org/rfc/rfc8446.txt) | TLS 1.3 | 305 KB |
| [RFC 5246](https://www.rfc-editor.org/rfc/rfc5246.txt) | TLS 1.2 | 201 KB |
| [RFC 6066](https://www.rfc-editor.org/rfc/rfc6066.txt) | TLS Extensions: Extension Definitions | 50 KB |
| [RFC 7301](https://www.rfc-editor.org/rfc/rfc7301.txt) | TLS ALPN Extension | 15 KB |
| [RFC 6797](https://www.rfc-editor.org/rfc/rfc6797.txt) | HTTP Strict Transport Security (HSTS) | 94 KB |
| [RFC 5280](https://www.rfc-editor.org/rfc/rfc5280.txt) | X.509 Certificate and CRL Profile | 321 KB |
| [RFC 6960](https://www.rfc-editor.org/rfc/rfc6960.txt) | Online Certificate Status Protocol (OCSP) | 73 KB |

**DNS** (6)

| Document | Title | Size |
|---|---|---|
| [RFC 1034](https://www.rfc-editor.org/rfc/rfc1034.txt) | Domain Names — Concepts and Facilities | 114 KB |
| [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.txt) | Domain Names — Implementation and Specification | 111 KB |
| [RFC 6891](https://www.rfc-editor.org/rfc/rfc6891.txt) | Extension Mechanisms for DNS (EDNS(0)) | 29 KB |
| [RFC 7858](https://www.rfc-editor.org/rfc/rfc7858.txt) | DNS over TLS | 38 KB |
| [RFC 8484](https://www.rfc-editor.org/rfc/rfc8484.txt) | DNS Queries over HTTPS (DoH) | 42 KB |
| [RFC 8499](https://www.rfc-editor.org/rfc/rfc8499.txt) | DNS Terminology | 108 KB |

**URIs, encodings and media types** (6)

| Document | Title | Size |
|---|---|---|
| [RFC 3986](https://www.rfc-editor.org/rfc/rfc3986.txt) | URI Generic Syntax | 129 KB |
| [RFC 3629](https://www.rfc-editor.org/rfc/rfc3629.txt) | UTF-8 | 31 KB |
| [RFC 2045](https://www.rfc-editor.org/rfc/rfc2045.txt) | MIME Part One: Message Bodies | 66 KB |
| [RFC 2046](https://www.rfc-editor.org/rfc/rfc2046.txt) | MIME Part Two: Media Types | 96 KB |
| [RFC 6838](https://www.rfc-editor.org/rfc/rfc6838.txt) | Media Type Specifications and Registration | 66 KB |
| [RFC 4648](https://www.rfc-editor.org/rfc/rfc4648.txt) | Base16, Base32 and Base64 Encodings | 31 KB |

**JSON and JOSE** (4)

| Document | Title | Size |
|---|---|---|
| [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259.txt) | The JSON Data Interchange Format | 25 KB |
| [RFC 7515](https://www.rfc-editor.org/rfc/rfc7515.txt) | JSON Web Signature (JWS) | 119 KB |
| [RFC 7518](https://www.rfc-editor.org/rfc/rfc7518.txt) | JSON Web Algorithms (JWA) | 141 KB |
| [RFC 7519](https://www.rfc-editor.org/rfc/rfc7519.txt) | JSON Web Token (JWT) | 57 KB |

**Transport and addressing** (5)

| Document | Title | Size |
|---|---|---|
| [RFC 9293](https://www.rfc-editor.org/rfc/rfc9293.txt) | Transmission Control Protocol (TCP) | 257 KB |
| [RFC 1122](https://www.rfc-editor.org/rfc/rfc1122.txt) | Requirements for Internet Hosts — Communication Layers | 273 KB |
| [RFC 5681](https://www.rfc-editor.org/rfc/rfc5681.txt) | TCP Congestion Control | 40 KB |
| [RFC 1918](https://www.rfc-editor.org/rfc/rfc1918.txt) | Address Allocation for Private Internets | 20 KB |
| [RFC 2131](https://www.rfc-editor.org/rfc/rfc2131.txt) | Dynamic Host Configuration Protocol | 104 KB |

**40 documents, 4.2 MB, 4,364,110 bytes.**

## What was changed, and what was not

Thirty-five of the forty are in the original fixed-width format, where every page carries a footer,
a form feed and a header. The build removes those three things and nothing else.

This is not tidiness. Page furniture interrupts sentences mid-clause: left in, a retrieved passage
reads *"...the server MUST"*, then an author's name and a page number, then *"abort the handshake"*.
That damages the evidence a citation points at, and phase 13 would be scoring faithfulness against
mangled text.

Measured over the whole corpus: **2,746 lines removed, 17,926 words, every one of them furniture.**
The two patterns are anchored at column 0, and collecting every distinct line they matched produced
1,291 lines of which **zero** were content — a reference entry citing another RFC is indented and
starts with a bracket, so it cannot match. No body text, no section, and no copyright notice is
touched.

## Licence

Every document here carries its own grant, in one of the three forms the RFC series has used:

> Distribution of this memo is unlimited.

> Copyright (C) The Internet Society (2005).

> Copyright (c) 2022 IETF Trust and the persons identified as the document authors. All rights
> reserved. This document is subject to BCP 78 and the IETF Trust's Legal Provisions Relating to
> IETF Documents (https://trustee.ietf.org/license-info).

`build_corpus.py` enforces this rather than this file claiming it: a document whose text contains
none of those three grants fails the build and is not written. The check runs against the text with
its line breaks flattened, because the statement wraps across lines in a fixed-width document.

That gate changed the corpus. RFC 791 (IP), RFC 768 (UDP) and RFC 826 (ARP) were in the first draft
and were removed: written in 1980–82, they predate both conventions and carry no distribution
statement at all. Rather than ship three documents on an assumption, they were replaced by
[RFC 1122](https://www.rfc-editor.org/rfc/rfc1122.txt), which specifies the same IP, UDP and TCP
behaviour in more checkable detail, plus RFC 5681 and RFC 1918 — all three of which say so in
writing.

The documents are the IETF's. Only the selection, the formatting change described above and the
questions labelled against them are this repository's, and those are MIT along with the rest of it.
