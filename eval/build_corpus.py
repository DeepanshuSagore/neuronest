"""Fetch and clean the evaluation corpus.

The corpus is forty IETF RFCs, committed as text rather than downloaded on
demand so that a published number can be reproduced offline years later. This
script is how they got here: running it reproduces the committed files exactly,
which is what makes the provenance checkable rather than asserted.

Why RFCs, out of everything redistributable: they are plain text, so no
extraction step stands between the document and the index and no bad score can
be blamed on a PDF parser; they are freely redistributable in full; they are
precise, so an answer is either in the text or it is not; and they cover one
subject with an edge that can be aimed at deliberately.

That last property is the one the evaluation set is built on. The corpus covers
HTTP/1.1 and HTTP/2 over TLS, DNS, URIs, JSON and the IPv4-era transport
basics, and deliberately excludes QUIC, HTTP/3, WebSockets, IPv6, mail, SSH and
routing. Twenty of the hundred questions ask about the excluded half. They
sound exactly like something this pile would cover, which is the only kind of
unanswerable question that tests anything: a corpus of forty networking RFCs
asked about the boiling point of mercury proves nothing, because every system
declines that one.
"""

import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple


class Rfc(NamedTuple):
    number: int
    title: str
    topic: str


CORPUS: tuple[Rfc, ...] = (
    Rfc(9110, "HTTP Semantics", "HTTP"),
    Rfc(9111, "HTTP Caching", "HTTP"),
    Rfc(9112, "HTTP/1.1", "HTTP"),
    Rfc(9113, "HTTP/2", "HTTP"),
    Rfc(6265, "HTTP State Management Mechanism", "HTTP"),
    Rfc(7239, "Forwarded HTTP Extension", "HTTP"),
    Rfc(6266, "Use of the Content-Disposition Header Field in HTTP", "HTTP"),
    Rfc(5789, "PATCH Method for HTTP", "HTTP"),
    Rfc(7617, "The 'Basic' HTTP Authentication Scheme", "auth"),
    Rfc(7616, "HTTP Digest Access Authentication", "auth"),
    Rfc(6749, "The OAuth 2.0 Authorization Framework", "auth"),
    Rfc(6750, "The OAuth 2.0 Authorization Framework: Bearer Token Usage", "auth"),
    Rfc(8446, "The Transport Layer Security (TLS) Protocol Version 1.3", "TLS"),
    Rfc(5246, "The Transport Layer Security (TLS) Protocol Version 1.2", "TLS"),
    Rfc(6066, "TLS Extensions: Extension Definitions", "TLS"),
    Rfc(7301, "TLS Application-Layer Protocol Negotiation Extension", "TLS"),
    Rfc(6797, "HTTP Strict Transport Security (HSTS)", "TLS"),
    Rfc(5280, "Internet X.509 PKI Certificate and CRL Profile", "TLS"),
    Rfc(6960, "X.509 Internet PKI Online Certificate Status Protocol (OCSP)", "TLS"),
    Rfc(1034, "Domain Names - Concepts and Facilities", "DNS"),
    Rfc(1035, "Domain Names - Implementation and Specification", "DNS"),
    Rfc(6891, "Extension Mechanisms for DNS (EDNS(0))", "DNS"),
    Rfc(7858, "Specification for DNS over Transport Layer Security (TLS)", "DNS"),
    Rfc(8484, "DNS Queries over HTTPS (DoH)", "DNS"),
    Rfc(8499, "DNS Terminology", "DNS"),
    Rfc(3986, "Uniform Resource Identifier (URI): Generic Syntax", "encoding"),
    Rfc(3629, "UTF-8, a transformation format of ISO 10646", "encoding"),
    Rfc(2045, "MIME Part One: Format of Internet Message Bodies", "encoding"),
    Rfc(2046, "MIME Part Two: Media Types", "encoding"),
    Rfc(6838, "Media Type Specifications and Registration Procedures", "encoding"),
    Rfc(4648, "The Base16, Base32, and Base64 Data Encodings", "encoding"),
    Rfc(8259, "The JavaScript Object Notation (JSON) Data Interchange Format", "JSON"),
    Rfc(7515, "JSON Web Signature (JWS)", "JSON"),
    Rfc(7518, "JSON Web Algorithms (JWA)", "JSON"),
    Rfc(7519, "JSON Web Token (JWT)", "JSON"),
    Rfc(9293, "Transmission Control Protocol (TCP)", "transport"),
    Rfc(1122, "Requirements for Internet Hosts - Communication Layers", "transport"),
    Rfc(5681, "TCP Congestion Control", "transport"),
    Rfc(1918, "Address Allocation for Private Internets", "transport"),
    Rfc(2131, "Dynamic Host Configuration Protocol", "transport"),
)

SOURCE_URL = "https://www.rfc-editor.org/rfc/rfc{number}.txt"

# The standing rule for this repository is never to commit a corpus that is not
# ours to redistribute, so the build enforces it instead of the README claiming
# it. Each document must carry one of the grants the RFC series has used:
# "distribution is unlimited" in the older ones, an Internet Society or IETF
# Trust copyright in the later ones.
#
# This is checked against the text with its line breaks flattened, because the
# statement wraps across lines in a fixed-width document and a line-by-line
# search silently misses it - which it did on the first pass here, and nearly
# cost three documents that were properly licensed all along.
LICENCE_GRANT = re.compile(
    r"distribution of this (?:memo|document) is unlimited"
    r"|copyright \(c\) \d{4} ietf trust"
    r"|copyright \(c\) the internet society",
    re.IGNORECASE,
)

# Page furniture from the line-printer era: 35 of the 40 carry it, and each page
# break injects a footer, a form feed and a header into the running text. Left
# in, it interrupts sentences mid-clause - a passage reads "the server MUST",
# then an author name and a page number, then "abort the handshake" - so it
# damages the evidence a citation points at, not just the tidiness of the file.
#
# Both patterns are anchored at column 0, which is what keeps them off content:
# a reference entry citing another RFC is indented and starts with a bracket.
PAGE_FOOTER = re.compile(r"^\S.*\[Page \d+\] *$")
PAGE_HEADER = re.compile(r"^RFC \d+ +.*(?:19|20)\d{2} *$")
BLANK_RUN = re.compile(r"\n{4,}")


def clean(text: str) -> str:
    """Strip page furniture, leaving the prose as one continuous document."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\f", "")
    kept = [
        line.rstrip()
        for line in text.split("\n")
        if not PAGE_FOOTER.match(line) and not PAGE_HEADER.match(line)
    ]
    return BLANK_RUN.sub("\n\n\n", "\n".join(kept)).strip() + "\n"


def fetch(number: int) -> str:
    url = SOURCE_URL.format(number=number)
    with urllib.request.urlopen(url, timeout=60) as response:
        raw: bytes = response.read()
    return raw.decode("utf-8", errors="strict")


def main() -> int:
    destination = Path(__file__).parent / "corpus"
    destination.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for rfc in CORPUS:
        try:
            raw = fetch(rfc.number)
        except (OSError, ValueError) as exc:
            failures.append(f"rfc{rfc.number}: {type(exc).__name__}: {exc}")
            continue

        # A redirect or an error page would also be text. Both checks below
        # would fail on one, which is the point: a corpus is not allowed to
        # quietly contain something that is not the document it claims to be.
        if len(raw) < 2000 or str(rfc.number) not in "\n".join(raw.split("\n")[:120]):
            failures.append(f"rfc{rfc.number}: does not look like the RFC it should be")
            continue

        body = clean(raw)
        if not LICENCE_GRANT.search(" ".join(body.split())):
            failures.append(f"rfc{rfc.number}: no redistribution grant, so it is not ours to ship")
            continue

        (destination / f"rfc{rfc.number}.txt").write_text(body, encoding="utf-8")
        print(f"rfc{rfc.number:<5} {len(raw):>7} -> {len(body):>7} bytes  {rfc.title}")
        time.sleep(0.2)

    if failures:
        print("\nfailed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\n{len(CORPUS)} documents written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
