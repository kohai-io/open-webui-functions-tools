# Document Reader offline fixtures

`corpus.json` contains synthetic OWUI **extracted text**, not original binary
documents. It tests the Function's actual boundary: exact strings supplied by
OWUI, including CRLF, repeated text, combining characters, emoji, table text and
untrusted markup. No external document or paid model is used.

The tests construct model responses using known fixture evidence IDs. This
verifies validation and source traceability, not the factual quality of a real
model. Real DOCX/PDF/Markdown extraction and human review remain pilot checks.

## Manual upload pilot

`pilot-brief.md`, `pilot-brief.docx` and `pilot-brief.pdf` contain the same short
synthetic business brief. The PDF has a text layer. Their headings and whitespace
may differ after extraction; their wording and qualifications should agree.

1. Configure and enable Document Reader as described in the Function guide.
2. Start a fresh saved chat for each format, choose Document Reader and attach
   exactly one file. Wait for OWUI processing, then send “Prepare this document”.
3. Inspect all five levels, expand a passage and check its source. Confirm that
   the text below survives and generated wording preserves its qualifications.
4. Reopen the chat and exercise zoom and source inspection. These local reader
   interactions should make no new model calls.
5. Record format, extraction quality, model, preparation time, completed/partial
   status and any missing or unsupported claims. A new generation uses the
   configured model and may incur its normal cost.

Expected qualifications:

- The four-week pilot covers two teams and is **not approved**.
- Only synthetic documents are in scope; private business/personal data is out.
- GBP 12,000 is a **proposed budget**, conditional on sponsor approval.
- 2 November 2026 is a **provisional date**, dependent on the access review.
- Reviewing 20 documents is a **target**, not a promised outcome.
- Material source errors may end the pilot early.
- The coordinator must obtain approval before describing the pilot as approved.

All names, figures and dates are fictional test content. Local `docx2txt` and
`pypdf` extraction are checked against the Markdown wording; this does not verify
the configured OWUI extraction engine or a real model's interpretation.

`build_pilot_fixtures.py` rebuilds the three files using existing `python-docx`
and `reportlab` packages. It requires no network or provider credentials.
