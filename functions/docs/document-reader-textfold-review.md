# TextFold review and proposed Reader updates

Reviewed 18 October 2026. Proposal only; no Function or OWUI application code changed.

## Baseline and recommendation

Reviewed the user's local `C:/Users/Robert/dev/AI/paperfold` checkout: TextFold **0.10.7**, commit `2cd77e0987b590f9f67154bcdda76c59e6870b73` (16 October). Compared it with Document Reader **0.6.1**, including the keyboard-focus correction merged in PR #7. This is a source review, not an execution or accuracy audit of TextFold. The upstream web release was not substituted for the local checkout.

Keep the standalone OWUI Pipe. Prioritize structured tables, better-scoped questions and reusable reading selections. Reuse OWUI uploads, file permissions, models, chat history, Notes and citations. Preserve our existing 1 = map, 5 = full-text ordering and level-bar gestures; do not add global shortcuts.

The existing Reader already has several upstream ideas: evidence-linked levels, section maps, word-motion transitions, partial results, saved-result reuse, bounded concurrency, separate question reasoning settings, passage questions and reading briefs. They are not new work.

## 1. Structured CSV/TSV and Excel reading — proposed 0.7

Upstream evidence: `textfold/office.py:653–729`, `textfold/semantic.py:315–399`, and `textfold/ladder.py:50–66`.

TextFold now reads CSV/TSV with delimiter detection and `.xlsx` using openpyxl. It renders tables as HTML, uses sheet headings, and can turn simple two-column explanatory tables into lists. Section generation receives limited table context.

Do not copy this adapter unchanged. It drops blank rows, assumes the first retained row is a header, limits output to 1,000 rows and 50 columns, formats floating-point values to ten significant digits, and loads Excel with `data_only=True`. Its resulting table does not retain original cell coordinates. Table context in the generation prompt is capped at 1,500 characters per table and three tables; table cells are not normal evidence units. This is useful for reading, but insufficient for auditable workbook interpretation.

Proposed implementation:

- Add a typed table snapshot alongside prose, with immutable sheet/table IDs, original row and column positions, headers, cell values and provenance. Keep existing v1 saved Readers readable through explicit version dispatch.
- Reuse OWUI's authorized file record and storage adapter to read original bytes when extracted text has lost structure. Never reconstruct confident cell references from flattened text. CSV parsing can use the standard library; Excel support needs an available, declared parser dependency and a clear unavailable-parser message.
- Deliver CSV/TSV first, then rectangular `.xlsx` tables. Preserve CSV strings such as `00123`; distinguish missing cells, empty strings and numeric zero. Store Excel cell types, formulas and cached results separately where available. Do not calculate formulas or execute macros/external links; label missing or potentially stale formula results.
- Treat header detection as a decision that can be inspected or corrected. Explicitly handle duplicate/blank headers, hidden sheets/rows, merged regions and multiple tables. Unsupported layouts should retain a source view with a clear qualification, not invented records.
- Render level 5 as a table with headers and row references; level 4 as cited rows; level 3 as explanations of bounded row groups; level 2 as their main points; level 1 as sheets/tables/topics. Search anchors become row or cell references and survive level changes.
- Cite precise source ranges, such as `Risks!B12:F12`, in the Reader and native OWUI citations. Calculate any totals or comparisons deterministically over an explicitly identified range; the model explains those results.
- Bound bytes, sheets, cells, row groups, generation work and embed size before preparation. Reject or explicitly select a smaller scope when limits are exceeded; never silently present a truncated sheet as complete.

Acceptance: quoted CSV delimiters/newlines, non-UTF-8 inputs with explicit decoding decisions, leading-zero identifiers, duplicate headers, missing values, Excel dates/formulas, multiple sheets, hidden content, unsupported layouts and limit failures. Verify every cited range against the original attachment and exercise Find → Open in reader → levels 1–5. Existing prose inputs and saved snapshots must still work.

## 2. Better questions, using OWUI chat — proposed 0.8

Upstream evidence: `textfold/notes.py:123–243`, `textfold/models.py:143–160`, and `tests/test_ask_model.py`.

TextFold supports selected text, section and whole-document questions. It combines a document map, nearby text and lexical retrieval, gives a short answer before evidence, and lets questions use a separate model. Our Reader currently sends the target passage and nearby source within a fixed budget, and uses the preparation model with a separate reasoning setting.

Proposed implementation:

- Add optional `QUESTION_MODEL_ID`; blank inherits `BASE_MODEL_ID`. Resolve aliases and enforce the current user's model access for the actual answering model. An explicitly configured but inaccessible model should produce an actionable error rather than silently use another provider. Changing only the answering model must not invalidate prepared reading levels.
- Add explicit Passage / Section / Document scope and optionally the selected quote. Record whether a selection came from original text or an AI level. Rebuild all context from the authorized saved snapshot, not client-supplied summary text.
- For wider scopes, use the saved map for navigation and bounded lexical matching to choose original evidence. Only allow citations to source text actually included in the answer context. Report insufficient coverage rather than treating an outline as proof of the entire document.
- Keep the existing draft-and-Send flow, native citations, normal chat history and Stop control. Add bounded follow-up context to support questions such as “Does that exception also apply here?” without creating another chat system inside the iframe.

Acceptance: cross-section qualifications, explicit evidence outside the nearby passages, unsupported answers, changed/revoked sources, answering-model access, bounded history, cancellation and no regeneration of reading levels.

## 3. Keep selected material visible; export it through existing workflows

Upstream evidence: `textfold/web/reader.js:70–85`, `docs/notes-format.md`, `textfold/fold.py`, and `docs/format.md`.

Start with **Keep selected passages visible**, building on Add to brief. A selected point gets a persistent marker at every level and an optional original-source excerpt alongside the current interpretation. Preserve the global level and show that the excerpt is the user's selection, not another AI summary.

If annotations follow, anchor them to source hash + passage/unit IDs + exact quote and surrounding text. Notes on generated text must be distinguished from source highlights. Do not automatically reattach annotations to a changed document just because a similar phrase occurs. TextFold itself documents quote-based recovery after document changes as unfinished.

Use OWUI Notes for durable editable notes, through a user-triggered Pipe command with normal permission checks and duplicate-write protection. Browser-only annotations must clearly say they are local and may not survive isolated iframe storage restrictions. Do not put account tokens in the iframe or depend on parent-DOM access.

Add export of the current level or selected sections as Markdown, plus a versioned JSON snapshot with source hash, model, timestamps, level provenance and evidence IDs. TextFold's `textfold/doc` v0 is a useful interchange reference, but not a replacement for our lossless source partition or table schema: its prose export reconstructs text and has no cell-range model. Offer a deliberate compatibility adapter later, without claiming a lossless round trip. Full offline HTML export is optional and subject to size and download support; chat-dependent actions must be disabled in an offline copy.

## 4. Performance changes only after a larger-document benchmark

Upstream evidence: `textfold/web/reader.js:442–493` and `1243` onward.

TextFold redraws visible/nearby material first, drains offscreen work in roughly 8 ms slices, and warms nearby zoom states while idle. Our Reader limits animated words but still rebuilds the main content when levels change. Table support makes that distinction more important.

Benchmark 50, 200 and 500 passages plus representative tables at desktop and mobile widths. Measure interaction latency, long tasks, memory and anchor drift, including rapid reversals and open source views. If whole-document redraw is the bottleneck, reuse passage nodes and schedule offscreen updates in bounded slices; retain accurate heights and the reading anchor. Add geometry precomputation only if measurement justifies the cache and its invalidation on resize, fonts, expansion, selection and source changes. The earlier 24-passage benchmark was already quick and does not justify a wholesale animation rewrite.

## Defer or reuse OWUI instead

- **PDF comparison:** TextFold's PDF viewer and extraction tiers are valuable references. Its lightweight text extractor explicitly lacks paragraph bounding boxes; precise paragraph-to-page highlighting requires richer ingestion metadata. First expose source/provenance and an authorized original-file link. Reuse OWUI's original-file preview where available; do not promise synchronized page highlighting from our current flattened text. Evaluate a better OWUI extraction configuration separately from the Pipe release.
- **Library/folders/search and provider setup:** keep OWUI's existing facilities. Do not embed TextFold's server, account bridge, local credentials or agent processes in the Function.
- **Folding every assistant reply:** TextFold's Claude/Pi extension relies on replies being written with the point first and then folds sentence prefixes without another model. It is not a reliable generic summarizer for arbitrary documents. Folding other OWUI models' messages would be separate optional Action/Filter work, not an automatic side effect of selecting this Pipe.
- **Translation, Zotero and new file families:** plausible later additions, but lower priority than reliable tables and questions. Translation needs separate provenance and original-language evidence. Keep current OWUI ingestion rather than bundling the multi-gigabyte extraction engine or PDF.js into every saved chat embed.
- **Code reuse:** TextFold is Apache-2.0 with a NOTICE file. If implementation copies source, retain applicable notices and attribution; adopting ideas does not require importing its entire package.

## Suggested order

1. CSV/TSV table schema, rendering, search and row evidence; then bounded `.xlsx` support.
2. Separate question model, section/document scopes and normal-chat follow-ups.
3. Keep-selected reading, current-level export and an explicit Save to OWUI Notes flow.
4. Performance work if benchmarks show a problem; original-PDF integration once ingestion can support it accurately.

The live Function was not changed during this review. The keyboard-focus correction merged into PR #7 also remains a separate installation check; the last recorded live installation predates that correction.
