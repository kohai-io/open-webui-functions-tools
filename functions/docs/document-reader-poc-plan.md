# Document Reader: OWUI Function proof of concept

Status: Function 0.2.1 is installed and verified on the live host, with 98 focused checks and four frontend bridge checks passing. It adds overview-first navigation, source-linked key concepts, section hierarchy, automatic browser-local position restoration where supported, and Focus reading. The Guidelines PDF completed 15/15 batches in 15 calls; its exact 28,014-character source was retained. Live checks covered concept navigation, sampled source qualifications, automatic restoration, fullscreen and wide/narrow resizing. The optional isolated-frame frontend bridge remains undeployed. Multi-format and ordinary-user acceptance remain outstanding. See the [installation and pilot guide](document-reader.md). The design below remains the scope reference for OWUI 0.11.3.

## Outcome

Build one installable Python **Pipe Function**, shown in OWUI's model selector as **Document Reader**. A user attaches one DOCX, text-based PDF or Markdown document and sends a message. OWUI processes the file; the Pipe prepares linked reading levels and saves a self-contained interactive reader in its assistant response.

The reader supports **Section map → Takeaways → Explanation → Extracts → Full text**, with in-place expansion and source inspection. Switching level keeps the reader at the same passage. Opening an existing result requires no model calls. This adapts the interaction described by [Paperfold](https://github.com/chenxiachan/paperfold); the PoC does not require importing its application stack.

OWUI supplies uploads, document extraction, file access, identity, configured models and saved chats. The Function adds deterministic source passages, validated model-generated representations and a fixed HTML/CSS/JavaScript reader. There are no new application routes, Studio changes, parser packages, provider keys or database tables. An optional frontend host bridge supports local position storage and viewport sizing in isolated embeds.

## User workflow

1. Start an ordinary saved chat and select **Document Reader**. Attach one supported document using OWUI's existing upload control. An existing individual file is also eligible when the UI can attach it.
2. Send a message such as “Prepare this document”. Any ordinary message with one eligible attachment starts preparation; the PoC is not a general document Q&A model.
3. See OWUI status updates for loading extracted text and preparing completed/total batches. A bounded processing wait handles a recently uploaded file.
4. Open the completed reader at **Section map**, with substantive topics, source-linked key concepts and source-derived subsection hierarchy. The response also reports filename, complete/partial status and source passage count.
5. Change level, expand a passage or inspect its supporting source. All these interactions happen locally inside the reader.
6. Reopen the saved chat to recover the generated reader. Use **Save place** and **Restore place** for a manual bookmark when needed. Rerunning the Function performs a new generation; it is not a continuation of the old result.

Use one document per chat during the PoC. Still implement explicit attachment selection so inherited chat/project files cannot silently replace the intended document. Temporary chats and API-only calls without the required embed/saved-message context receive an actionable message before generation.

## Scope and visible limits

- English business prose first, with Unicode preserved. DOCX, PDF and Markdown support uses the deployment's configured OWUI extractor. No new OCR configuration or original-file parsing is added.
- **Full text means the frozen text extracted by OWUI**, in its supplied order. It does not reproduce Word/PDF layout. Keep the existing attachment as the route to the original file; do not invent page numbers or original-page highlights.
- Retain headings, paragraphs, lists and table text where extraction preserves them. Use deterministic “Part 1”, “Part 2” groups where structure is absent. Ambiguous table regions remain source-only; do not infer values from lost columns or image-only charts.
- A saved result is a document snapshot. Editing/reprocessing the source or changing the generation model does not update existing results.
- Reading position and expansion state restore automatically in browser-local storage when supported by the host. Otherwise use a manual bookmark; a new snapshot starts at Section map. Live generation callbacks and durable job recovery remain deferred.
- A partial result remains useful: successful passages retain generated levels, and unavailable levels show a labelled source-text fallback. Do not describe an incomplete map or takeaway view as complete.

Display this notice in **About this reader** and document it in setup guidance: **“This reader is saved in the chat and contains the full extracted document. Sharing or exporting the chat may disclose that text.”** Revoking access to the original file cannot revoke a copy already embedded in a saved/shared chat. Document this consequence in setup guidance as well.

## Package and OWUI configuration

Follow the existing standalone Function layout:

| Planned file                                                      | Purpose                                                                                                 |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `functions_tools/functions/document_reader.py`                    | Importable Pipe, Valves, source loading, segmentation, generation, validation and fixed reader template |
| `functions_tools/functions/docs/document-reader.md`               | Installation, model configuration, use, limits and troubleshooting                                      |
| `functions_tools/functions/tests/test_document_reader.py`         | Offline contract, source, model-delegation and failure tests                                            |
| `functions_tools/functions/tests/test_document_reader_browser.py` | Browser tests against the rendered template and real sandbox settings                                   |
| `functions_tools/functions/tests/test_document_reader_resize.py`  | Live iframe/browser resizing, preserved reading state, responsive map and source inspector              |
| `functions_tools/functions/tests/fixtures/document_reader/`       | Synthetic business-document fixtures and approved expected results                                      |

The deployable artifact is one Python file, using standard-library code and packages already available in OWUI. Bundle the authored UI template in that file; no build step, CDN or external JavaScript framework is required. The model returns data only, never executable HTML or JavaScript.

An administrator imports/enables the Function and configures a permitted, ordinary server-backed text model in `BASE_MODEL_ID`. Configure the Reader model with **file upload enabled** and **file context disabled** (`info.meta.capabilities.file_context = false`) so ordinary RAG processing does not run before the Pipe. Disable unrelated tools, web search and code execution for this dedicated model. Verify these settings in the actual model editor and retain attachment metadata. A companion Filter is not part of the design.

## Source loading and canonical text

Implement the async Pipe using injected `__request__`, `__user__`, `__files__`, `__metadata__`, `__task__` and `__event_emitter__` plus the normal body. Return immediately for automatic title/tag/follow-up tasks, before file reads or generation.

Select a concrete file in this order:

1. Eligible files on `__metadata__.user_message.files`, when present.
2. For regeneration, files on the corresponding stored user message, after authorizing access to that chat/message.
3. Use `__files__` to resolve attachment details only for the ID selected above. A sole inherited chat/project file is not sufficient evidence of the intended source. If current-message provenance cannot be established, ask the user to attach the intended document explicitly.

Accept the known `item.id` and `item.file.id` attachment shapes. Reject ambiguous multiple files, Knowledge collections/folders, images, URLs and unsupported types. Never choose by upload timestamp or trust client-supplied text, ownership or storage paths.

Resolve the actual OWUI user and fresh file record through the async user/file models. Enforce the same ownership/admin/shared-read policy as the Files API, including `has_access_to_file`. Read the stored `file.data.content`, which backs `/api/v1/files/{id}/data/content`, rather than top-k vector retrieval. Close short-lived database reads before making model calls.

For a pending upload, reread processing status/content within the configured wait. If there is no usable text after failure or timeout, report the condition and let the user rerun after OWUI processing completes. Usable stored text can still be read when vector indexing failed; report that condition without rerunning extraction. Legacy text-bearing records without a status must not wait indefinitely. Do not modify or automatically reprocess the source.

Freeze the exact extracted string before segmentation. Hash its UTF-8 bytes together with the segmentation version. Derive deterministic sections, passages and sentence/list-item evidence units while preserving order, whitespace and intervening text. Repeated identical sentences still receive different location-based IDs. Long prose blocks may be split at sentence boundaries; fragments and table text must not be silently discarded. Reject input exceeding structural limits rather than truncate it. If a single indivisible source unit exceeds the batch limit, reject it before generation with a clear size message; arbitrary mid-unit splitting is outside this first version.

The server owns all source slicing. If offsets are stored, define them as Python Unicode code-point offsets. Serialize the resulting exact passage/evidence strings and IDs to the reader; JavaScript must not apply those offsets directly to UTF-16 string slicing. Highlight evidence using the corresponding DOM spans. Test emoji, combining characters, non-Latin text and CRLF explicitly.

## Representation and generation contract

Store a versioned snapshot containing source filename/ID/hash, segmentation/schema/prompt versions, generator model ID, preparation time/status, ordered sections/passages/evidence units, validated generated items and per-batch failures. The full canonical text must be recoverable exactly from its ordered source segments. Avoid repeatedly embedding full source in each representation.

| Level       | Content                                                      | Evidence rule                                                                                         |
| ----------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Full text   | Complete canonical passages                                  | Original extracted strings; no LLM rewriting                                                          |
| Extracts    | One to three selected source units per passage               | Model chooses IDs; Python copies exact text in source order                                           |
| Explanation | One or two short explanatory sentences per passage           | Each item cites one or more known source units                                                        |
| Takeaways   | One concise, qualified point per passage, grouped by section | Each point cites known source units; no unsupported additions                                         |
| Section map | Heading/group cards and a short overview                     | Retained headings distinguished from fallback labels; any AI overview cites its contributing passages |

Make one structured generation call per bounded batch, requesting extracts, explanations, takeaways and a batch overview together. Keep section boundaries when they fit. For large sections, display separately labelled subsection/batch overviews rather than inventing a whole-section summary from the first batch. No additional global summarization call is needed.

Prompts identify document content as untrusted data, preserve qualifications and separate source wording from interpretation. Source-only passages are excluded from model inputs and expected generated coverage; render their original text and a clear source-only label in every text level, and mark their presence on the map. Validate returned JSON with strict schemas, bounded counts/lengths, known unique passage IDs, mandatory evidence and complete expected coverage of eligible passages. Unknown references, missing eligible passages or malformed outputs fail that batch. A valid evidence ID establishes traceability, not factual entailment: assess whether AI claims are actually supported during the fixture and human evaluation.

Allow at most one repair call for a completed but invalid model response, within the overall call/time budget. Do not automatically retry an ambiguous transport failure that may already have incurred generation. Run batches sequentially in the first version. Zooming or inspecting evidence never calls a model.

### Isolate delegated model requests

Use `open_webui.utils.chat.generate_chat_completion`, with the real requesting user and normal model access checks. Do not use the full chat-route alias in `main.py`, set access-bypass flags, impersonate an administrator or add provider credentials.

The current utility merges `request.state.metadata` into a supplied payload and mutates request-state flags. Therefore create a **fresh Starlette Request for every delegated call**, retaining required app/auth context but replacing scope state with a new dictionary. Do not mutate or pass through the Reader's outer request state. Supply a fresh body containing only the chosen model, bounded system/user messages, supported output parameters and `stream=false`. Exclude outer chat/message/session IDs, previous conversation, files, tools, skills and feature selections.

Reject the Reader itself, all Pipe-backed models, arena models and browser-direct connections as generation targets. Confirm the selected model is available to the requesting user. Check provider-specific output limits and optional JSON-schema support during the first delivery slice; validate locally regardless. This internal utility does not rerun every chat middleware/filter, so verify any deployment-required processing policy before the pilot.

## Reader interface and place preservation

Use a toolbar with the five named level buttons, a section navigator, one internally scrolling reading column and a source inspector. Mark AI-generated text explicitly; typography/colour alone are insufficient. Every AI item offers **Inspect source**; each passage offers **Expand here**. Noncontiguous source extracts remain visibly separate. Multi-passage evidence has individual navigable citations.

Before a level change, capture the passage crossing a fixed reading line below the toolbar, its optional source-unit ID and its viewport position. A visible selected/focused evidence item takes precedence. Render the new level and restore the corresponding passage to that position, clamped to available scrolling. When a finer source unit has no equivalent in a compressed view, use its containing passage. All four text levels retain that passage container, including short/non-substantive and source-only passages.

Entering the section map stores the current return position. Returning without choosing a section restores it; choosing a different section deliberately navigates to that section. Opening/closing the source overlay preserves the reading column and returns keyboard focus to its trigger. Include neighbouring-passage controls. Support visible focus, keyboard operation, labelled dialogs, a 390-pixel frame width and reduced motion.

The reader requests a viewport-derived height through OWUI's supported `iframe:height` message when the host supplies its size, with a 720-pixel fallback. Focus reading uses browser fullscreen when permitted; the reading column scrolls inside it. Wide frames automatically provide a broader reading column, a multi-column section map and a wider source inspector; narrow frames stack the map cards. No separate widescreen mode is needed.

Cache the current reading anchor before layout changes. On browser or parent-container resize, restore the passage and offset after reflow without recreating the iframe, changing reading level, collapsing expanded passages or closing source inspection. Retain the map's return position. This is live in-memory preservation, not persistence across reloads; no model calls are needed.

**Save place** reveals a selectable compact token containing a bookmark version, source/segmentation fingerprint, passage ID and level, with an optional within-passage offset. **Restore place** accepts pasted text, validates all fields and rejects incompatible source versions. An attempted clipboard copy may have a selectable-text fallback. The token contains no source text or credentials; it is not an access token. Automatic restoration uses only validated position fields in browser-local storage. The isolated-frame host bridge scopes keys to the signed-in account and checks source-window identity; it adds no sandbox permission. The manual token remains a fallback.

## Embed, persistence and failure behaviour

Emit ordinary status events during preparation. On completion or a recoverable failure/deadline, emit one final complete/partial HTML snapshot using an `embeds` event with `data.embeds = [html]` and `data.replace = true` for this dedicated response. Return a short textual result. Do not repeatedly replace a reader while the user is interacting with it: each replacement can recreate the iframe and lose position.

Keep the default opaque-origin sandbox: scripts enabled, same-origin access disabled. The reader makes no network requests and needs no cookies, bearer tokens, parent DOM access, external fonts or authenticated API bridge. Encode embedded JSON so strings such as `</script>` cannot escape their container; render source/generated strings with safe text nodes, never unsanitized HTML. Preserve useful Markdown characters as text in this first implementation.

OWUI persists the embed in saved chat-message metadata. Reopening restores its generated content, and creates a new frame whose reading state restores when compatible local storage is available. Verify the deployed iframe CSP permits the fixed script; do not automatically weaken CSP or enable same-origin access. Unsupported host configuration produces a clear setup failure and a readable fallback message.

Keep generation results only in invocation-local memory until the final snapshot is saved. Successful batches survive in a saved partial result. Rerunning creates a new result and may regenerate all batches; no failed-only resume, cross-chat cache, background updates or process-restart recovery is claimed. Cancellation propagates promptly and stops new calls. Delivering a partial embed after cancellation is best effort, not guaranteed. Browser disconnect and server restart behaviour must be recorded from the pilot, not presented as durable jobs.

Validate file access on each new generation. Saved readers follow OWUI chat access/retention, not a fresh original-file check on each opening. Keep source text, prompts and credentials out of Function logs; note that host/provider logging policy remains deployment-controlled.

## Initial bounds and Valves

These are conservative starting values to verify against the selected model and representative documents, not a throughput promise:

| Setting                                         | Initial value or rule                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `BASE_MODEL_ID`                                 | Required, permitted server text model                                                       |
| `MAX_SOURCE_CHARS`                              | 100,000 Unicode code points                                                                 |
| `MAX_PASSAGES`                                  | 500                                                                                         |
| `MAX_BATCH_SOURCE_CHARS` / `MAX_BATCH_PASSAGES` | 10,000 / 20, whichever binds first                                                          |
| `MAX_MODEL_CALLS`                               | 32 total, including repairs; reject an input whose minimum batch count already exceeds this |
| `FILE_READY_TIMEOUT_SECONDS`                    | 60                                                                                          |
| `MODEL_TIMEOUT_SECONDS`                         | 120 per call                                                                                |
| `RUN_TIMEOUT_SECONDS`                           | 600 for generation, ending in a labelled partial result when possible                       |
| `MAX_EMBED_BYTES`                               | 2 MiB UTF-8; validate before emission and constrain generated item lengths in advance       |

Character counts are not token counts. Set explicit output budgets and leave context room for prompts, identifiers and schema. Reject oversized source/structure before model calls; if the final embed unexpectedly exceeds its cap, return an actionable error rather than silently dropping source. OWUI's upload/type limits continue to apply. Keep the pilot group small and use existing host/provider capacity controls; this single Function does not introduce a distributed queue or rate limiter.

## Delivery plan

Budget **5–7 engineering days for one developer**, assuming the current rich-embed behaviour is available and a suitable text model is configured. Record extraction quality and larger-document latency during the pilot; these may change bounds or schedule.

| Slice                                       | Work and exit condition                                                                                                                                                                                                    | Estimate   |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1. Prove the host contract                  | Minimal Pipe with a fixed five-level fixture; real saved embed/reload; sandbox/CSP/height; file-context toggle with retained attachments; one isolated permitted model call. Confirm task guards and no stray chat writes. | 0.5–1 day  |
| 2. Authorised source and canonical passages | Three formats through OWUI; ambiguous/inherited attachments; pending/failed processing; exact segmentation and evidence IDs. Full text and deterministic map work without an LLM.                                          | 1 day      |
| 3. Validated generation                     | Bounded batches, all generated levels, evidence validation, deadlines, repair and partial results; provider-specific output settings verified.                                                                             | 1–1.5 days |
| 4. Usable reader                            | Level/anchor mapping, expansion, source overlay, manual bookmarks, keyboard/narrow-screen operation and safe embedding.                                                                                                    | 1.5–2 days |
| 5. Acceptance and pilot package             | Offline/browser checks, representative real documents, one bounded model evaluation, install/use guide and recorded compatibility limits.                                                                                  | 1 day      |

Slice 1 is a compatibility gate. If the deployed OWUI cannot persist/run the embed under its approved settings, record the concrete limitation before doing the full reader UI. The later [Studio Reader plan](../../../../open-webui-studio/docs/reader-poc-plan.md) remains available for durable jobs, cross-device bookmarks or a dedicated document library; it is not part of this Function PoC.

## Acceptance criteria

- DOCX, text PDF and Markdown each complete the attach → prepare → zoom → inspect → reopen loop in the actual OWUI deployment. A scanned/empty PDF with no usable extracted text produces a clear extraction message; text already supplied by configured OWUI OCR remains usable.
- Every verbatim extract equals its canonical source unit, including whitespace/Unicode. Every displayed AI point opens valid evidence; human review checks that claims retain the source's qualifications.
- All transitions among the four text levels preserve the active passage, with the chosen anchor within 24 CSS pixels of its intended position where scroll bounds permit. Section-map return, expansion and source inspection preserve the defined position/focus.
- Resizing the same iframe between wide and narrow layouts preserves passage, level, expansion and bookmark location. The source inspector stays open with unchanged evidence, the map adapts its columns, without horizontal overflow. Verify viewport-derived height where supported, the fallback height and fullscreen separately. Test both browser and parent-container width changes.
- After initial preparation, zoom, expansion, source inspection and bookmark restoration cause zero network/model calls. Saved generated content survives reload; reload position and manual restoration behave exactly as documented.
- Two-user tests reject unauthorized files and unavailable models before generation. Inherited/multiple attachments never silently choose the wrong document. Automatic title/tag tasks make zero Reader generation calls.
- Delegated calls do not inherit/mutate outer metadata, recurse into the Pipe, emit into the outer chat or bypass model access. Database sessions are not held during model waits.
- Malformed JSON, unknown evidence, provider errors, timeouts and call-budget exhaustion produce bounded behaviour with honest partial/source fallbacks. Cancellation starts no new calls. Explicit reruns do not imply reuse of previous results.
- Source strings containing HTML, scripts and document-borne instructions remain inert data. No credentials appear in the embed. Keyboard/dialog behaviour and a 390-pixel frame work under the actual sandbox.
- Include short and long business briefs, repeated sentences, headings absent/present, list items, table text, CRLF, emoji and extraction failures. Record preparation time, model calls/usage when available and final embed size for the pilot samples.

## Verified implementation references

These links identify current code contracts to use and retest; they are not a claim that the PoC has already been exercised live.

- [Pipe registration and injected parameters](../../../backend/open_webui/functions.py); [chat metadata and user-message handling](../../../backend/open_webui/main.py).
- [File access, content and processing routes](../../../backend/open_webui/routers/files.py); [shared-file authorization](../../../backend/open_webui/utils/access_control/files.py); [extracted-text storage](../../../backend/open_webui/routers/retrieval.py).
- [Internal completion utility and metadata merge](../../../backend/open_webui/utils/chat.py); [file-context capability gate](../../../backend/open_webui/utils/middleware.py); [model capability editor](../../../src/lib/components/workspace/Models/Capabilities.svelte).
- [Embed persistence](../../../backend/open_webui/socket/main.py); [response embed rendering](../../../src/lib/components/chat/Messages/ResponseMessage.svelte); [iframe sandbox and height contract](../../../src/lib/components/common/FullHeightIframe.svelte).
- [Official OWUI Rich UI documentation](https://docs.openwebui.com/features/extensibility/plugin/development/rich-ui/).

## PDF and theme corrections (0.1.3)

Widely spaced PDF prose is eligible for generation; table detection requires explicit delimiters or repeated aligned short columns. Conservative title-case headings improve navigation, while reference codes remain source text. Reading views collapse display spacing without altering evidence strings. Batch failures retain safe, specific diagnostics.

The host iframe now propagates OWUI light/dark through CSS `color-scheme` without changing sandbox permissions. The Function has a guarded compatibility path for hosts where same-origin was already enabled. The 71 local checks cover both paths, including opposite browser/app themes and live switching. Saved readers contain their own UI code and need regeneration for Function changes.

## Streaming compatibility (0.1.4)

Optional `STREAM_COMPLETIONS` privately collects bounded Chat Completions SSE with terminal-state checks, explicit cleanup and unchanged source/schema validation. It does not stream unvalidated model text into the chat. Interrupted/filtered/truncated/error streams remain failures, and raw Responses API events receive a clear unsupported-mode diagnostic. Version 0.1.4 passed 85 combined checks, including cancellation and cleanup.

## Responsive reading and live resizing (0.1.5)

Reading width, section-map columns and source-inspector width adapt automatically to the available frame. Cached anchors preserve reading position across live browser and container resizes while keeping the existing iframe and interaction state. No extra mode, generation call or height change is required.

The [resize suite](../tests/test_document_reader_resize.py) adds four scenarios covering the same iframe at 1,120 → 1,800 → 390 → 1,120 pixels, including source inspection and map return. All 89 local checks pass: 75 backend checks and 14 browser scenarios. Segmentation remains version 3, so bookmarks from 0.1.4 stay compatible when the source text and segmentation settings are unchanged. The installation guide records the live 0.1.5 responsive check separately from the initial 0.1.4 generation check.

## Structured reading (0.1.6)

The Function formats an inert Markdown subset and preserves exact raw extraction and offsets for source inspection. Conservative PDF rules group short covers and recognisable contents listings as source-only front matter. A repeated versioned footer requires at least three copies and at least two neighbouring page labels before exclusion; substantive repeated statements remain eligible. Footer and OCR placeholder units remain in the source partition but are absent from model batches and rejected as generated evidence. Repeated footer text is shown once as inspectable source metadata, retaining edition/date context.

Segmentation version 4 requires a new bookmark for newly prepared snapshots. The five new structure checks cover source recovery, front matter, exclusions and invalid citations, repetition false positives, fenced headings, nested lists, emphasis, exact extracts and an untouched source inspector with no network requests. Full page metadata and original-page navigation remain a separate OWUI ingestion change.

## Understanding acceptance (0.2.0)

Prioritise purpose, key concepts, navigable source-derived hierarchy and qualified interpretations. Selected list lead-ins include their source children; generated evidence remains validated. Run the task comparison described in the guide with at least three pilot users before claiming better comprehension or time saved. Front-matter suppression and semantic labels cannot repair lost PDF layout or missing original page boundaries.

## Evidence selection correction (0.2.1)

A same-passage evidence selection now visibly distinguishes the selected sentence with an underline and a numbered location label, while retaining the complete source context and other cited highlights. The additional browser regression reproduced the previous ineffective feedback and verifies both selections and surrounding-passage navigation. The fix is installed and checked in the live Guidelines Reader.
