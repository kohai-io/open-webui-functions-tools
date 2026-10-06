# Document Reader

Document Reader is an Open WebUI Pipe Function for reading one document at five levels: **Section map, Takeaways, Explanation, Extracts and Full text**. Attach a DOCX, text-based PDF or Markdown file, prepare it once, then change levels and inspect the source inside the saved response.

The installable artifact is [document_reader.py](../document_reader.py). It uses Open WebUI's existing extraction, users, file permissions and configured models. It adds no routes, provider credentials, parser packages or database tables. The [PoC plan](document-reader-poc-plan.md) records the design and acceptance criteria.

This package targets the current OWUI checkout, version 0.11.3. Local verification covers 0.2.1, installed on the live host on 6 October 2026 after the 0.2.0 overview-first check. Recorded earlier PDF checks cover generation in 0.1.4, the responsive update in 0.1.5 and structured reading in 0.1.6. These checks do not establish compatibility with another deployment.

The responsive layout introduced in **0.1.5** automatically widens the reading column and source inspector on large screens, and arranges the section map in columns when space permits. No extra mode is needed. Resizing the browser or the Reader's containing panel preserves the cached reading position, selected level, expanded passages and open source inspector. Version 0.2.0 fits the iframe height to the browser viewport when host integration is available, with a 720-pixel fallback. **Focus reading** uses browser fullscreen when the embed permits it. The reading column still scrolls internally.

The earlier PDF fixes are included: widely spaced prose stays eligible for generation, conservative title-case headings aid navigation, and reference codes remain source text. Reading views collapse excessive spaces; **Inspect source** preserves exact extracted whitespace and evidence offsets. Replace the installed Function's code and retain its existing Valves. Existing saved embeds keep their old code: regenerate a response to get the updated Reader.

The Reader follows OWUI's resolved light/dark setting. For the default opaque sandbox, the accompanying host change in [FullHeightIframe.svelte](../../../src/lib/components/common/FullHeightIframe.svelte) sets the iframe's CSS `color-scheme` from `html.dark`; this must be included in the deployed frontend. The browser passes that scheme into the iframe without granting parent access. On older hosts where same-origin is **already** enabled, the Function can read and observe only the parent's theme class as a compatibility fallback. Do not enable same-origin to obtain this behaviour; use the host CSS patch for isolated embeds.

Version 0.1.4 also adds the optional `STREAM_COMPLETIONS` Valve (default `false`). Set it to `true` for a Chat Completions connection that requires streaming. Reader collects the stream internally, requires a terminal completion marker, and validates the full JSON before displaying anything. Truncated, filtered, refused, interrupted or oversized streams remain labelled failures; transport requests are never retried automatically. Raw Responses API event streams are not supported by this mode: use a compatible Chat Completions connection or leave it disabled.

The earlier 0.1.1 model-selection fix is included: OWUI's normal server-connected providers labelled `connection_type: "external"` are supported. Browser-direct models use the separate `direct` flag and remain unsupported.

## What is saved

The result is a self-contained reader embedded in an assistant message. Opening an existing result, changing levels, expanding a passage, inspecting evidence and restoring a bookmark require no further model calls.

**This reader is saved in the chat and contains the full extracted document. Sharing or exporting the chat may disclose that text.** Removing or restricting the original file does not remove a copy already saved in a chat. The saved result follows the chat's access and retention rules.

Full text is a formatted reading view of OWUI's frozen extraction, in source order. Markdown headings, emphasis, lists, blockquotes and fenced code use an inert formatting subset. Recognised PDF page furniture and OCR image placeholders are hidden from the body; repeated footer text, including edition/date information, appears once as source metadata with an inspection link. Inspect source preserves every original character and offset. Unsupported Markdown and document HTML remain inert text; links and images do not load external resources. It is not a Word or PDF page viewer. Keep the attachment to open or download the original. Extracts copy exact source units from this frozen text; explanations and takeaways are labelled AI interpretations. A source citation makes an interpretation traceable, but does not establish that the interpretation is correct.

## Install and configure

An OWUI administrator performs these steps. No installation or configuration change is performed by opening this guide.

1. In **Admin Panel → Functions**, create a Function and paste the entire contents of [document_reader.py](../document_reader.py) into its code editor. Give it the name **Document Reader**, save it and enable it. The local **Import JSON** control expects an exported Function JSON file, not the raw Python source.
2. Open the Function's **Valves** and set `BASE_MODEL_ID` to the exact ID of an existing, server-backed text model. Choose a model the pilot users are allowed to use. Do not select Document Reader itself, another Pipe, a pipeline model, an arena model or a browser-direct connection.
3. In **Workspace → Models**, configure the Document Reader model entry with **File Upload enabled** and **File Context disabled**. If a separate workspace model entry is needed, use the installed Document Reader Pipe as its base. These settings correspond to `info.meta.capabilities.file_upload = true` and `info.meta.capabilities.file_context = false`.
4. Disable unrelated tools, web search and code execution for this dedicated Reader entry. Do not attach Knowledge collections, folders or default documents to it. Keep status updates enabled so preparation progress is visible.
5. Grant the intended pilot users access to the Reader entry and its configured generation model. Confirm both using an ordinary user account: an administrator's successful run does not prove another user has access.
6. Keep the default embed sandbox: **scripts enabled, same-origin access disabled**. The reader needs no cookies, API bridge, external scripts, fonts or network access. Check that the host's iframe Content Security Policy permits its fixed script. Do not enable same-origin access or weaken the host policy to work around an unexplained failure.
7. Run the live compatibility check below with a small, non-sensitive Markdown file before trying longer documents.

Disabling File Context avoids normal retrieval augmentation before the Pipe runs; it does not disable uploads or OWUI's stored text extraction. The Pipe needs the current message's attachment metadata even though the file content is not injected into the ordinary chat prompt.

For example, if the model ID in your OWUI catalogue is `chatgpt/gpt-5.6-sol`, enter exactly that string in `BASE_MODEL_ID`, including the slash. Do not use its display name, a full URL or the URL-encoded `%2F`. Whether the model can be used depends on its server connection and access grants, not the spelling of its ID.

### Valves

Start with these defaults. Tune them against the selected model and representative documents rather than assuming a larger number is better.

| Valve                        | Default                   | Purpose                                                                                                                              |
| ---------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `BASE_MODEL_ID`              | Empty; must be configured | Authorized, server-backed text model used to prepare the reader.                                                                     |
| `MAX_SOURCE_CHARS`           | `100000`                  | Maximum extracted source length, measured in Python Unicode code points.                                                             |
| `MAX_PASSAGES`               | `500`                     | Maximum canonical passages after grouping adjacent prose; this is not a PDF page or paragraph count.                                 |
| `MAX_BATCH_SOURCE_CHARS`     | `10000`                   | Source-character limit for one generation batch.                                                                                     |
| `MAX_BATCH_PASSAGES`         | `20`                      | Passage-count limit for one batch. Both batch limits apply.                                                                          |
| `MAX_MODEL_CALLS`            | `32`                      | Total generation-call budget, including repair calls.                                                                                |
| `FILE_READY_TIMEOUT_SECONDS` | `60`                      | Maximum wait for an uploaded file's extracted text.                                                                                  |
| `MODEL_TIMEOUT_SECONDS`      | `120`                     | Timeout for one delegated model call.                                                                                                |
| `RUN_TIMEOUT_SECONDS`        | `600`                     | Overall generation deadline.                                                                                                         |
| `MAX_EMBED_BYTES`            | `2097152`                 | Maximum final reader HTML size: 2 MiB in UTF-8.                                                                                      |
| `MAX_OUTPUT_TOKENS`          | `6000`                    | Output-token budget sent to the selected provider.                                                                                   |
| `OUTPUT_TOKEN_PARAMETER`     | `max_tokens`              | Output-budget parameter; set to `max_completion_tokens` when required by the configured provider/model.                              |
| `USE_JSON_MODE`              | `false`                   | Optional JSON response mode. Enable only when supported by the selected backend. Local validation always applies.                    |
| `STREAM_COMPLETIONS`         | `false`                   | Privately collect Chat Completions SSE; useful when the connection fails on completed responses. Full JSON validation still applies. |

Character limits are not token limits. Leave context space for prompts, source IDs, output and any provider reasoning budget. The Function permits at most one repair call for a completed but invalid batch response within the call/time budgets. It does not automatically repeat an ambiguous transport failure. Reducing batch size is often more useful than increasing timeouts when structured results are incomplete.

Adjacent prose is grouped up to the smaller of 2,400 characters and `MAX_BATCH_SOURCE_CHARS`, splitting only between preserved source units. A single larger source unit may occupy its own passage if it fits the batch limit. Current snapshots use segmentation version 4. Version 0.1.6 groups recognised PDF covers and contents listings as source-only front matter, excluded from AI generation and collapsed at compressed levels. It retains excluded footer/placeholder units in the source partition and prevents generated evidence from citing them. New snapshots therefore need new bookmarks; 0.1.4/0.1.5 bookmarks still work with their matching older saved readers. Previously saved readers and their bookmarks continue to work together.

No provider key is entered in these Valves. The delegated call uses OWUI's existing model connection and the requesting user's model permissions. The internal completion utility does not run the full outer chat middleware/filter pipeline; verify any deployment-required processing policy before using this with business documents. Function code avoids document/prompt logging, but host DEBUG logging and provider logging remain deployment-controlled.

## Prepare and read a document

1. Start an ordinary **saved chat** and select the configured Document Reader entry. Temporary chats and API-only calls without the required message/embed context are not supported.
2. Attach **one** DOCX, PDF or Markdown file to the message you are about to send. Wait for OWUI's upload/processing indication, then send “Prepare this document”. An existing individual file is also suitable when the UI lets you attach it explicitly.
3. Wait for preparation status updates. Reader loads the extracted text and processes bounded batches. The message prompt starts preparation; it is not a general document question-and-answer request.
4. The result opens at **Section map**, showing substantive topics, source-linked key concepts and a purpose overview when the source has a recognised purpose/scope section. Source-derived subsections sit under their parent; cover and contents text stays available in a collapsed group. The reader shows its source filename; the accompanying response reports whether preparation completed or produced a partial result and how many batches were prepared.
5. Use the level buttons to read more or less detail. **Expand here** unfolds a passage locally. **Inspect source** opens the cited wording, with nearby-passage controls. Closing the source inspector returns focus to its trigger.
6. Use the section map to navigate. Returning from the map without choosing a section restores your previous reading position; choosing a section deliberately moves there.
7. The footer reports whether your position is saved automatically in this browser. **Save place** also provides an optional bookmark token to keep yourself. After reopening the saved chat, choose **Restore place** and paste the token. Use selectable text if clipboard access is unavailable in the sandbox.

Use one document per chat during this PoC. Reader selects the file explicitly attached to the current user message, or the corresponding stored user message when regenerating. A file inherited from earlier chat history, a project, a folder or a Knowledge collection is not sufficient. Attach the intended document again rather than relying on an old upload. Multiple or unsupported attachments receive a selection error rather than silently choosing a file.

The layout adapts to the available frame width, including changes made while reading. Wide frames provide a broader reading column, a larger source inspector and a section-map grid; narrow frames stack the map cards and keep the controls accessible. These changes preserve your place without reloading the iframe or making model calls.

The bookmark token contains a source/version fingerprint and location, not document text or credentials. It only applies to a compatible snapshot. Automatic restoration retains the level, passage, offset and expanded passages for the same saved snapshot. It uses browser-local storage where the existing host permits it, or the accompanying isolated-frame host bridge. Storage unavailable or blocked leaves the manual token fallback. A new snapshot starts at Section map. A new preparation may not accept an old bookmark if the source or segmentation changed.

## Partial results and source quality

Successful batches remain readable when another batch fails or the generation deadline is reached. Passages without a valid generated representation show labelled source-text fallbacks. A partial result is not a complete generated summary or map. Retrying the Function prepares a new result and can regenerate every batch; it does not resume just the failed work.

Cancellation stops new calls and attempts to cancel the current wait. An upstream provider may already have accepted the request. This version does not save a new partial reader after cancellation; partial snapshots are produced for recoverable batch failures or generation deadlines. Browser disconnects and server restarts do not have durable job recovery in this PoC.

Support depends on the host's OWUI extraction:

- DOCX and Markdown retain structure only to the extent the extractor supplies it. Missing headings receive deterministic part labels.
- PDFs need usable extracted text. This Function does not configure OCR; text already produced by the host's OCR can be read.
- Multi-column pages, charts and complex tables may lose meaning during extraction. Ambiguous detected table regions remain source-only. Check the original when reading order, column association or omitted images matter.
- Source size and structural limits reject an oversized document before generation rather than silently truncating it. A single source unit that exceeds the batch limit also needs a smaller input or a reviewed limit change.

## Troubleshooting

| Symptom                                                         | Check or action                                                                                                                                                                                             |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reader is absent from the model selector                        | Enable the Pipe, check the workspace entry and its access grants, then refresh the model list.                                                                                                              |
| No attachment control                                           | Enable **File Upload** on the Reader entry and check the user's upload permission.                                                                                                                          |
| Reader asks for an explicit document despite an old attachment  | Attach the intended file to the current message. Inherited files are deliberately not selected.                                                                                                             |
| Multiple/unsupported attachment error                           | Send one DOCX, PDF or Markdown file. Remove images, links, collections and folders from that message.                                                                                                       |
| Extracted text is not ready                                     | Let OWUI finish processing, then rerun. Inspect OWUI's file-processing error if it failed. Reader does not automatically reprocess the file.                                                                |
| No usable text from a PDF                                       | Check that text can be extracted on this host. Use a text-bearing source or the host's approved OCR workflow.                                                                                               |
| File indexing failed but text exists                            | Reader can use stored text independently of vector indexing. Review the reported extraction/indexing condition and compare with the original.                                                               |
| File not found or access denied                                 | Check the current user's file access. Attachment IDs or text supplied in a request do not override server permissions.                                                                                      |
| Model unavailable; admin succeeds but users fail                | Check `BASE_MODEL_ID`, its access grants and any underlying base-model access. Use an allowed server-backed text model.                                                                                     |
| External API model rejected as browser-direct in version 0.1.0  | Update the Function code to 0.1.1 or later. Normal server-connected external providers are supported; keep the same exact `BASE_MODEL_ID`.                                                                  |
| “Querying” / “No sources found” before Reader preparation       | Turn **File Context off** on the Document Reader model entry and leave **File Upload on**. These messages come from OWUI's normal retrieval step, which Reader does not need.                               |
| Provider rejects `max_tokens`                                   | If the provider requires it, set `OUTPUT_TOKEN_PARAMETER` to `max_completion_tokens`. Verify the supported output budget on that model.                                                                     |
| Provider rejects JSON response mode                             | Leave `USE_JSON_MODE` at `false`; outputs still undergo local schema/evidence validation.                                                                                                                   |
| Invalid output, unknown evidence or repeated partial batches    | Try smaller batch limits or a model that reliably follows structured-output instructions. Check both output budget and context capacity. An evidence reference is required, not guessed.                    |
| Long wait or timeout                                            | Review provider latency and OWUI status. Adjust bounds only after a small document succeeds. A rerun may incur new generation charges.                                                                      |
| Reader theme differs from OWUI                                  | Update to 0.1.3 and deploy the `FullHeightIframe.svelte` colour-scheme rules for opaque embeds. Existing saved readers need regeneration for Function UI changes. Do not enable same-origin to fix a theme. |
| Ordinary PDF prose labelled table-only                          | Update to 0.1.3 and regenerate. Repeated inter-word spacing alone no longer counts as a table.                                                                                                              |
| Blank embed or buttons do not work                              | Check the deployed rich-embed support, browser console and iframe CSP. Scripts are required; same-origin access is not. Record the host limitation before changing its policy.                              |
| Reader disappears on reload                                     | Confirm the chat was saved and the deployed host persists assistant-message embeds. Treat this as a failed compatibility check.                                                                             |
| Reload lost the reading position                                | Check the footer for automatic-saving support. Deploy the host bridge for isolated embeds, or use **Restore place** with a saved token when browser storage is unavailable.                                 |
| Bookmark is rejected                                            | Restore it into the matching document snapshot. A changed source or segmentation version requires a new bookmark.                                                                                           |
| Passage limit reached on a PDF with many short extracted blocks | Update to 0.1.2 or later before raising `MAX_PASSAGES`. Adjacent prose blocks are grouped without removing text; isolated headings and table blocks still count separately.                                 |
| Embed-size limit reached                                        | Use a shorter document or review the configured limit. Reader does not drop source passages to make the result fit.                                                                                         |

## Verification

### Local checks

The focused suites are [backend contracts](../tests/test_document_reader.py), [failure diagnostics](../tests/test_document_reader_diagnostics.py), [reader interaction](../tests/test_document_reader_browser.py), [theme inheritance](../tests/test_document_reader_theme.py), [stream collection](../tests/test_document_reader_streaming.py) [live resizing](../tests/test_document_reader_resize.py) and [extraction structure](../tests/test_document_reader_structure.py). **98 checks pass for Function version 0.2.1**: 81 backend checks and 17 browser scenarios. The added [understanding and restoration suite](../tests/test_document_reader_understanding.py) covers hierarchy, concept evidence, list context, isolated-frame automatic restoration and fullscreen. Four frontend bridge checks also pass. Coverage includes PDF word spacing, title-case/reference headings, exact source offsets, explicit table formats, bounded failures, opposite OS/app themes, live theme changes and opaque sandbox isolation.

The supplied three-page policy PDF was inspected locally. Its 11,397-character pypdf extraction now yields 12 sections and 10 eligible prose passages, without falsely treating the prose as tables; all characters and offsets match the input. This is an extraction check, separate from provider generation. Synthetic regression fixtures contain no private policy text.

From the repository root, using the Python environment with the test dependencies installed:

```powershell
python -m pytest functions_tools/functions/tests/test_document_reader.py -q
```

To run all focused suites, use Python with Pydantic 2, Starlette, pytest and Playwright installed, plus an installed Chrome browser:

```powershell
$env:RUN_DOCUMENT_READER_BROWSER_TESTS = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest functions_tools/functions/tests -k document_reader -q
```

The browser tests run headlessly in an opaque-origin iframe with scripts allowed and same-origin access disabled. No OWUI server, credentials or provider calls are used. Set `DOCUMENT_READER_SCREENSHOTS` to a local directory to save inspection images. Check output for skipped cases rather than counting skips as passes.

Local coverage exercises attachment provenance and permissions, exact Unicode/whitespace recovery, batching and bounds, isolated delegated requests, schema/evidence failures, partial results and safe embedding. Browser checks cover all text-level transitions, source inspection, map return, bookmark restoration, narrow frames, keyboard focus and zero reader network requests after preparation. Resize checks change the same iframe through 1,120 → 1,800 → 390 → 1,120 pixels using both browser and parent-container widths. They verify preserved passage, level, expansion and bookmarks, an open source inspector with unchanged evidence, responsive map columns, no overflow and a stable fallback height. The new isolated-frame scenario additionally verifies host-derived viewport height and fullscreen.

### Recorded live PDF check — 5 October 2026

Function 0.1.4 was installed through OWUI's Function editor. The existing `BASE_MODEL_ID` remained `chatgpt/gpt-5.6-sol`; `STREAM_COMPLETIONS` was enabled. No sandbox or access permissions were changed.

- The completed-response trial produced HTTP 500 for eight batches, with only 2/10 ready. A simple streaming control succeeded on the same model.
- The subsequent streaming Reader run completed **10/10 batches in 10 calls**: 12 sections, 22 source passages (12 headings and 10 generated prose passages), with a complete snapshot.
- Live full-text rendering exactly matched all 11,378 characters frozen from OWUI extraction. All 28 displayed extracts matched their source units. Source inspection opened with cited units highlighted, and all 12 section-map cards rendered.
- Reloading the saved chat restored the same completed snapshot and generation metadata. It did not start another preparation.
- The Reader displayed light mode within light-mode OWUI. This host already allowed same-origin embeds, so the guarded compatibility path supplied the app theme. The isolated-iframe CSS fix was tested locally; its frontend deployment has not been performed here.

This is one document and one administrator session. The broader format, normal-user permissions, load and recovery checks below remain pilot acceptance work. The earlier “Querying / No sources found” status is OWUI retrieval; disable File Context on the dedicated Reader model entry as described above if it remains enabled.

### Recorded live responsive check — 5 October 2026

Function 0.1.5 was installed with the existing model and streaming Valves retained. Regenerating the same PDF completed all 10 batches in 11 calls, including one structured-output repair. The saved snapshot reopened successfully in a separate browser tab.

The live Reader retained Full text, passage 17 and its approximately 86-pixel offset through browser widths of 1,846, 1,050 and 460 pixels, then back to 1,846. Offset drift remained below one pixel, with no horizontal overflow. The reading column measured 1,100 pixels in the wide view, and the section map displayed three columns. Light-mode OWUI still produced a light Reader. The original tab was returned to its saved Full text position after verification.

### Recorded live structured-reading check — 5 October 2026

Function 0.1.6 was installed and the reported Guidelines PDF regenerated using the existing streaming model connection. The final run completed **15/15 batches in 15 calls**, compared with 18 batches before front-matter exclusion. The 28,014-character source partition exactly matched the prior saved OWUI extraction.

- The cover and contents became source-only front matter, replacing the logo-only Part 1 and avoiding three low-information generation batches. The snapshot contains 18 sections and 37 passages.
- All 13 repeated versioned footers were retained in source units and excluded from body rendering and AI preparation. The footer's edition/date text appears once as inspectable source metadata.
- Eight nested lists and ten bold labels rendered; section titles no longer showed stray Markdown emphasis markers. Section navigation placed the visible heading about 18 pixels below the reading viewport top.
- All 52 extracts exactly matched their selected raw source units. Generated passage citations referenced no excluded units. Source inspection retained the original passage text, including the repeated footer.
- A separate opening of the saved chat loaded the same fingerprint and preparation timestamp with complete generation metadata. The original tab was left in Full text at the productivity section for review.

This verifies the reading layer on the existing extraction. It does not establish that OCR recovered every original PDF detail. Page boundaries, full layout metadata and original-page navigation still require the separate ingestion work.

### Live compatibility and pilot checks

Complete the remaining cases on the intended deployment before describing the full pilot matrix as passed:

Use the supplied [pilot fixtures and expected qualifications](../tests/fixtures/document_reader/README.md): [Markdown](../tests/fixtures/document_reader/pilot-brief.md), [DOCX](../tests/fixtures/document_reader/pilot-brief.docx) and [text PDF](../tests/fixtures/document_reader/pilot-brief.pdf). They contain the same short synthetic brief. Local extraction recovered matching wording from all three, and each extracted string passed through Reader segmentation without losing characters; the deployed OWUI extractor still needs the checks below.

1. As a normal pilot user, prepare a short Markdown document, inspect a citation, save the chat and reload it. Confirm the generated content remains, the footer accurately reports storage support, and automatic restoration or the manual fallback works.
2. Repeat with a representative DOCX and a text PDF. Compare full text, extracts and AI claims against the original; include qualifications, repeated sentences, lists and table text.
3. Switch through all four text levels at a middle passage, enter/leave the map and open/close source inspection. Resize the browser or containing panel between wide and narrow frames while reading and while the source inspector is open. Verify the layout adapts and place, reading level and keyboard focus behave as described.
4. Check unauthorized files and an unavailable generation model using another user account. Check no-file, multiple-file, inherited-file and scanned/empty-file cases.
5. Test malformed model output, a deadline and cancellation using controlled fixtures or a test provider. Confirm partial/source fallback labels, bounded retries and no implication of failed-only resume.
6. Confirm title/tag tasks do not prepare readers, delegated calls produce no stray outer-chat writes, and zoom/source/bookmark actions produce no network/model calls from the reader.
7. Record OWUI version, model ID/provider configuration, sandbox/CSP settings, sample type/length, preparation time, model-call count, any available usage, embed size and observed disconnect/restart behavior. Keep document text and credentials out of the test record.

Installing the Function, running the local suites and completing this live checklist are separate steps. None requires changing the default same-origin sandbox restriction.

## Overview-first reading (0.2.0)

The tabs now run **Section map → Takeaways → Explanation → Extracts → Full text**. Section map is the starting view; source headings determine the hierarchy, while AI concepts provide short evidence-linked navigation labels. These are navigation aids, not an assurance of interpretation accuracy. Generation prompts require important conditions and exceptions to remain visible. Extract selection includes list children when a selected lead-in would otherwise leave “including” or “for example” hanging. Every included unit remains verbatim.

The optional host integration comprises [documentReaderBridge.ts](../../../src/lib/utils/documentReaderBridge.ts) and [FullHeightIframe.svelte](../../../src/lib/components/common/FullHeightIframe.svelte). It accepts messages only from the embedded frame, validates this snapshot's IDs/key, and stores only position fields in an account-scoped browser key. It also reports the available viewport height. No document text or credentials are stored in the position record; no new sandbox permissions are needed. This frontend integration is prepared and tested locally, not deployed. Legacy hosts with same-origin already enabled can use the guarded compatibility path; do not enable same-origin to gain automatic restoration.

The focused Function suite and four bridge tests pass. The changed iframe component compiles independently, but the repository-wide frontend check fails across many files; it is not a clean deployment gate. Live 0.2.0 generation is verified below; ordinary-user acceptance remains outstanding.

Usability acceptance should compare the Reader with OWUI's original attachment preview on the same document: identify its purpose and three key concepts, locate a topic, state a qualified permission accurately, inspect the exact source, then reopen at the same place. Record task completion, wrong or missed conditions, source-check success and time with at least three pilot users. An expert walkthrough alone does not establish time saved or improved comprehension. Original PDF pages and extraction fidelity remain separate upstream work.

### Recorded live overview-first check — 6 October 2026

Function 0.2.0 was installed through the OWUI editor after explicit approval. The pasted code exactly matched the local source, and the saved header survived reload. Existing model/streaming Valves and sandbox permissions were retained. The Guidelines PDF regenerated with **15/15 batches in 15 calls**, 37 canonical passages, 18 source sections (16 substantive topics) and 54 source-linked concept labels. Its entire 28,014-character source matched the previous frozen extraction exactly.

The opening map shows purpose, navigation concepts and seven use-case subsections grouped under their parent. Concept navigation opened the productivity section. A source check confirmed that its explanation preserved contract/consent, transparency, challenge/human review, relevant data, a specific DPIA and other qualifications from the selected passage. This is a sampled interpretation review, not validation of every AI claim.

Reopening the saved chat automatically restored Explanation, passage p0012 and its expanded source view. Focus reading entered and exited fullscreen. Browser resizing from 1,846 to 390 pixels retained that passage, selected level and expansion without horizontal overflow. The temporary viewport override was reset, and the chat was left at Section map in OWUI's light theme. This host already allows same-origin embeds, so automatic storage and viewport sizing used the guarded compatibility path. The new isolated-frame frontend bridge remains locally tested and undeployed.

## Evidence selection correction (0.2.1)

A live report on 6 October 2026 exposed an inspector feedback defect: two cited units in the same passage stayed highlighted identically, while the evidence buttons changed their pressed state. If both units were visible, the existing nearest-scroll behaviour produced no movement, so selection appeared ineffective.

The selected evidence now has a distinct background and underline. Other cited units remain highlighted, and the location label states **Evidence N of M**. Complete extracted passage text remains unchanged, including its whitespace. The same-passage regression scenario failed before the fix and passes afterward; all 98 focused Reader checks pass.

Function 0.2.1 was saved through the live editor with exact local-source verification and existing Valves retained. The Guidelines response regenerated with 15/15 batches in 15 calls and 28,014 source characters. Live switching between Evidence 1 and Evidence 2 moved the selected underline between the approval-conditions sentence and the individual-responsibility sentence; the location label also changed. The inspector was left open with Evidence 2 selected for review. Earlier saved responses retain their old embedded interface.
