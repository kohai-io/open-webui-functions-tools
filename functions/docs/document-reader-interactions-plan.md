# Document Reader: zoom controls, passage questions and reading briefs

Status: implemented in Function 0.3.0 with 123 focused tests passing. This document retains the agreed scope and acceptance criteria. Live installation and compatibility evidence are recorded in the main Reader guide.

Constraint: **all implementation changes must remain in the Pipe Function package. Do not change OWUI frontend code, routes, middleware, model configuration or database schema.** Existing OWUI capabilities can be reused, with local fallbacks where a host does not expose them.

## Outcome and scope

Add three connected features to the existing Document Reader Pipe:

1. Change reading level by scrolling or dragging within the level bar.
2. Ask a question about a specific passage through the ordinary OWUI composer and receive a source-cited answer in the chat.
3. Collect useful passages and download a Markdown reading brief locally.

Do not add number-key shortcuts, global key handlers, global wheel interception, continuous pinch zoom, annotations, translation, whole-document question answering or new extraction engines. Continue using the existing five levels and source-linked word animation. Original PDF fidelity remains governed by OWUI extraction.

## Level-bar zoom

Keep the five labelled buttons in their current overview-to-detail order. Add a small instruction beside or below the bar: “Scroll or drag here to change detail.” The labels remain the primary discoverable controls.

- Wheel input over the bar steps towards more or less detail. Input elsewhere retains its existing scrolling behaviour. Do not intercept Ctrl/Cmd-wheel, which can belong to browser zoom or trackpad pinch.
- Accumulate small wheel deltas and normalise line/pixel delta modes. Require a meaningful threshold, reset after an idle gap or direction reversal, and discard inertial excess after a step. A single wheel burst must not race through all five levels.
- Drag horizontally across the bar to choose a level. Preview the target label as the pointer moves; release commits one transition. This first version snaps to the five levels rather than rendering intermediate reading states.
- On touch screens, begin a drag only after clear horizontal intent. Vertical movement remains scrolling, and a cancelled gesture makes no level change. Ordinary taps still select buttons.
- Preserve the current passage and expansion state. Use the 0.2.2 transition when committing a change, and its reduced-motion/interruption behaviour.
- At either end, leave the level unchanged. No gesture may escape into a document-wide zoom handler.
- Keep the existing buttons' normal Tab/Enter/Space accessibility. Add no custom keyboard shortcuts; the user's restriction concerns conflicts with OWUI, not removal of standard button operation.

This is an embedded HTML/JavaScript change only. It uses no OWUI requests, host storage changes or model calls.

## Ask about this passage

### Reading workflow

Add “Ask about this passage” alongside “Expand here” and “Inspect source”. Opening it shows a compact dialog containing the section name, a source preview and a question field. Optional starter questions populate that field: “What does this require?”, “What conditions or exceptions apply?” and “Explain this in plain English”.

Show a human-readable draft identifying the document, section and question. The default action is “Copy question”, followed by pasting into OWUI's composer. Try clipboard copying on the user's click and offer selected text for manual copying if the browser blocks it. This works without reading or replacing an existing chat draft.

An optional “Replace chat draft” action uses OWUI's existing `input:prompt` postMessage support to populate the composer. Its label makes the replacement explicit: the isolated embed cannot inspect the current composer contents or detect the selected model. Keep the copy fallback available because this existing hook varies between hosts. Neither action submits a message, switches models or reattaches the entire document.

The dialog explains that the user should send the question with Document Reader selected. The user reviews the draft and presses OWUI's ordinary Send button. The Pipe recognises it as a passage question and returns a short answer in the chat, with numbered citations that open OWUI's source viewer. The original Reader embed remains intact. The question must never start preparation of a new Reader.

Without an OWUI change the embed cannot enforce model selection, show a composer context chip, preserve an arbitrary existing composer draft automatically or confirm that a postMessage draft was received. The copy-first flow and visible “Send with Document Reader selected” guidance are intentional limitations of this implementation. A question sent to another model will not use the Reader's question handler.

Phase one covers explicit passage questions, including questions from a partial or source-only passage. Bare follow-ups such as “what about exceptions?” without a passage context do not silently reuse a previous target: explain how to select a passage again. Do not add arbitrary text-selection handling or multi-turn context inference in this release.

### Function-only routing and saved context

Use ordinary saved message text rather than adding OWUI message metadata fields or a new host bridge:

- Add the preparation response's assistant-message reference to the Reader's own snapshot. Resolve the current chat from authenticated server metadata, never from a client-selected chat ID.
- Include a Markdown “Source: document — section” link in the copied draft. It opens the existing source chat; its URL fragment carries a small versioned routing envelope containing only the saved Reader message ID, snapshot fingerprint and passage ID. The fragment adds no new OWUI route and makes no request to an external address. A readable source link avoids relying on HTML comments surviving the composer.
- Recognise the explicit Reader-question heading and exactly one source link with the known Reader-question fragment version. Require the link's path to match the current saved chat, reject external destinations and never fetch the URL. Limit the encoded reference to 1,200 characters and the question to 2,000 characters; reject duplicate, malformed or unknown fields. The envelope is untrusted routing data, never instructions or proof of permission. No source text, model IDs, arbitrary URLs or provider tokens belong in it.
- Resolve the reference against the authenticated user's current saved chat and validate it against the server-stored Reader snapshot before making a model call. The browser supplies a target reference, not authoritative document content.
- OWUI already persists ordinary user message content and passes it to the Pipe. Read the latest user message, strip the routing link/envelope before model generation, and retain it in the saved question so regeneration has the same explicit target. If editing removes the source reference, require a fresh question from the Reader rather than guessing the target.
- The existing `input:prompt` mechanism, when used, remains unchanged. No new postMessage message types, composer state, host storage fields or frontend permission checks are added.

Do not grant same-origin iframe access, put provider credentials in the embed or send model requests from the browser. Explain the source link in the question dialog as “This draft includes a reference to the saved passage”; do not imply that it is secret or tamper-proof, or that clicking it scrolls OWUI to the exact passage. Source permissions and evidence validation remain server-side responsibilities.

### Pipe question path

Dispatch a marked passage question before the current attachment-selection/preparation path, after the same authentication and chat-ownership checks. A malformed Reader-question marker must fail explicitly and must not fall through into preparation. New document uploads continue to use the existing preparation workflow. An unmarked message without a current attachment receives actionable guidance rather than guessing between preparation and question answering.

1. Load the referenced saved assistant message and its Reader embed from the current owned chat. Bound embed size and parse only the known inert Reader data block; never evaluate its HTML or script. Validate snapshot structure, fingerprint, source-unit IDs and offsets. Older snapshots without the required question reference receive an instruction to regenerate the Reader.
2. Verify the original file still exists and the requesting user currently has read access. Recheck access to the configured `BASE_MODEL_ID`. No question may bypass these checks by supplying another snapshot, file ID or copied source string.
3. Answer against the frozen source edition the user was reading, not freshly reprocessed text with different offsets. If current extraction differs, identify the answer as concerning the saved edition and invite explicit preparation of a new Reader. Do not combine editions or invent original PDF page numbers.
4. Build bounded context from the complete target passage, its heading and nearby passages within the same section when these preserve a lead-in, list or qualification. Exclude extraction furniture. Cap context at 12,000 source characters and never silently truncate the target passage. If it cannot fit, stop before a model call and explain the limit.
5. Use raw source units as evidence, not generated takeaways as truth. Treat source instructions as quoted document data. State that the answer is limited to the supplied passage context; if a question needs other sections or missing table relationships, say that context is insufficient.
6. Delegate through OWUI's existing server-backed completion utility using the same isolated-request and streaming compatibility handling. Use at most one completion plus one schema repair, with bounded answer/context sizes, existing timeout/cancellation behaviour and no automatic retry after an ambiguous transport failure. Do not run the document generation batches.
7. Validate a separate answer schema: a short list of answer points, each with allowed source-unit IDs, or an insufficient-context result. Reject missing, unknown or out-of-context citations. Render inert answer text and OWUI citation events containing the exact cited wording and document/section labels. Citation validity demonstrates traceability, not factual correctness.
8. Save the answer and citations through OWUI's existing message lifecycle. Do not replace the original Reader embed. Auxiliary title/tag tasks remain excluded from question generation.

First build a compatibility spike that copies and submits one synthetic passage question through the unmodified composer, retains its routing envelope in the stored message, returns one real OWUI citation and successfully regenerates that answer. Also test the existing `input:prompt` hook with explicit draft replacement. This confirms the end-to-end contract before building the complete question dialog.

## Export a reading brief

### Collection and review

Add an “Add to brief” toggle to substantive passages and a “Brief (N)” control in the Reader header. Add it to the source inspector too, selecting the complete owning passage rather than whichever citation is currently highlighted.

Selecting a passage means “keep this point in my brief”. It is independent of the current reading level and never triggers AI generation. The brief drawer lists selected passages grouped in document order, with their section name, a short preview and Remove controls. Empty state explains how to add a point. Limit the first version to 50 passages and 60,000 exported characters, with visible feedback before exceeding either limit.

Default export: **Takeaways with supporting source wording**. Offer an optional **Include explanations** toggle. Preview the resulting document before “Download Markdown”. If generated levels are unavailable, include the exact source passage under “Source only”; do not omit a selected passage or fabricate an interpretation.

The brief contains:

- Document filename, section headings and the saved edition's preparation date.
- The selected takeaways and optional explanations, explicitly labelled AI generated.
- Deduplicated exact cited source units, in source order, with stable numbered references. Preserve the context needed by selected list lead-ins and qualifications.
- A partial-preparation note when applicable, and a concise statement that source wording comes from OWUI extraction and may differ from original layout.

Do not export the whole source document by default, local file paths, credentials, opaque routing metadata or unrelated chat history. Escape Markdown/control syntax in untrusted filenames and generated prose. Put exact source units in plain-text fenced blocks, choosing a fence longer than any backtick run in the source, so quotes retain their original characters without introducing active HTML or remote resources. Use a safe `.md` filename.

### Local state and download

Build a UTF-8 Markdown Blob inside the embed and download it using the existing `allow-downloads` sandbox permission. Provide a copyable preview if download is unavailable. No model calls or server export endpoint are needed.

Keep selected passage IDs in the Reader's own in-memory state. They survive level changes and resizing, but reopening/reloading the Reader resets the collection. Say “Download to keep this brief” in the drawer. Do not extend host storage or rely on same-origin/clipboard permissions for persistence. Manual reading-position bookmarks remain position-only in this release.

Briefs do not yet include answers from subsequent chat messages, personal notes, PDF/DOCX export or cross-device synchronisation. Those are separate enhancements.

## Changes and delivery order

| Stage | Deliverable | OWUI features used |
| --- | --- | --- |
| 1 | Level-bar gestures, preserving the local word animation | Existing saved embed and normal iframe input handling |
| 2 | Passage collection and Markdown preview/download | Existing embed download permission; Function-owned in-memory selection |
| 3 | One-question compatibility spike, then complete question dialog and answer path | Ordinary copy/paste or existing embed-to-composer messaging, ordinary Send, saved message content/embeds, chat/file/model permissions, completion utility and source citations |
| 4 | Install the updated Function and regenerate a pilot Reader | Existing Function editor and retained Valves; no frontend deployment |

All runtime changes are in `functions/document_reader.py`, with focused tests and updates to the installation guide. Keep the Function installable as a single Python artifact and retain existing Valves. No OWUI application files, routes, database schema, provider configuration, parser or vector collection change is planned. Previously merged host improvements are neither extended nor required by these features.

The default copy/paste question flow works without the embed-to-composer hook; direct draft placement is an optional convenience when the host already supports it. Existing saved Reader responses retain their embedded code and need regeneration for the new controls and question reference.

## Acceptance checks

- Wheel and drag change levels only over the bar; ordinary document scrolling, browser zoom, text selection and OWUI shortcuts remain unchanged. No custom keyboard handler is added.
- Touch vertical scrolling, cancelled drags and wheel inertia do not accidentally cycle levels. Reading passage/offset, expanded state and reduced-motion behaviour survive every transition.
- Add/remove/export works at every level and from the source inspector. Selection survives resize; reopening resets it as described. Empty, partial and source-only exports are understandable; every quote exactly matches the frozen source, citations resolve and untrusted text remains inert.
- Zoom and brief selection/preview/download cause zero network or model calls after initial preparation.
- Ask produces an editable, visibly scoped copyable draft, preserves existing drafts unless the explicit replacement action is chosen, explains the Reader model requirement and never auto-submits. A blocked clipboard or missing composer hook leaves a working manual-copy path.
- A submitted question uses the saved target passage, preserves its qualifications, returns inspectable citations, saves normally and regenerates with the same context. It makes no Reader preparation calls and leaves the original embed intact.
- Forged/foreign references, revoked file/model access, malformed/duplicate envelopes, unknown citations, overly large context, missing snapshot and insufficient evidence receive useful failures. Provider truncation, refusal, cancellation and transport errors retain bounded handling.
- Verify the real source-citation rendering and ordinary message-content persistence with an ordinary permitted user, not only an administrator. Exercise two Readers in one chat, edited questions, removed envelopes, navigation, reload and regeneration to catch stale context. Confirm that implementation changes touch no OWUI application files.
- Re-run the existing backend, theme, resize, restoration and animation checks. Pilot with a business PDF and Markdown document; check DOCX compatibility too. Measure whether users can locate a condition, ask a useful question and assemble a brief without losing their place.
