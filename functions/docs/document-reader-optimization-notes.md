# Document Reader 0.4.0

Reviewed the local PaperFold checkout's result caching (`adr/llm.py`), concurrent section preparation (`adr/ladder.py`), operation-specific reasoning (`adr/models.py`), compact output affordances and shared animation groups (`web/reader.js`). The implementation uses those ideas within the standalone OWUI Pipe. It does not run CLI agents, extend OWUI routes or introduce another extraction service.

## Delivered

- Explicit retry of missing work from an authorised saved Reader; reconstruct current source and revalidate cached results before use. Cache compatibility includes batching, model catalogue/alias configuration, generation options and prompt/schema versions. No shared cache or disk service.
- Two-worker maximum with one shared call budget and deadline, deterministic document order, bounded validation repairs and cancellation of all workers. Default remains one worker.
- Opt-in strict provider JSON Schema plus separate preparation/question reasoning settings. Provider defaults remain unchanged; unsupported options never trigger silent transport retries.
- Group words with matching transforms, and fade entering/leaving words in passage groups. Keep exact source DOM, reading anchors, reduced motion and cancellation behavior.
- Word counts for generated/extracted levels against source length. No summary is skipped or fabricated merely because its source is short.
- Accept the canonical repeated-footer and OCR-placeholder labels when opening saved-PDF questions, while excluding that furniture from answer context.

## Local validation — 7 October 2026

137 focused tests pass: source partition and evidence, access control, retry compatibility and malformed references, invalid cached citations, zero-call reuse, provider payloads, concurrent completion order, shared budgets and cancellation, streamed failures, opaque iframe controls, reading briefs, themes, responsive resize and semantic zoom.

Synthetic 24-passage Chrome benchmark, Explanation → Full text, passage 12 anchored near the viewport top; seven samples per width, first sample discarded:

| Width | Animations before → after | Visible ghost words | Median synchronous transition setup before → after |
| --- | --- | --- | --- |
| 1120 px | 189 → 34 | 189 | 7.10 → 5.70 ms |
| 390 px | 111 → 21 | 111 | 5.70 → 4.75 ms |

This measures headless Chrome setup and animation counts on this machine, not GPU frame pacing or production end-to-end latency. Idle geometry precomputation was deferred: measured setup was already short, and avoiding a geometry cache removes invalidation work for resizing, fonts, expanded passages and source inspection. Browser regression checks require substantially fewer animations than ghost words and preserve the existing position/evidence guarantees.

No new live PDF extraction claim follows from these checks. Retry requires a saved 0.4.0 snapshot; a request interrupted before any embed is saved has no recovery checkpoint. The supported model must be tested separately before enabling optional schema/effort settings or two-worker concurrency on a deployment.

## Live check — 7 October 2026

Installed through the existing Function editor, retaining `chatgpt/gpt-5.6-sol` and streaming. Compared persisted Python syntax to the local tested artifact; only the final newline differed. Regenerated the existing [synthetic pilot](https://owui.theoldschool.house/c/a28fb5bf-d712-40f7-b843-538996ffded1): all three batches completed in three calls. Sent an explicit retry reference through the ordinary composer to the complete saved Reader: all three batches were reused with zero new model calls. Both saved Readers had the same source SHA-256. Checked the new reading-length metadata and visible preparation counts. Partial recovery and concurrency remain local regression checks; the live compatibility defaults were retained.
