"""Offline source, evidence, authorization and execution contracts for the Pipe.

Run with ``python -m unittest discover -s functions_tools/functions/tests
-p test_document_reader.py``. OWUI and model calls are replaced with isolated
module mocks; no server, credentials, network or paid generation is required.
"""

import asyncio
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from starlette.requests import Request
from starlette.responses import JSONResponse


ROOT = Path(__file__).parents[1]
CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/document_reader/corpus.json").read_text(
        encoding="utf-8"
    )
)
MODEL_ID = "permitted-text-model"
USER_ID = "reader-user"
FILE_ID = "11111111-1111-4111-8111-111111111111"
SECOND_FILE_ID = "22222222-2222-4222-8222-222222222222"


def load_reader():
    spec = importlib.util.spec_from_file_location(
        "document_reader_under_test", ROOT / "document_reader.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Dataclass/type-inspection helpers require their defining module to exist.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_text(snapshot):
    return "".join(
        unit["text"] for passage in snapshot["passages"] for unit in passage["units"]
    )


def batch_response(snapshot, batch):
    passages = {passage["id"]: passage for passage in snapshot["passages"]}
    rows = []
    all_evidence = []
    for passage_id in batch["passage_ids"]:
        passage = passages[passage_id]
        evidence = [unit["id"] for unit in passage["units"] if unit["text"].strip()]
        if not evidence:
            evidence = [passage["units"][0]["id"]]
        rows.append(
            {
                "id": passage_id,
                "extract_ids": evidence[:1],
                "explanation": [
                    {
                        "text": "The source describes this part of the proposal.",
                        "evidence": evidence[:1],
                    }
                ],
                "takeaways": [
                    {
                        "text": "Read the stated qualification before acting.",
                        "evidence": evidence[:1],
                    }
                ],
            }
        )
        all_evidence.extend(evidence[:1])
    return {
        "passages": rows,
        "overview": {
            "text": "This part records the proposal and its qualifications.",
            "evidence": all_evidence[:1],
        },
    }


def completion_response(value):
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = load_reader()

    def valves(self, **changes):
        return self.reader.Pipe.Valves(BASE_MODEL_ID=MODEL_ID).model_copy(
            update=changes
        )

    def snapshot(self, name="brief", **changes):
        fixture = CORPUS[name]
        return self.reader.build_snapshot(
            fixture["text"],
            fixture["filename"],
            FILE_ID,
            MODEL_ID,
            self.valves(**changes),
        )

    def assert_source_partition(self, snapshot, text):
        self.assertEqual(source_text(snapshot), text)
        position = 0
        ids = []
        for passage in snapshot["passages"]:
            for unit in passage["units"]:
                ids.append(unit["id"])
                self.assertEqual(unit["start"], position)
                self.assertEqual(unit["end"], position + len(unit["text"]))
                self.assertEqual(text[unit["start"] : unit["end"]], unit["text"])
                position = unit["end"]
        self.assertEqual(position, len(text))
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_source_character_survives_segmentation(self):
        for name, fixture in CORPUS.items():
            with self.subTest(name=name):
                snapshot = self.snapshot(name)
                self.assert_source_partition(snapshot, fixture["text"])
                self.assertTrue(snapshot["sections"])
                self.assertTrue(snapshot["passages"])
                for term in fixture["expected_terms"]:
                    self.assertIn(term, source_text(snapshot))

    def test_fingerprint_and_location_ids_are_stable_and_source_sensitive(self):
        first, second = self.snapshot(), self.snapshot()
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(
            [p["id"] for p in first["passages"]], [p["id"] for p in second["passages"]]
        )
        ids = [u["id"] for p in first["passages"] for u in p["units"]]
        self.assertEqual(len(ids), len(set(ids)))
        changed = self.reader.build_snapshot(
            CORPUS["brief"]["text"] + " ",
            "pilot-brief.md",
            FILE_ID,
            MODEL_ID,
            self.valves(),
        )
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])

    def test_segmentation_configuration_changes_the_bookmark_fingerprint(self):
        first = self.snapshot(MAX_BATCH_SOURCE_CHARS=1000)
        second = self.snapshot(MAX_BATCH_SOURCE_CHARS=2000)
        self.assertEqual(source_text(first), source_text(second))
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_short_pdf_paragraphs_coalesce_without_exhausting_passage_or_call_limits(
        self,
    ):
        text = "".join(
            f"Item {i}: budget approval remains pending.\n\n" for i in range(501)
        )
        valves = self.valves()
        snapshot = self.reader.build_snapshot(
            text, "paragraph-heavy.pdf", FILE_ID, MODEL_ID, valves
        )
        self.assert_source_partition(snapshot, text)
        self.assertLess(len(snapshot["passages"]), 30)
        batches = self.reader.make_batches(snapshot, valves)
        self.assertLessEqual(len(batches), valves.MAX_MODEL_CALLS)
        self.assertLess(len(batches), 10)
        for passage in snapshot["passages"]:
            self.assertLessEqual(sum(len(u["text"]) for u in passage["units"]), 2400)

    def test_short_paragraphs_near_source_limit_keep_bounded_generation(self):
        paragraph = "Delivery approval remains pending for item {index:04d}.\n\n"
        text = "".join(paragraph.format(index=i) for i in range(1960))
        valves = self.valves()
        self.assertGreater(len(text), valves.MAX_SOURCE_CHARS * 0.9)
        self.assertLessEqual(len(text), valves.MAX_SOURCE_CHARS)
        snapshot = self.reader.build_snapshot(
            text, "long-extracted.pdf", FILE_ID, MODEL_ID, valves
        )
        self.assert_source_partition(snapshot, text)
        self.assertLess(len(snapshot["passages"]), 60)
        batches = self.reader.make_batches(snapshot, valves)
        self.assertLessEqual(len(batches), valves.MAX_MODEL_CALLS)
        lookup = {p["id"]: p for p in snapshot["passages"]}
        for batch in batches:
            size = sum(
                len(unit["text"])
                for pid in batch["passage_ids"]
                for unit in lookup[pid]["units"]
            )
            self.assertLessEqual(size, valves.MAX_BATCH_SOURCE_CHARS)
            self.assertLessEqual(len(batch["passage_ids"]), valves.MAX_BATCH_PASSAGES)

    def test_grouping_flushes_before_headings_and_keeps_tables_source_only(self):
        introduction = (
            "Earlier context remains provisional.\n\nApproval is pending.\n\n"
        )
        before_table = (
            "Two teams can participate.\n\nTheir access review must finish.\n\n"
        )
        table = "| Team | Budget |\n| --- | --- |\n| Alpha | GBP 12000 |\n\n"
        after_table = "The figures are proposals.\n\nSpend still requires approval.\n\n"
        delivery = "# Delivery\n\n" + before_table + table + after_table
        review = "# Review\n\nReview every claim.\n\nRetain the original wording.\n"
        text = introduction + delivery + review
        valves = self.valves()
        snapshot = self.reader.build_snapshot(
            text, "boundaries.md", FILE_ID, MODEL_ID, valves
        )
        self.assert_source_partition(snapshot, text)
        self.assertEqual(
            [s["title"] for s in snapshot["sections"]],
            ["Part 1", "Delivery", "Review"],
        )
        lookup = {p["id"]: p for p in snapshot["passages"]}
        for section, expected_text in zip(
            snapshot["sections"], (introduction, delivery, review)
        ):
            self.assertEqual(
                "".join(
                    u["text"]
                    for pid in section["passage_ids"]
                    for u in lookup[pid]["units"]
                ),
                expected_text,
            )
        eligible_texts = [
            "".join(u["text"] for u in p["units"])
            for p in snapshot["passages"]
            if not p["source_only"]
        ]
        self.assertIn(before_table, eligible_texts)
        self.assertIn(after_table, eligible_texts)
        table_passages = [
            p
            for p in snapshot["passages"]
            if "GBP 12000" in "".join(u["text"] for u in p["units"])
        ]
        self.assertEqual(len(table_passages), 1)
        self.assertTrue(table_passages[0]["source_only"])
        self.assertEqual("".join(u["text"] for u in table_passages[0]["units"]), table)
        sent_ids = {
            pid
            for batch in self.reader.make_batches(snapshot, valves)
            for pid in batch["passage_ids"]
        }
        self.assertNotIn(table_passages[0]["id"], sent_ids)

    def test_grouped_unicode_crlf_paragraphs_keep_stable_ids_and_offsets(self):
        text = CORPUS["unicode_crlf"]["text"] * 20
        valves = self.valves(MAX_BATCH_SOURCE_CHARS=1000)
        first = self.reader.build_snapshot(
            text, "unicode.pdf", FILE_ID, MODEL_ID, valves
        )
        second = self.reader.build_snapshot(
            text, "unicode.pdf", FILE_ID, MODEL_ID, valves
        )
        self.assert_source_partition(first, text)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["passages"], second["passages"])
        self.assertEqual(first["sections"], second["sections"])
        self.assertLess(len(first["passages"]), 12)
        for passage in first["passages"]:
            self.assertLessEqual(sum(len(u["text"]) for u in passage["units"]), 1000)

    def test_table_values_are_preserved_but_never_sent_for_generation(self):
        snapshot = self.snapshot("table")
        table_passages = [
            p
            for p in snapshot["passages"]
            if "£12,000" in "".join(u["text"] for u in p["units"])
        ]
        self.assertTrue(table_passages)
        self.assertTrue(all(p["source_only"] for p in table_passages))
        sent_ids = {
            pid
            for b in self.reader.make_batches(snapshot, self.valves())
            for pid in b["passage_ids"]
        }
        self.assertTrue(all(p["id"] not in sent_ids for p in table_passages))

    def test_widely_spaced_pdf_prose_is_eligible_and_preserves_source(self):
        text = (
            "The   pilot   supports   two   teams   while   funding   approval   remains   pending.\r\n"
            "Each   team   must   keep   the   original   evidence   and   review   material   claims.\r\n\r\n"
            "Short   continuation\r\n"
            "lines      can   have   irregular   gaps   without   creating   any   table   columns.\r\n"
        )
        self.assertFalse(self.reader._table_like(text))
        snapshot = self.reader.build_snapshot(
            text, "spaced-prose.pdf", FILE_ID, MODEL_ID, self.valves()
        )
        self.assert_source_partition(snapshot, text)
        self.assertTrue(all(not p["source_only"] for p in snapshot["passages"]))
        self.assertTrue(self.reader.make_batches(snapshot, self.valves()))

    def test_explicit_and_aligned_tables_remain_isolated_from_generation(self):
        tables = (
            "| Team | Budget |\n| --- | --- |\n| Research | GBP 12000 |\n",
            "Team\tBudget\nResearch \t GBP 12000\nOperations\tGBP 8000\n",
            "Team         Budget\nResearch     GBP 12000\nOperations   GBP 8000\n",
        )
        for table in tables:
            with self.subTest(table_format=repr(table[:25])):
                self.assertTrue(self.reader._table_like(table))
                snapshot = self.reader.build_snapshot(
                    table, "table.pdf", FILE_ID, MODEL_ID, self.valves()
                )
                self.assert_source_partition(snapshot, table)
                self.assertTrue(all(p["source_only"] for p in snapshot["passages"]))
                self.assertTrue(
                    all(s["heading_kind"] == "fallback" for s in snapshot["sections"])
                )
                self.assertEqual(self.reader.make_batches(snapshot, self.valves()), [])

    def test_plain_headings_accept_titles_but_not_references_bullets_or_prose(self):
        for title in (
            "Delivery Readiness",
            "Terms   and   Conditions",
            "Review of the Pilot",
            "BUDGET REVIEW",
        ):
            with self.subTest(title=title):
                self.assertEqual(self.reader._heading(title), " ".join(title.split()))
        for prose in (
            "AB.CD-3, EF.GH-7)",
            "CONTROL-123",
            "The team must review the proposal",
            "Review the proposal before approval.",
            "Review The Proposal.",
            "- Delivery Readiness",
            "• DELIVERY READINESS",
            "Delivery Readiness\nReview Schedule",
            "Planning Review Delivery Control Scope Schedule Budget Approval Conditions",
        ):
            with self.subTest(prose=prose):
                self.assertIsNone(self.reader._heading(prose))

    def test_pdf_titles_create_sections_and_reference_codes_stay_in_prose(self):
        text = (
            "Delivery   Readiness\r\n"
            "The   pilot   remains   provisional   until   a   reviewer   confirms   approval.\r\n"
            "(AB.CD-3,   EF.GH-7)\r\n\r\n"
            "Review of the Pilot\r\n"
            "The   review   covers   evidence   and   does   not   approve   spending.\r\n"
        )
        snapshot = self.reader.build_snapshot(
            text, "titles.pdf", FILE_ID, MODEL_ID, self.valves()
        )
        self.assert_source_partition(snapshot, text)
        self.assertEqual(
            [s["title"] for s in snapshot["sections"]],
            ["Delivery Readiness", "Review of the Pilot"],
        )
        code_passage = next(
            p
            for p in snapshot["passages"]
            if any("AB.CD-3" in u["text"] for u in p["units"])
        )
        self.assertFalse(code_passage["source_only"])

    def test_batches_cover_each_eligible_passage_once_within_one_section(self):
        snapshot = self.snapshot()
        valves = self.valves(MAX_BATCH_PASSAGES=1)
        batches = self.reader.make_batches(snapshot, valves)
        expected = {p["id"] for p in snapshot["passages"] if not p["source_only"]}
        actual = [pid for batch in batches for pid in batch["passage_ids"]]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        lookup = {p["id"]: p for p in snapshot["passages"]}
        for batch in batches:
            self.assertLessEqual(len(batch["passage_ids"]), valves.MAX_BATCH_PASSAGES)
            self.assertTrue(
                all(
                    lookup[pid]["section_id"] == batch["section_id"]
                    for pid in batch["passage_ids"]
                )
            )

    def test_size_and_structure_limits_reject_without_truncation(self):
        with self.assertRaises((ValueError, self.reader.ReaderError)):
            self.snapshot(MAX_SOURCE_CHARS=10)
        with self.assertRaises((ValueError, self.reader.ReaderError)):
            self.snapshot(MAX_PASSAGES=1)

    def test_indivisible_source_over_batch_limit_is_rejected(self):
        valves = self.valves(MAX_BATCH_SOURCE_CHARS=100)
        with self.assertRaises((ValueError, self.reader.ReaderError)):
            snapshot = self.reader.build_snapshot(
                "A" * 300, "long.md", FILE_ID, MODEL_ID, valves
            )
            self.reader.make_batches(snapshot, valves)

    def test_minimum_batch_count_cannot_exceed_call_budget(self):
        valves = self.valves(MAX_BATCH_PASSAGES=1, MAX_MODEL_CALLS=1)
        with self.assertRaises((ValueError, self.reader.ReaderError)):
            snapshot = self.snapshot()
            self.reader.make_batches(snapshot, valves)


class ResultValidationTests(unittest.TestCase):
    valves = SourceContractTests.valves
    snapshot = SourceContractTests.snapshot

    @classmethod
    def setUpClass(cls):
        cls.reader = load_reader()

    def setUp(self):
        self.document = self.snapshot()
        self.batch = self.reader.make_batches(self.document, self.valves())[0]
        self.valid = batch_response(self.document, self.batch)

    def validate(self, value):
        raw = value if isinstance(value, str) else json.dumps(value)
        return self.reader.validate_result(raw, self.batch, self.document)

    def assert_invalid(self, value):
        with self.assertRaises((ValueError, self.reader.ReaderError)):
            self.validate(value)

    def test_valid_response_has_known_evidence_and_preserves_source(self):
        before = copy.deepcopy(self.document)
        result = self.validate(self.valid)
        self.assertEqual(
            {p["id"] for p in result["passages"]}, set(self.batch["passage_ids"])
        )
        self.assertEqual(self.document, before)

    def test_invalid_json_or_unknown_fields_cannot_be_accepted(self):
        self.assert_invalid("This is not JSON")
        invalid = copy.deepcopy(self.valid)
        invalid["unexpected"] = "ignore the schema"
        self.assert_invalid(invalid)

    def test_duplicate_json_keys_are_rejected_instead_of_silently_overwritten(self):
        raw = json.dumps(self.valid)
        duplicated = (
            raw[:-1] + ', "overview": ' + json.dumps(self.valid["overview"]) + "}"
        )
        self.assert_invalid(duplicated)

    def test_missing_duplicate_or_unknown_passages_are_rejected(self):
        for kind in ("missing", "duplicate", "unknown"):
            value = copy.deepcopy(self.valid)
            if kind == "missing":
                value["passages"].pop()
            elif kind == "duplicate":
                value["passages"].append(copy.deepcopy(value["passages"][0]))
            else:
                value["passages"][0]["id"] = "invented-passage"
            with self.subTest(kind=kind):
                self.assert_invalid(value)

    def test_unknown_duplicate_or_absent_extract_ids_are_rejected(self):
        for ids in (
            ["invented-evidence"],
            [],
            self.valid["passages"][0]["extract_ids"] * 2,
        ):
            value = copy.deepcopy(self.valid)
            value["passages"][0]["extract_ids"] = ids
            with self.subTest(ids=ids):
                self.assert_invalid(value)

    def test_all_generated_items_and_overviews_require_valid_evidence(self):
        for field in ("explanation", "takeaways", "overview"):
            for evidence in ([], ["invented-evidence"]):
                value = copy.deepcopy(self.valid)
                item = (
                    value["overview"]
                    if field == "overview"
                    else value["passages"][0][field][0]
                )
                item["evidence"] = evidence
                with self.subTest(field=field, evidence=evidence):
                    self.assert_invalid(value)

    def test_evidence_outside_its_passage_is_rejected(self):
        target = self.valid["passages"][0]["id"]
        foreign = next(
            p for p in self.document["passages"] if p["id"] != target and p["units"]
        )
        value = copy.deepcopy(self.valid)
        value["passages"][0]["explanation"][0]["evidence"] = [foreign["units"][0]["id"]]
        self.assert_invalid(value)

    def test_model_cannot_supply_rewritten_source_text(self):
        value = copy.deepcopy(self.valid)
        value["passages"][0]["extract_text"] = "A paraphrase presented as a quote."
        self.assert_invalid(value)


class PipeExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        names = (
            "open_webui",
            "open_webui.env",
            "open_webui.models",
            "open_webui.models.files",
            "open_webui.models.users",
            "open_webui.models.chats",
            "open_webui.models.models",
            "open_webui.utils",
            "open_webui.utils.models",
            "open_webui.utils.chat",
            "open_webui.utils.access_control",
            "open_webui.utils.access_control.files",
        )
        self.modules = {}
        for name in names:
            module = ModuleType(name)
            module.__path__ = []
            self.modules[name] = module
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.user = NS(
            id=USER_ID, role="user", model_dump=lambda: {"id": USER_ID, "role": "user"}
        )
        self.users = AsyncMock(return_value=self.user)
        self.files = AsyncMock(return_value=self.file_record())
        self.access = AsyncMock(return_value=False)
        self.model_access = AsyncMock()
        self.complete = AsyncMock()
        self.chat_owner = AsyncMock(return_value=True)
        self.chat_message = AsyncMock(return_value=None)
        self.modules["open_webui.models.users"].Users = NS(get_user_by_id=self.users)
        self.modules["open_webui.models.files"].Files = NS(get_file_by_id=self.files)
        self.modules["open_webui.models.chats"].Chats = NS(
            is_chat_owner=self.chat_owner,
            get_message_by_id_and_message_id=self.chat_message,
        )
        self.modules["open_webui.models.models"].Models = NS(
            get_model_by_id=AsyncMock(return_value=None)
        )
        self.modules["open_webui.utils.access_control.files"].has_access_to_file = (
            self.access
        )
        self.modules["open_webui.utils.models"].check_model_access = self.model_access
        self.modules["open_webui.utils.models"].get_all_models = AsyncMock()
        self.modules["open_webui.utils.chat"].generate_chat_completion = self.complete
        self.modules["open_webui.env"].BYPASS_ADMIN_ACCESS_CONTROL = True
        self.modules["open_webui.env"].BYPASS_MODEL_ACCESS_CONTROL = False
        self.reader = load_reader()
        self.pipe = self.reader.Pipe()
        self.pipe.valves.BASE_MODEL_ID = MODEL_ID
        self.emitter = AsyncMock()
        self.rendered = []
        self.render_patch = patch.object(
            self.reader, "render_reader", side_effect=self.capture_render
        )
        self.render_patch.start()
        self.addCleanup(self.render_patch.stop)
        self.models = {
            MODEL_ID: {
                "id": MODEL_ID,
                "owned_by": "openai",
                "connection_type": "external",
                "info": {"params": {}, "meta": {}},
            }
        }
        self.token = NS(credentials="secret-token-never-in-embed")
        self.outer_state = {
            "metadata": {
                "chat_id": "outer-chat",
                "message_id": "outer-message",
                "files": ["private-file"],
            },
            "token": self.token,
            "direct": True,
            "bypass_filter": True,
            "internal": True,
        }
        self.request = Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "http",
                "path": "/api/chat/completions",
                "root_path": "",
                "query_string": b"",
                "headers": [],
                "server": ("localhost", 8080),
                "client": ("127.0.0.1", 1234),
                "app": NS(state=NS(MODELS=self.models)),
                "state": self.outer_state,
            }
        )

    def file_record(self, **changes):
        return NS(
            **{
                "id": FILE_ID,
                "user_id": USER_ID,
                "filename": CORPUS["brief"]["filename"],
                "meta": {"content_type": "text/markdown", "size": 1024},
                "data": {"content": CORPUS["brief"]["text"], "status": "completed"},
                **changes,
            }
        )

    def capture_render(self, snapshot):
        self.rendered.append(copy.deepcopy(snapshot))
        return "<!doctype html><title>Offline reader</title>"

    def metadata(self):
        return {
            "chat_id": "saved-reader-chat",
            "message_id": "assistant-message",
            "user_message_id": "user-message",
            "user_message": {
                "id": "user-message",
                "role": "user",
                "files": [{"type": "file", "id": FILE_ID}],
            },
        }

    def expected(self):
        source = self.files.return_value
        snapshot = self.reader.build_snapshot(
            source.data["content"],
            source.filename,
            FILE_ID,
            self.pipe.valves.BASE_MODEL_ID,
            self.pipe.valves,
        )
        return snapshot, self.reader.make_batches(snapshot, self.pipe.valves)

    def successful_calls(self):
        snapshot, batches = self.expected()
        self.complete.side_effect = [
            completion_response(batch_response(snapshot, b)) for b in batches
        ]
        return batches

    async def run_pipe(self, **changes):
        kwargs = {
            "body": {
                "model": "document_reader",
                "messages": [{"role": "user", "content": "Prepare this document"}],
            },
            "__user__": {"id": USER_ID, "role": "user"},
            "__request__": self.request,
            "__event_emitter__": self.emitter,
            "__files__": [{"type": "file", "id": FILE_ID}],
            "__metadata__": self.metadata(),
        }
        kwargs.update(changes)
        return await self.pipe.pipe(**kwargs)

    def embed_events(self):
        return [
            call.args[0]
            for call in self.emitter.await_args_list
            if call.args[0].get("type") == "embeds"
        ]

    async def test_success_emits_one_final_snapshot_and_isolates_every_inner_request(
        self,
    ):
        batches = self.successful_calls()
        before = copy.deepcopy(self.outer_state["metadata"])
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, len(batches))
        self.assertEqual(len(self.embed_events()), 1)
        self.assertTrue(self.embed_events()[0]["data"]["replace"])
        self.assertEqual(source_text(self.rendered[-1]), CORPUS["brief"]["text"])
        self.assertEqual(self.rendered[-1]["status"], "complete")
        self.assertEqual(self.outer_state["metadata"], before)
        self.assertTrue(self.outer_state["direct"])
        seen_requests = []
        for call in self.complete.await_args_list:
            inner = call.kwargs.get("request", call.args[0] if call.args else None)
            payload = call.kwargs.get(
                "form_data", call.args[1] if len(call.args) > 1 else None
            )
            real_user = call.kwargs.get(
                "user", call.args[2] if len(call.args) > 2 else None
            )
            self.assertIsNot(inner, self.request)
            self.assertIsNot(inner.scope["state"], self.outer_state)
            self.assertIs(inner.state.token, self.token)
            self.assertFalse(getattr(inner.state, "direct", False))
            self.assertFalse(getattr(inner.state, "bypass_filter", False))
            self.assertFalse(getattr(inner.state, "metadata", {}))
            self.assertIs(real_user, self.user)
            self.assertEqual(payload["model"], MODEL_ID)
            self.assertIs(payload["stream"], False)
            self.assertFalse(
                set(payload)
                & {
                    "chat_id",
                    "id",
                    "session_id",
                    "files",
                    "tools",
                    "tool_ids",
                    "skill_ids",
                }
            )
            self.assertNotIn("outer-message", json.dumps(payload))
            self.assertNotIn(self.token.credentials, json.dumps(payload))
            seen_requests.append(id(inner))
        self.assertEqual(len(seen_requests), len(set(seen_requests)))
        self.model_access.assert_awaited()

    async def test_inner_completion_state_mutations_cannot_change_outer_chat_state(
        self,
    ):
        snapshot, batches = self.expected()
        responses = iter(
            completion_response(batch_response(snapshot, batch)) for batch in batches
        )
        before = copy.deepcopy(self.outer_state["metadata"])

        async def mutating_model(request, form_data, user):
            # Match mutations performed by OWUI's internal completion utility.
            request.state.metadata = {
                "chat_id": "inner-only",
                "message_id": "inner-only",
            }
            request.state.bypass_filter = False
            request.state.direct = False
            form_data["metadata"] = request.state.metadata
            return next(responses)

        self.complete.side_effect = mutating_model
        await self.run_pipe()
        self.assertEqual(self.outer_state["metadata"], before)
        self.assertIs(self.outer_state["bypass_filter"], True)
        self.assertIs(self.outer_state["direct"], True)
        self.assertEqual(self.rendered[-1]["status"], "complete")

    async def test_progress_event_failure_does_not_discard_the_completed_reader(self):
        self.successful_calls()

        async def disconnected_status(event):
            if event["type"] == "status":
                raise ConnectionError("status delivery failed")

        self.emitter.side_effect = disconnected_status
        await self.run_pipe()
        self.assertEqual(len(self.embed_events()), 1)
        self.assertEqual(self.rendered[-1]["status"], "complete")

    async def test_auxiliary_tasks_never_read_files_or_generate(self):
        for task in ("title_generation", "tags_generation", "follow_up_generation"):
            with self.subTest(task=task):
                self.assertEqual(await self.run_pipe(__task__=task), "")
        self.files.assert_not_awaited()
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])

    async def test_inherited_attachment_alone_is_not_selected(self):
        metadata = self.metadata()
        metadata["user_message"]["files"] = []
        await self.run_pipe(__metadata__=metadata)
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])

    async def test_multiple_current_attachments_are_rejected(self):
        metadata = self.metadata()
        metadata["user_message"]["files"].append({"type": "file", "id": SECOND_FILE_ID})
        await self.run_pipe(__metadata__=metadata)
        self.complete.assert_not_awaited()

    async def test_collection_and_url_are_not_treated_as_individual_files(self):
        for item in (
            {"type": "collection", "id": FILE_ID},
            {"type": "folder", "id": FILE_ID},
            {"type": "image", "id": FILE_ID},
            {"type": "url", "url": "https://example.test/document.pdf"},
        ):
            metadata = self.metadata()
            metadata["user_message"]["files"] = [item]
            with self.subTest(item=item):
                await self.run_pipe(__metadata__=metadata)
        self.complete.assert_not_awaited()

    async def test_fresh_source_is_used_instead_of_client_supplied_text(self):
        self.successful_calls()
        metadata = self.metadata()
        metadata["user_message"]["files"] = [
            {
                "type": "file",
                "file": {"id": FILE_ID, "data": {"content": "FORGED TEXT"}},
            }
        ]
        await self.run_pipe(__metadata__=metadata)
        self.assertEqual(source_text(self.rendered[-1]), CORPUS["brief"]["text"])
        self.assertNotIn("FORGED TEXT", source_text(self.rendered[-1]))

    async def test_unauthorized_file_stops_before_model_call(self):
        self.files.return_value = self.file_record(user_id="another-user")
        await self.run_pipe()
        self.access.assert_awaited()
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])

    async def test_permitted_shared_file_can_generate(self):
        self.files.return_value = self.file_record(user_id="another-user")
        self.access.return_value = True
        self.successful_calls()
        await self.run_pipe()
        self.assertEqual(len(self.embed_events()), 1)

    async def test_unauthorized_chat_stops_before_model_call(self):
        self.chat_owner.return_value = False
        await self.run_pipe()
        self.complete.assert_not_awaited()

    async def test_missing_user_temporary_chat_or_embed_context_never_generates(self):
        self.users.return_value = None
        await self.run_pipe()
        self.users.return_value = self.user
        metadata = self.metadata()
        metadata["chat_id"] = "local:temporary-chat"
        # Unsaved temporary IDs have no owned Chat row in the real OWUI model.
        self.chat_owner.side_effect = (
            lambda chat_id, user_id: chat_id != "local:temporary-chat"
        )
        await self.run_pipe(__metadata__=metadata)
        await self.run_pipe(__event_emitter__=None)
        self.complete.assert_not_awaited()

    async def test_unavailable_or_unpermitted_model_stops_before_generation(self):
        self.model_access.side_effect = PermissionError("not permitted")
        await self.run_pipe()
        self.complete.assert_not_awaited()
        self.model_access.side_effect = None
        self.models.clear()
        await self.run_pipe()
        self.complete.assert_not_awaited()

    async def test_pipe_arena_direct_and_pipeline_targets_are_rejected(self):
        for forbidden in (
            {"pipe": {"type": "pipe"}},
            {"owned_by": "arena"},
            {"direct": True},
            {"connection_type": "external", "direct": True},
            {"pipeline": {"type": "pipe"}},
        ):
            self.models[MODEL_ID] = {"id": MODEL_ID, "owned_by": "openai", **forbidden}
            with self.subTest(forbidden=forbidden):
                await self.run_pipe()
        self.complete.assert_not_awaited()

    async def test_external_server_model_with_provider_prefix_delegates(self):
        model_id = "chatgpt/gpt-5.6-sol"
        self.pipe.valves.BASE_MODEL_ID = model_id
        self.models[model_id] = {
            "id": model_id,
            "owned_by": "openai",
            "connection_type": "external",
            "urlIdx": 0,
            "openai": {"id": model_id},
        }
        batches = self.successful_calls()
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, len(batches))
        self.assertEqual(self.rendered[-1]["status"], "complete")
        self.assertEqual(self.rendered[-1]["model_id"], model_id)
        for call in self.complete.await_args_list:
            self.assertEqual(call.args[1]["model"], model_id)
        self.model_access.assert_awaited_once_with(self.user, self.models[model_id])

    async def test_alias_of_external_server_model_delegates_alias_id(self):
        base_id = "chatgpt/gpt-5.6-sol"
        self.models[MODEL_ID] = {
            "id": MODEL_ID,
            "owned_by": "openai",
            "connection_type": "external",
            "preset": True,
            "info": {"base_model_id": base_id},
        }
        self.models[base_id] = {
            "id": base_id,
            "owned_by": "openai",
            "connection_type": "external",
        }
        batches = self.successful_calls()
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, len(batches))
        self.assertEqual(self.rendered[-1]["status"], "complete")
        for call in self.complete.await_args_list:
            self.assertEqual(call.args[1]["model"], MODEL_ID)
        self.model_access.assert_awaited_once_with(self.user, self.models[MODEL_ID])

    async def test_falsy_optional_pipe_and_pipeline_markers_allow_server_model(self):
        # OWUI's internal dispatcher and OpenAI adapter check marker truthiness.
        for marker in (None, False, {}):
            self.models[MODEL_ID].update(
                {"pipe": marker, "pipeline": marker, "direct": False}
            )
            batches = self.successful_calls()
            before = self.complete.await_count
            with self.subTest(marker=marker):
                await self.run_pipe()
                self.assertEqual(self.complete.await_count - before, len(batches))
                self.assertEqual(self.rendered[-1]["status"], "complete")

    async def test_alias_of_pipe_or_browser_direct_model_is_also_rejected(self):
        self.models[MODEL_ID] = {
            "id": MODEL_ID,
            "owned_by": "openai",
            "connection_type": "external",
            "info": {"base_model_id": "reader-alias"},
        }
        for marker in ({"pipe": {"type": "pipe"}}, {"direct": True}):
            self.models["reader-alias"] = {"id": "reader-alias", **marker}
            with self.subTest(marker=marker):
                await self.run_pipe()
        self.complete.assert_not_awaited()

    async def test_failed_extraction_without_text_never_generates(self):
        self.files.return_value = self.file_record(
            data={"content": "", "status": "failed", "error": "Extraction failed"}
        )
        await self.run_pipe()
        self.complete.assert_not_awaited()

    async def test_source_limit_is_enforced_before_any_model_call(self):
        self.pipe.valves.MAX_SOURCE_CHARS = 100
        await self.run_pipe()
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])

    async def test_pending_source_waits_for_owui_and_returns_fresh_text(self):
        self.files.side_effect = [
            self.file_record(data={"content": "", "status": "pending"}),
            self.file_record(),
        ]
        with patch.object(self.reader.asyncio, "sleep", new=AsyncMock()) as sleep:
            _, text, _ = await self.pipe._load_source(
                FILE_ID, self.user, self.pipe.valves, self.emitter
            )
        self.assertEqual(text, CORPUS["brief"]["text"])
        self.assertEqual(self.files.await_count, 2)
        sleep.assert_awaited_once()

    async def test_pending_source_has_a_bounded_wait(self):
        self.files.return_value = self.file_record(
            data={"content": "", "status": "pending"}
        )
        self.pipe.valves.FILE_READY_TIMEOUT_SECONDS = 0
        await self.run_pipe()
        self.complete.assert_not_awaited()
        self.assertEqual(self.files.await_count, 1)

    async def test_usable_text_survives_indexing_failure_and_legacy_missing_status(
        self,
    ):
        for data in (
            {
                "content": CORPUS["brief"]["text"],
                "status": "failed",
                "error": "Embedding unavailable",
            },
            {"content": CORPUS["brief"]["text"]},
        ):
            self.files.return_value = self.file_record(data=data)
            self.successful_calls()
            with self.subTest(data=data):
                await self.run_pipe()
                self.assertEqual(
                    source_text(self.rendered[-1]), CORPUS["brief"]["text"]
                )

    async def test_invalid_output_gets_one_repair_then_preserves_partial_result(self):
        self.pipe.valves.MAX_BATCH_PASSAGES = 1
        snapshot, batches = self.expected()
        self.assertGreater(len(batches), 1)
        replies = [
            completion_response(batch_response(snapshot, batches[0])),
            completion_response("invalid"),
            completion_response("still invalid"),
        ]
        replies.extend(
            completion_response(batch_response(snapshot, b)) for b in batches[2:]
        )
        self.complete.side_effect = replies
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, len(batches) + 1)
        self.assertEqual(len(self.embed_events()), 1)
        final = self.rendered[-1]
        self.assertEqual(final["status"], "partial")
        self.assertTrue(any(p["generated"] for p in final["passages"]))
        self.assertTrue(
            any(not p["generated"] for p in final["passages"] if not p["source_only"])
        )
        self.assertEqual(source_text(final), CORPUS["brief"]["text"])

    async def test_transport_failure_is_not_automatically_retried(self):
        snapshot, batches = self.expected()
        self.complete.side_effect = [TimeoutError("upstream timeout")] + [
            completion_response(batch_response(snapshot, b)) for b in batches[1:]
        ]
        await self.run_pipe()
        self.assertLessEqual(self.complete.await_count, len(batches))
        self.assertEqual(self.rendered[-1]["status"], "partial")

    async def test_provider_error_response_does_not_become_a_repair_prompt(self):
        _, batches = self.expected()
        self.complete.return_value = JSONResponse(
            {"error": "provider unavailable"}, status_code=503
        )
        await self.run_pipe()
        self.assertLessEqual(self.complete.await_count, len(batches))
        self.assertEqual(self.rendered[-1]["status"], "partial")

    async def test_call_budget_includes_repair_calls(self):
        self.pipe.valves.MAX_BATCH_PASSAGES = 1
        _, batches = self.expected()
        self.pipe.valves.MAX_MODEL_CALLS = len(batches)
        self.complete.return_value = completion_response("invalid")
        await self.run_pipe()
        self.assertLessEqual(self.complete.await_count, len(batches))
        self.assertEqual(self.rendered[-1]["status"], "partial")

    async def test_cancellation_propagates_and_no_new_calls_start(self):
        self.complete.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.run_pipe()
        self.assertEqual(self.complete.await_count, 1)

    async def test_overall_deadline_stops_new_calls_and_saves_partial_source(self):
        self.pipe.valves.RUN_TIMEOUT_SECONDS = 0.02

        async def slow_model(*args, **kwargs):
            await asyncio.sleep(1)
            return completion_response("too late")

        self.complete.side_effect = slow_model
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, 1)
        self.assertEqual(self.rendered[-1]["status"], "partial")
        self.assertEqual(source_text(self.rendered[-1]), CORPUS["brief"]["text"])

    async def test_rerun_is_explicit_new_generation_not_global_cached_output(self):
        batches = self.successful_calls()
        await self.run_pipe()
        self.successful_calls()
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, len(batches) * 2)
        self.assertEqual(len(self.embed_events()), 2)


if __name__ == "__main__":
    unittest.main()
