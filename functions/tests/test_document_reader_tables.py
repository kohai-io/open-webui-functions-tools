"""Structured uploads: original coordinates, bounded parsing and saved provenance."""

import copy
import base64
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch
from zipfile import ZipFile

import pytest
import test_document_reader as base


def xlsx_bytes(sheet_xml=None, merged="", second=True):
    """Minimal OOXML byte fixtures, including deliberately malformed inputs.

    These test the import format directly (not an authored business workbook).
    Formula caches must remain exactly as encoded, never recalculated by a writer.
    """
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    data = (
        sheet_xml
        or """<cols><col min="2" max="2" hidden="1"/></cols><sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>ID</t></is></c><c r="B1" t="inlineStr"><is><t>Value</t></is></c><c r="C1" t="inlineStr"><is><t>Due</t></is></c></row>
    <row r="2" hidden="1"><c r="A2" t="inlineStr"><is><t>00123</t></is></c><c r="B2"><v>0</v></c><c r="C2" s="1"><v>45292</v></c></row>
    <row r="3"><c r="A3"><f>SUM(B2:B2)</f><v>0</v></c><c r="B3"><f>1+1</f></c><c r="C3" t="b"><v>1</v></c></row>
    <row r="4"><c r="A4" t="e"><v>#N/A</v></c><c r="B4" t="inlineStr"><is><t></t></is></c></row>
    </sheetData>"""
    )
    out = io.BytesIO()
    with ZipFile(out, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/></Types>',
        )
        z.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        z.writestr(
            "xl/workbook.xml",
            f'<workbook {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Risks" sheetId="1" r:id="rId1"/>'
            + (
                '<sheet name="Archive" sheetId="2" state="hidden" r:id="rId2"/>'
                if second
                else ""
            )
            + "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            + (
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
                if second
                else ""
            )
            + "</Relationships>",
        )
        z.writestr(
            "xl/styles.xml",
            f'<styleSheet {ns}><fonts count="1"><font/></fonts><fills count="1"><fill><patternFill/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
        )
        z.writestr(
            "xl/worksheets/sheet1.xml", f"<worksheet {ns}>{data}{merged}</worksheet>"
        )
        if second:
            z.writestr(
                "xl/worksheets/sheet2.xml",
                f'<worksheet {ns}><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Archived</t></is></c></row></sheetData></worksheet>',
            )
    return out.getvalue()


@pytest.fixture
def reader():
    return base.load_reader()


def snapshot(
    reader,
    raw=b'ID,Name,Value\n00123,"Alice, A",0\n00456,"two\nlines",\n00789\n',
    name="data.csv",
    **options,
):
    valves = reader.Pipe.Valves(**options)
    source = reader.parse_table_bytes(raw, name, valves)
    result = reader.build_snapshot(source, name, base.FILE_ID, base.MODEL_ID, valves)
    result.update(
        reader_chat_id="saved-reader-chat",
        reader_message_id="saved-reader",
        source_origin={"kind": "file"},
    )
    return result


def validate_saved(reader, s):
    ref = {
        "message": s["reader_message_id"],
        "fingerprint": s["fingerprint"],
        "passage": s["passages"][0]["id"],
    }
    return reader.question_snapshot(reader.render_reader(s), ref, s["reader_chat_id"])


def test_csv_preserves_quotes_multiline_identifiers_empty_and_missing(reader):
    s = snapshot(reader)
    rows = s["table_source"]["tables"][0]["rows"]
    assert rows[1][0]["value"] == "00123"
    assert rows[1][1]["value"] == "Alice, A"
    assert rows[2][1]["value"] == "two\nlines"
    assert rows[2][2]["kind"] == "text" and rows[2][2]["value"] == ""
    assert rows[3][2]["kind"] == "missing"
    units = [u for p in s["passages"] for u in p["units"]]
    assert units[-1]["source_range"] == "'Table'!A4:C4"
    assert units[-1]["table_row"] == 4  # record rows, not physical lines
    assert validate_saved(reader, s)["source_sha256"] == s["source_sha256"]


def test_blank_rows_duplicate_headers_and_no_header_remain_inspectable(reader):
    s = snapshot(reader, b",Name,Name\n\n001,one,two\n", TABLE_HEADER_ROW=0)
    rows = s["table_source"]["tables"][0]["rows"]
    assert len(rows) == 3
    assert all(c["kind"] == "missing" for c in rows[1])
    assert s["table_source"]["header_row"] == 0
    assert rows[0][1] == rows[0][2]
    assert validate_saved(reader, s)


def test_encoding_and_delimiter_are_explicit_not_guessed(reader):
    raw = "ID;Name\n001;café\n".encode("cp1252")
    with pytest.raises(reader.ReaderError, match="encoding"):
        snapshot(reader, raw)
    s = snapshot(reader, raw, CSV_ENCODING="cp1252", CSV_DELIMITER="semicolon")
    assert s["table_source"]["encoding"] == "cp1252"
    assert s["table_source"]["tables"][0]["rows"][1][1]["value"] == "café"
    tsv = snapshot(reader, b"ID\tName\n0001\tBob\n", "test.tsv")
    assert tsv["table_source"]["delimiter"] == "\t"
    assert len(tsv["table_source"]["tables"][0]["rows"][0]) == 2


@pytest.mark.parametrize(
    "options",
    [
        {"MAX_TABLE_ROWS": 2},
        {"MAX_TABLE_COLUMNS": 2},
        {"MAX_TABLE_CELLS": 4},
        {"MAX_SOURCE_CHARS": 100},
        {"TABLE_HEADER_ROW": 9},
    ],
)
def test_limits_reject_without_truncating(reader, options):
    with pytest.raises(reader.ReaderError):
        snapshot(reader, **options)


def test_invalid_csv_empty_and_byte_limit(reader):
    for raw in (b"", b'ID,Value\n1,"unterminated'):
        with pytest.raises(reader.ReaderError):
            snapshot(reader, raw)
    with pytest.raises(reader.ReaderError, match="MAX_TABLE_BYTES"):
        snapshot(reader, b"x" * 1025, MAX_TABLE_BYTES=1024)


def test_excel_coordinates_types_formulas_dates_hidden_and_blank(reader):
    s = snapshot(reader, xlsx_bytes(), "example.xlsx")
    t, hidden = s["table_source"]["tables"]
    assert hidden["hidden"]
    assert t["hidden_rows"] == [2] and t["hidden_columns"] == [2]
    rows = t["rows"]
    assert rows[1][0]["value"] == "00123"
    assert rows[1][1]["kind"] == "number" and rows[1][1]["value"] == "0"
    assert rows[1][2]["kind"] == "date" and rows[1][2]["value"].startswith("2024-01-01")
    assert rows[2][0]["value"] == "=SUM(B2:B2)" and rows[2][0]["cached"] == "0"
    assert rows[2][1]["cached"] is None
    assert rows[2][2]["kind"] == "boolean"
    assert rows[3][0]["kind"] == "error"
    assert rows[3][2]["kind"] == "blank"
    assert "'Risks'!A2:C2" in [
        u["source_range"] for p in s["passages"] for u in p["units"]
    ]
    assert validate_saved(reader, s)


def test_merged_cells_remain_source_only(reader):
    s = snapshot(
        reader,
        xlsx_bytes(
            merged='<mergeCells count="1"><mergeCell ref="A1:B1"/></mergeCells>',
            second=False,
        ),
        "merged.xlsx",
    )
    assert all(p["source_only"] for p in s["passages"])
    assert s["table_source"]["tables"][0]["merged"] == ["A1:B1"]
    # Normal-mode spreadsheet readers discard non-anchor merged values; keep
    # the actual uploaded cell even in unusual/malformed merged layouts.
    assert s["table_source"]["tables"][0]["rows"][0][1]["value"] == "Value"
    assert validate_saved(reader, s)


@pytest.mark.parametrize(
    "xml,merged",
    [
        ('<sheetData><row r="1"><c r="XFD1"><v>1</v></c></row></sheetData>', ""),
        (
            '<sheetData><row r="1000000"><c r="A1000000"><v>1</v></c></row></sheetData>',
            "",
        ),
        ("<sheetData/>", '<mergeCells><mergeCell ref="A1:XFD1048576"/></mergeCells>'),
        ('<!DOCTYPE x [<!ENTITY x "abc">]><sheetData/>', ""),
    ],
)
def test_excel_hostile_dimensions_and_entities_rejected_before_load(
    reader, xml, merged
):
    with patch("openpyxl.load_workbook") as load:
        with pytest.raises(reader.ReaderError):
            snapshot(reader, xlsx_bytes(xml, merged), "bad.xlsx")
        load.assert_not_called()


def test_excel_sheet_limit_and_bad_archive(reader):
    with pytest.raises(reader.ReaderError, match="SHEETS"):
        snapshot(reader, xlsx_bytes(), "many.xlsx", MAX_TABLE_SHEETS=1)
    with pytest.raises(reader.ReaderError, match="could not be read"):
        snapshot(reader, b"not a zip", "bad.xlsx")


def test_excel_empty_worksheet_and_empty_formula_cache(reader):
    s = snapshot(reader, xlsx_bytes("<sheetData/>", second=False), "empty.xlsx")
    assert s["table_source"]["tables"][0]["rows"][0][0]["kind"] == "blank"
    xml = '<sheetData><row r="1"><c r="A1" t="str"><f>IF(1=1,&quot;&quot;,&quot;value&quot;)</f><v></v></c></row></sheetData>'
    s = snapshot(reader, xlsx_bytes(xml, second=False), "formula.xlsx")
    cell = s["table_source"]["tables"][0]["rows"][0][0]
    assert cell["cached"] == "" and cell["cached_kind"] == "text"
    assert validate_saved(reader, s)


def test_excel_dependency_error_is_actionable(reader):
    with patch.dict(sys.modules, {"openpyxl": None}):
        with pytest.raises(reader.ReaderError, match="requires openpyxl"):
            snapshot(reader, xlsx_bytes(), "test.xlsx")


def test_excel_expansion_limit_before_parser(reader):
    from zipfile import ZIP_DEFLATED

    raw = io.BytesIO()
    with ZipFile(raw, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("xl/large.xml", b" " * (20 * 1024 * 1024 + 1))
    with patch("openpyxl.load_workbook") as load:
        with pytest.raises(reader.ReaderError, match="expands beyond"):
            snapshot(reader, raw.getvalue(), "large.xlsx")
        load.assert_not_called()


@pytest.mark.parametrize(
    "mutation", ["cell", "range", "row", "section", "type", "version"]
)
def test_saved_table_tampering_cannot_diverge_from_evidence(reader, mutation):
    s = snapshot(reader)
    if mutation == "cell":
        s["table_source"]["tables"][0]["rows"][1][0]["value"] = "99999"
    elif mutation == "range":
        s["passages"][0]["units"][0]["source_range"] = "'Other'!A99:C99"
    elif mutation == "row":
        s["passages"][0]["units"][0]["table_row"] = 4
    elif mutation == "section":
        s["sections"][0]["title"] = "Invented"
    elif mutation == "type":
        s["passages"][0]["source_only"] = True
    else:
        s["version"] = 1
    with pytest.raises(reader.ReaderError, match="invalid"):
        validate_saved(reader, s)


class TablePipeTests(unittest.IsolatedAsyncioTestCase):
    setUp = base.PipeExecutionTests.setUp
    file_record = base.PipeExecutionTests.file_record
    capture_render = base.PipeExecutionTests.capture_render
    metadata = base.PipeExecutionTests.metadata
    run_pipe = base.PipeExecutionTests.run_pipe
    embed_events = base.PipeExecutionTests.embed_events

    def storage(self, raw=b"ID,Name\n00123,Example\n", filename="test.csv"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / filename
        path.write_bytes(raw)
        provider = ModuleType("open_webui.storage.provider")
        self.storage_get = Mock(return_value=str(path))
        provider.Storage = NS(get_file=self.storage_get)
        p = patch.dict(sys.modules, {"open_webui.storage.provider": provider})
        p.start()
        self.addCleanup(p.stop)
        self.files.return_value = self.file_record(
            filename=filename,
            path="server-controlled-path",
            data={"status": "failed", "content": "flattened and wrong"},
        )
        return path

    async def prepare(self):
        name, source, fid, _, _ = await self.pipe._load_input(
            self.metadata(), self.user, self.pipe.valves, self.emitter
        )
        s = self.reader.build_snapshot(
            source, name, fid, base.MODEL_ID, self.pipe.valves
        )
        batches = self.reader.make_batches(s, self.pipe.valves)
        self.complete.side_effect = [
            base.completion_response(base.batch_response(s, b)) for b in batches
        ]
        out = await self.run_pipe()
        self.assertIn("Reader ready", out)
        return self.rendered[-1]

    async def test_original_bytes_used_even_when_extraction_failed(self):
        self.storage()
        s = await self.prepare()
        self.assertEqual(s["version"], 2)
        self.assertIn("00123", base.source_text(s))
        self.assertNotIn("flattened and wrong", base.source_text(s))
        self.storage_get.assert_called_with("server-controlled-path")

    async def test_no_read_before_authorization_and_no_model_call(self):
        self.storage()
        self.files.return_value.user_id = "other-owner"
        out = await self.run_pipe()
        self.assertIn("do not have access", out)
        self.storage_get.assert_not_called()
        self.complete.assert_not_called()

    async def test_current_source_rechecks_access_and_detects_original_byte_change(
        self,
    ):
        path = self.storage()
        s = await self.prepare()
        current = await self.pipe._current_source(
            s, self.metadata(), self.user, self.pipe.valves, self.emitter
        )
        self.assertEqual(current, base.source_text(s))
        path.write_bytes(b"ID,Name\n00123,Changed\n")
        current = await self.pipe._current_source(
            s, self.metadata(), self.user, self.pipe.valves, self.emitter
        )
        self.assertNotEqual(current, base.source_text(s))
        self.files.return_value.user_id = "other-owner"
        self.storage_get.reset_mock()
        with self.assertRaises(self.reader.ReaderError):
            await self.pipe._current_source(
                s, self.metadata(), self.user, self.pipe.valves, self.emitter
            )
        self.storage_get.assert_not_called()

    async def test_excel_prepare_uses_rows_as_validatable_model_evidence(self):
        self.storage(xlsx_bytes(), "test.xlsx")
        s = await self.prepare()
        self.assertEqual(len(s["sections"]), 2)
        self.assertTrue(any(p["generated"] for p in s["passages"]))

    async def save_reader(self):
        s = copy.deepcopy(await self.prepare())
        self.render_patch.stop()
        self.chat_message.return_value = {
            "role": "assistant",
            "embeds": [self.reader.render_reader(s)],
        }
        self.complete.reset_mock(side_effect=True)
        self.emitter.reset_mock()
        return s

    async def followup(self, s, operation):
        ref = {
            "v": 1,
            "message": s["reader_message_id"],
            "fingerprint": s["fingerprint"],
        }
        if operation == "brief":
            ref.update(passages=[s["passages"][0]["id"]], explanations=True)
        else:
            ref["passage"] = s["passages"][0]["id"]
        encoded = (
            base64.urlsafe_b64encode(json.dumps(ref).encode()).decode().rstrip("=")
        )
        text = f"# Reader {operation}\n\nRead the stored row values.\n\n[Source: saved Reader](/c/saved-reader-chat#document-reader-{operation}-v1={encoded})"
        metadata = self.metadata()
        metadata.update(
            message_id="followup-response",
            user_message={"role": "user", "content": text},
        )
        return await self.run_pipe(__metadata__=metadata)

    async def test_saved_table_retry_reuses_and_changed_header_or_bytes_rejects(self):
        path = self.storage()
        s = await self.save_reader()
        self.assertIn("Reader ready", await self.followup(s, "retry"))
        self.complete.assert_not_awaited()
        self.pipe.valves.TABLE_HEADER_ROW = 0
        self.assertIn("changed", await self.followup(s, "retry"))
        self.pipe.valves.TABLE_HEADER_ROW = 1
        path.write_bytes(b"ID,Name\n00123,Changed\n")
        self.assertIn("changed", await self.followup(s, "retry"))
        self.complete.assert_not_awaited()

    async def test_saved_table_question_and_brief_emit_precise_native_citations(self):
        self.storage(xlsx_bytes(), "test.xlsx")
        s = await self.save_reader()
        unit = s["passages"][0]["units"][1]
        self.complete.return_value = base.completion_response(
            {
                "status": "answered",
                "points": [
                    {"text": "The identifier is 00123.", "evidence": [unit["id"]]}
                ],
            }
        )
        self.assertIn("Answer about this passage", await self.followup(s, "question"))
        event = next(
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        )
        self.assertEqual(event["data"]["metadata"][0]["source_range"], "'Risks'!A2:C2")
        self.assertEqual(event["data"]["metadata"][0]["file_id"], base.FILE_ID)
        self.assertEqual(event["data"]["document"], [unit["text"]])
        self.complete.reset_mock()
        self.emitter.reset_mock()
        self.assertIn("## Reading brief:", await self.followup(s, "brief"))
        self.complete.assert_not_awaited()
        citations = [
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        ]
        self.assertTrue(citations)
        self.assertTrue(
            all("source_range" in e["data"]["metadata"][0] for e in citations)
        )
        self.files.return_value.user_id = "other-owner"
        self.emitter.reset_mock()
        self.assertIn("no longer have access", await self.followup(s, "question"))
        self.assertIn("no longer have access", await self.followup(s, "brief"))
        self.complete.assert_not_awaited()
        self.assertFalse(
            any(c.args[0]["type"] == "citation" for c in self.emitter.await_args_list)
        )
