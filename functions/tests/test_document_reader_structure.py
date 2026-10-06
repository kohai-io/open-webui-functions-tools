"""Synthetic extraction structure, exact-source and browser rendering contracts."""

import copy
import json
import os

import pytest

from test_document_reader import load_reader
from test_document_reader_browser import browser, render_reader


FOOTER = "Service review pack - Version 2.0 October 2026"
TEXT = (
    "acme\n\n# Service Review Pack\nfor the Customer Support Pilot\n\n"
    "# Contents\n\n1. Scope 3\n2. Controls 4\n3. Review 5\n\n"
    + FOOTER
    + "\n\n2 acme\n\n"
    "## 1. Scope\n\n"
    "- **Communication:** Review messages before sending.\n"
    "  - Email composition\n"
    "  - Meeting notes\n"
    "- *Research:* Verify every material claim.\n\n"
    "The pilot is limited to the current\n\n"
    + FOOTER
    + "\n\n3 ![img-0.jpeg](img-0.jpeg)\n\n"
    "support team. Approval is still required.\n\n"
    "## *2. Controls*\n\n"
    "You must retain the approval condition.\n\n" + FOOTER + "\n\n4 acme\n\n"
    "## 3. Review\n\n"
    "The review is provisional. " + FOOTER + "\n\n5 acme\n"
)


@pytest.fixture
def reader():
    return load_reader()


def snapshot(reader, text=TEXT, filename="review.pdf"):
    return reader.build_snapshot(
        text, filename, "synthetic-file", "permitted-model", reader.Pipe.Valves()
    )


def test_source_partition_survives_front_matter_and_footer_detection(reader):
    value = snapshot(reader)
    units = [u for p in value["passages"] for u in p["units"]]
    assert "".join(u["text"] for u in units) == TEXT
    assert all(TEXT[u["start"] : u["end"]] == u["text"] for u in units)
    assert all(a["end"] == b["start"] for a, b in zip(units, units[1:]))
    assert [s["title"] for s in value["sections"][:3]] == [
        "Document cover",
        "Contents",
        "1. Scope",
    ]
    assert value["sections"][3]["title"] == "2. Controls"
    assert all(p["source_only"] for p in value["passages"][:2])
    excluded = "".join(u["text"] for u in units if u.get("excluded"))
    assert excluded.count(FOOTER) == 4
    assert "img-0.jpeg" in excluded
    assert "support team" not in excluded
    assert value["extraction_metadata"][0]["text"] == FOOTER
    assert value["extraction_metadata"][0]["evidence"]


def test_only_eligible_source_reaches_model_and_excluded_citations_are_rejected(reader):
    value = snapshot(reader)
    batches = reader.make_batches(value, reader.Pipe.Valves())
    assert len(batches) == 3
    messages = [reader.batch_messages(batch, value)[1]["content"] for batch in batches]
    assert all(
        FOOTER not in message and "img-0.jpeg" not in message for message in messages
    )
    assert all("Service Review Pack" not in message for message in messages)
    batch = batches[0]
    lookup = {p["id"]: p for p in value["passages"]}
    results = []
    for pid in batch["passage_ids"]:
        eligible = [
            u["id"]
            for u in lookup[pid]["units"]
            if u["text"].strip() and not u.get("excluded")
        ]
        item = {"text": "Approval remains required.", "evidence": eligible[:1]}
        results.append(
            {
                "id": pid,
                "extract_ids": eligible[:1],
                "explanation": [item],
                "takeaways": [item],
            }
        )
    result = {"passages": results, "overview": results[0]["explanation"][0]}
    reader.validate_result(json.dumps(result), batch, value)
    bad = copy.deepcopy(result)
    excluded_id = next(
        u["id"] for u in lookup[batch["passage_ids"][0]]["units"] if u.get("excluded")
    )
    bad["passages"][0]["extract_ids"] = [excluded_id]
    with pytest.raises(reader.ReaderError, match="invalid source references"):
        reader.validate_result(json.dumps(bad), batch, value)


def test_repetition_is_not_enough_to_remove_substantive_text(reader):
    condition = "You must retain version 2.0 until approval.\n\n3 acme\n"
    text = condition * 4
    assert reader._page_furniture(text, "review.pdf") == []
    assert reader._page_furniture(TEXT, "review.md") == []
    only_two = (FOOTER + "\n\n3 acme\n") * 2
    assert reader._page_furniture(only_two, "review.pdf") == []
    without_pages = (FOOTER + "\nThe approval is provisional.\n") * 4
    assert reader._page_furniture(without_pages, "review.pdf") == []


def test_substantive_preface_remains_eligible_and_fenced_headings_are_inert(reader):
    text = TEXT.replace(
        "acme\n\n# Service Review Pack\nfor the Customer Support Pilot",
        "# Executive Summary\nApproval must remain provisional.",
    )
    value = snapshot(reader, text)
    assert value["sections"][0]["title"] == "Executive Summary"
    assert any(
        not p["source_only"]
        and "Approval must" in "".join(u["text"] for u in p["units"])
        for p in value["passages"]
    )
    code = "# Real heading\n\n```text\n# Literal code heading\n```\n"
    value = snapshot(reader, code, "code.md")
    assert [s["title"] for s in value["sections"]] == ["Real heading"]


@pytest.mark.skipif(
    os.getenv("RUN_DOCUMENT_READER_BROWSER_TESTS") != "1",
    reason="opt-in sandbox browser test",
)
def test_formatted_reading_preserves_lists_and_exact_inspector_without_network(
    reader, browser, render_reader
):
    # Keep hostile HTML, code, and URL syntax in the same reading passage.
    hostile = "\n\n<script>window.injected=true</script>\n\n![private](https://invalid.test/leak)\n\n`safe code`\n"
    value = snapshot(
        reader,
        TEXT.replace(
            "support team. Approval is still required.",
            "support team. Approval is still required." + hostile,
        ),
    )
    scope = next(p for p in value["passages"] if not p["source_only"])
    extract = next(u for u in scope["units"] if "**Communication:" in u["text"])
    scope["generated"] = {
        "extract_ids": [extract["id"]],
        "explanation": [{"text": "Review messages.", "evidence": [extract["id"]]}],
        "takeaways": [{"text": "Review messages.", "evidence": [extract["id"]]}],
    }
    page = browser.new_page(viewport={"width": 1800, "height": 900})
    page.set_default_timeout(10000)
    network, errors = [], []
    page.on("request", lambda request: network.append(request.url))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_content(
        '<iframe title="Reader" sandbox="allow-scripts" style="width:100%;height:720px;border:0"></iframe>'
    )
    page.locator("iframe").evaluate(
        "(el, html) => el.srcdoc=html", render_reader(value)
    )
    frame = page.locator("iframe").content_frame
    frame.locator("#reader").wait_for(state="visible")
    assert frame.locator(".front-matter").count() == 0
    assert (
        frame.locator(".front-details")
        .inner_text()
        .startswith("Cover and original contents")
    )
    frame.get_by_role("button", name="Full text", exact=True).click()
    scope_section = next(s for s in value["sections"] if s["title"] == "1. Scope")
    frame.locator(f'#outline-list [data-section="{scope_section["id"]}"]').click()
    heading_top = frame.get_by_role("heading", name="1. Scope", exact=True).evaluate(
        "el=>el.getBoundingClientRect().top-document.getElementById('reading-column').getBoundingClientRect().top"
    )
    assert abs(heading_top - 18) < 4
    passage = frame.locator("#passage-" + scope["id"])
    assert passage.locator("ul>li>ul>li").count() == 2
    assert passage.locator("strong").inner_text() == "Communication:"
    assert passage.locator("em").inner_text() == "Research:"
    assert FOOTER not in passage.inner_text()
    assert "current support team" in passage.inner_text()
    assert "img-0.jpeg" not in passage.inner_text()
    assert passage.locator("script,img,a").count() == 0
    assert frame.locator("body").evaluate("() => window.injected") is None
    frame.get_by_role("button", name="About this reader", exact=True).click()
    assert frame.locator(".source-metadata p").inner_text() == FOOTER
    frame.get_by_role("button", name="Close information", exact=True).click()
    frame.get_by_role("button", name="Extracts", exact=True).click()
    quote = frame.locator(f'[data-extract="{extract["id"]}"]')
    assert quote.text_content() == extract["text"]
    assert quote.locator("strong").inner_text() == "Communication:"
    assert "**" not in quote.inner_text()
    frame.get_by_role("button", name="Full text", exact=True).click()
    passage.get_by_role("button", name="Inspect source", exact=False).click()
    expected = "".join(u["text"] for u in scope["units"])
    assert frame.locator("#source-text").text_content() == expected
    assert FOOTER in expected and "img-0.jpeg" in expected
    assert network == [] and errors == []
    page.close()
