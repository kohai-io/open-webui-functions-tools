"""Reader controls remain local, inert and usable in an opaque iframe."""

import base64
import json
from pathlib import Path

import pytest

from test_document_reader_browser import browser, mounted, pytestmark, render_reader


def test_partial_reader_retry_draft_uses_composer_without_sending(
    mounted, render_reader
):
    page, frame, snapshot, *_ = mounted
    assert not frame.locator("#retry-button").is_visible()
    snapshot["preparation_identity"] = "b" * 64
    snapshot["status"] = "partial"
    page.evaluate(
        "html=>{window.retryMessages=[];window.addEventListener('message',e=>{if(e.data.type==='input:prompt')window.retryMessages.push(e.data)});document.querySelector('iframe').srcdoc=html;}",
        render_reader(snapshot),
    )
    frame.locator("#reader").wait_for(state="visible")
    frame.get_by_role("button", name="Retry missing sections", exact=True).click()
    draft = frame.locator("#retry-draft").input_value()
    assert draft.startswith("# Reader retry\n")
    encoded = draft.split("document-reader-retry-v1=")[1].split(")")[0]
    ref = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert ref["message"] == snapshot["reader_message_id"]
    assert ref["fingerprint"] == snapshot["fingerprint"]
    assert page.evaluate("window.retryMessages") == []
    frame.locator("#retry-replace").click()
    page.wait_for_function("window.retryMessages.length===1")
    assert page.evaluate("window.retryMessages") == [
        {"type": "input:prompt", "text": draft}
    ]
    frame.get_by_role("button", name="Close retry", exact=True).click()
    assert frame.locator("#retry-button").evaluate("n=>n===document.activeElement")


def test_reading_length_counts_visible_words_without_hiding_source(mounted):
    _, frame, snapshot, *_ = mounted
    frame.get_by_role("button", name="Takeaways", exact=True).click()
    first = snapshot["passages"][0]
    source = "".join(u["text"] for u in first["units"])
    shown = " ".join(item["text"] for item in first["generated"]["takeaways"])
    assert (
        frame.locator("#passage-p0001 .reading-length").text_content()
        == f"{len(shown.split())} words · source {len(source.split())}"
    )
    frame.get_by_role("button", name="Full text", exact=True).click()
    assert frame.locator(".reading-length").count() == 0


def level(frame):
    return frame.locator('#level-controls [aria-pressed="true"]').get_attribute(
        "data-level"
    )


def test_wheel_steps_once_per_burst_and_does_not_steal_browser_zoom(mounted):
    _, frame, *_ = mounted
    frame.get_by_role("button", name="Full text", exact=True).click()
    bar = frame.locator("#level-controls")
    bar.dispatch_event("wheel", {"deltaY": -30})
    assert level(frame) == "full"
    bar.dispatch_event("wheel", {"deltaY": -30})
    assert level(frame) == "extracts"
    bar.dispatch_event("wheel", {"deltaY": -500})
    assert level(frame) == "extracts"
    bar.dispatch_event("wheel", {"deltaY": 60})
    assert level(frame) == "full"
    bar.dispatch_event("wheel", {"deltaY": 100, "ctrlKey": True})
    bar.dispatch_event("wheel", {"deltaY": 100, "metaKey": True})
    frame.locator("#reading-column").dispatch_event("wheel", {"deltaY": 100})
    frame.locator("#reading-column").press("3")
    assert level(frame) == "full"
    bar.dispatch_event("wheel", {"deltaY": -4, "deltaMode": 1})
    assert level(frame) == "extracts"


def test_drag_previews_then_commits_and_cancel_preserves_level(mounted):
    page, frame, *_ = mounted
    frame.get_by_role("button", name="Full text", exact=True).click()
    first = frame.get_by_role("button", name="Full text", exact=True).bounding_box()
    last = frame.get_by_role("button", name="Takeaways", exact=True).bounding_box()
    page.mouse.move(first["x"] + first["width"] / 2, first["y"] + first["height"] / 2)
    page.mouse.down()
    page.mouse.move(
        last["x"] + last["width"] / 2, last["y"] + last["height"] / 2, steps=6
    )
    assert level(frame) == "full"
    assert (
        frame.locator('[data-zoom-target="true"]').get_attribute("data-level")
        == "takeaways"
    )
    page.mouse.up()
    assert level(frame) == "takeaways"
    bar = frame.locator("#level-controls")
    bar.dispatch_event(
        "pointerdown", {"pointerId": 99, "button": 0, "clientX": 100, "clientY": 100}
    )
    bar.dispatch_event("pointermove", {"pointerId": 99, "clientX": 102, "clientY": 140})
    bar.dispatch_event("pointerup", {"pointerId": 99, "clientX": 102, "clientY": 140})
    bar.dispatch_event("pointercancel")
    bar.dispatch_event(
        "pointerdown", {"pointerId": 98, "button": 0, "clientX": 100, "clientY": 100}
    )
    bar.dispatch_event("pointerleave", {"pointerId": 98})
    bar.dispatch_event("pointermove", {"pointerId": 98, "clientX": 140, "clientY": 100})
    assert level(frame) == "takeaways"
    assert frame.locator('[data-zoom-target="true"]').count() == 0


def test_question_draft_is_readable_scoped_and_never_submitted(mounted):
    page, frame, snapshot, *_ = mounted
    frame.get_by_role("button", name="Full text", exact=True).click()
    page.evaluate(
        "window.readerPrompts=[];window.addEventListener('message',e=>{if(e.data.type?.startsWith('input:'))window.readerPrompts.push(e.data)})"
    )
    frame.locator("#passage-p0001").get_by_role(
        "button", name="Ask about this passage"
    ).click()
    assert frame.locator("#question-copy").is_disabled()
    frame.locator("#question-input").fill("What approval is needed?")
    draft = frame.locator("#question-draft").input_value()
    assert draft.startswith("# Reader question\n\nWhat approval is needed?")
    token = draft.split("document-reader-question-v1=", 1)[1].rstrip(")")
    ref = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    assert ref == {
        "v": 1,
        "message": "saved-reader",
        "fingerprint": snapshot["fingerprint"],
        "passage": "p0001",
    }
    assert "/c/saved-reader-chat#" in draft
    assert frame.locator("#question-source").text_content() == "".join(
        u["text"] for u in snapshot["passages"][0]["units"]
    )
    frame.locator("#question-replace").click()
    page.wait_for_function("window.readerPrompts.length===1")
    assert page.evaluate("window.readerPrompts") == [
        {"type": "input:prompt", "text": draft}
    ]


def test_brief_is_source_ordered_deduplicated_and_downloads_exact_utf8(mounted):
    page, frame, snapshot, *_ = mounted
    frame.get_by_role("button", name="Full text", exact=True).click()
    for pid in ("p0002", "p0001"):
        frame.locator(f"#passage-{pid}").get_by_role(
            "button", name="Add to brief"
        ).click()
    frame.get_by_role("button", name="Brief (2)").click()
    text = frame.locator("#brief-preview").input_value()
    assert text.index("Test workstream 1") < text.index("Test workstream 2")
    assert "**AI explanations**" not in text
    for p in snapshot["passages"][:2]:
        for u in p["units"]:
            normalised = u["text"].replace("\r\n", "\n")
            expected_count = sum(
                v["text"] == u["text"]
                for p in snapshot["passages"][:2]
                for v in p["units"]
            )
            assert text.count(normalised) == expected_count
    frame.locator("#brief-explanations").check()
    assert "**AI explanations**" in frame.locator("#brief-preview").input_value()
    text = frame.locator("#brief-preview").input_value()
    with page.expect_download() as result:
        frame.get_by_role("button", name="Download Markdown").click()
    download = result.value
    assert download.suggested_filename.endswith("-reading-brief.md")
    downloaded = Path(download.path()).read_bytes().decode("utf-8")
    assert downloaded.replace("\r\n", "\n") == text
    for p in snapshot["passages"][:2]:
        for u in p["units"]:
            assert u["text"] in downloaded


def test_source_inspector_selection_survives_levels_resize_and_resets_reload(mounted):
    page, frame, snapshot, mount, *_ = mounted
    frame.get_by_role("button", name="Full text", exact=True).click()
    frame.locator("#passage-p0022").get_by_role(
        "button", name="Inspect source", exact=False
    ).first.click()
    frame.locator("#source-body").get_by_role("button", name="Add to brief").click()
    frame.get_by_role("button", name="Close source").click()
    frame.get_by_role("button", name="Explanation", exact=True).click()
    page.set_viewport_size({"width": 390, "height": 850})
    frame.get_by_role("button", name="Brief (1)").click()
    text = frame.locator("#brief-preview").input_value()
    assert "**Source only**" in text
    assert snapshot["passages"][21]["units"][0]["text"].replace("\r\n", "\n") in text
    assert frame.locator("#brief-dialog").evaluate(
        "el=>el.getBoundingClientRect().width<=innerWidth"
    )
    frame = mount()
    assert frame.get_by_role("button", name="Brief (0)").count() == 1


def test_brief_inert_source_fences_and_visible_size_limit(mounted, render_reader):
    page, frame, snapshot, *_ = mounted
    snapshot["passages"][23]["generated"] = None
    snapshot["passages"][23]["units"][0]["text"] = "```\nunsafe"
    snapshot["passages"][0]["generated"] = None
    snapshot["passages"][0]["units"][0]["text"] = "x" * 60001
    page.locator("iframe").evaluate(
        "(el, html)=>{el.srcdoc=html}", render_reader(snapshot)
    )
    frame.locator("#reader").wait_for(state="visible")
    frame.get_by_role("button", name="Full text", exact=True).click()
    frame.locator("#passage-p0024").get_by_role("button", name="Add to brief").click()
    frame.get_by_role("button", name="Brief (1)").click()
    text = frame.locator("#brief-preview").input_value()
    assert '<img src="https://invalid.test/leak"' in text
    assert frame.locator("#brief-dialog img, #brief-dialog script").count() == 0
    assert (
        frame.locator("body").evaluate("()=>typeof window.readerInjected")
        == "undefined"
    )
    assert "````text\n```\nunsafe\n````" in text
    frame.get_by_role("button", name="Close brief").click()
    frame.locator("#passage-p0001").get_by_role("button", name="Add to brief").click()
    assert frame.get_by_role("button", name="Brief (1)").count() == 1
    assert frame.locator("#brief-limit").is_visible()
