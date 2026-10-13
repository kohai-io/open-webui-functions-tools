"""Local source search and explicit brief-to-chat drafts in opaque embeds."""

import base64
import json

from test_document_reader_browser import browser, mounted, pytestmark, render_reader


def test_search_finds_hidden_source_and_opens_exact_evidence(mounted, render_reader):
    page, frame, snapshot, *_ = mounted
    # Real backend snapshots omit the browser-only index; derive it at runtime.
    for passage in snapshot["passages"]:
        passage.pop("index", None)
    page.evaluate(
        "html=>document.querySelector('iframe').srcdoc=html", render_reader(snapshot)
    )
    frame.locator("#reader").wait_for(state="visible")
    frame.get_by_role("button", name="Section map", exact=True).click()
    frame.get_by_role("button", name="Find", exact=True).click()
    frame.get_by_label("Search the complete source").fill("WORKSTREAM 17 will test")
    assert "1 match" in frame.locator("#search-status").inner_text()
    assert "Passage 17" in frame.locator(".search-result").inner_text()
    assert (
        frame.locator("#search-results mark").text_content()
        == "Workstream 17 will   test"
    )
    frame.locator(".search-result").get_by_role(
        "button", name="Inspect source", exact=True
    ).click()
    assert (
        frame.locator('#level-controls [aria-pressed="true"]').get_attribute(
            "data-level"
        )
        == "full"
    )
    assert frame.locator("#source-dialog").is_visible()
    assert frame.locator("#source-text").text_content() == "".join(
        u["text"] for u in snapshot["passages"][16]["units"]
    )
    assert "Workstream 17" in frame.locator("#source-text .highlight").inner_text()
    frame.get_by_role("button", name="Close source", exact=True).click()
    assert frame.locator("#passage-p0017").evaluate("el=>el===document.activeElement")


def test_search_is_literal_bounded_and_cancel_does_not_change_level(mounted):
    _, frame, *_ = mounted
    frame.get_by_role("button", name="Takeaways", exact=True).click()
    frame.get_by_role("button", name="Find", exact=True).click()
    field = frame.get_by_label("Search the complete source")
    field.fill(".*[<script>")
    assert "No matches" in frame.locator("#search-status").inner_text()
    assert frame.locator("#search-results script").count() == 0
    field.fill("e")
    assert frame.locator(".search-result").count() == 100
    assert "first 100" in frame.locator("#search-status").inner_text()
    field.fill("")
    assert frame.locator(".search-result").count() == 0
    frame.get_by_role("button", name="Close search", exact=True).click()
    assert (
        frame.locator('#level-controls [aria-pressed="true"]').get_attribute(
            "data-level"
        )
        == "takeaways"
    )
    assert frame.locator("#search-button").evaluate("el=>el===document.activeElement")


def test_search_spans_units_and_matches_unicode(mounted):
    _, frame, *_ = mounted
    frame.get_by_role("button", name="Find", exact=True).click()
    field = frame.get_by_label("Search the complete source")
    field.fill("team. The £24,000")
    frame.locator(".search-result").first.get_by_role(
        "button", name="Inspect source", exact=True
    ).click()
    assert frame.locator("#source-text .highlight").count() == 2
    frame.get_by_role("button", name="Close source", exact=True).click()
    frame.get_by_role("button", name="Find", exact=True).click()
    field.fill("東京")
    assert frame.locator("#search-results mark").first.inner_text() == "東京"


def test_brief_chat_draft_contains_selection_and_never_sends_automatically(mounted):
    page, frame, snapshot, *_ = mounted
    page.evaluate(
        "() => {window.briefMessages=[];window.addEventListener('message',e=>{if(e.data.type==='input:prompt')window.briefMessages.push(e.data)})}"
    )
    frame.get_by_role("button", name="Takeaways", exact=True).click()
    frame.locator("#passage-p0002").get_by_role(
        "button", name="Add to brief", exact=True
    ).click()
    frame.locator("#passage-p0001").get_by_role(
        "button", name="Add to brief", exact=True
    ).click()
    frame.get_by_role("button", name="Brief (2)", exact=True).click()
    frame.get_by_label("Include AI explanations").check()
    frame.get_by_role("button", name="Send brief to chat", exact=True).click()
    draft = frame.get_by_label("Brief request").input_value()
    assert draft.startswith("# Reader brief\n")
    encoded = draft.split("document-reader-brief-v1=")[1].split(")")[0]
    ref = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert ref == {
        "v": 1,
        "message": snapshot["reader_message_id"],
        "fingerprint": snapshot["fingerprint"],
        "passages": ["p0001", "p0002"],
        "explanations": True,
    }
    assert page.evaluate("window.briefMessages") == []
    frame.get_by_role("button", name="Replace chat draft", exact=True).click()
    page.wait_for_function("window.briefMessages.length===1")
    assert page.evaluate("window.briefMessages") == [
        {"type": "input:prompt", "text": draft}
    ]
    frame.get_by_role("button", name="Close brief draft", exact=True).click()
    assert frame.locator("#brief-button").evaluate("el=>el===document.activeElement")


def test_empty_brief_cannot_be_sent_and_mobile_controls_fit(mounted):
    page, frame, *_ = mounted
    page.set_viewport_size({"width": 390, "height": 800})
    frame.get_by_role("button", name="Brief (0)", exact=True).click()
    assert frame.get_by_role(
        "button", name="Send brief to chat", exact=True
    ).is_disabled()
    assert frame.locator("body").evaluate(
        "()=>document.documentElement.scrollWidth<=innerWidth"
    )


def test_search_opens_reader_and_keeps_found_passage_through_all_levels(mounted):
    page, frame, *_ = mounted
    frame.get_by_role("button", name="Find", exact=True).click()
    field = frame.get_by_label("Search the complete source")
    field.fill("Workstream 17 will test")
    frame.get_by_role("button", name="Open in reader", exact=True).filter(
        visible=True
    ).click()
    assert not frame.locator("#search-dialog").is_visible()
    assert not frame.locator("#source-dialog").is_visible()
    assert frame.locator("#passage-p0017").evaluate("el=>el===document.activeElement")
    for level, key in (
        ("Full text", "full"),
        ("Extracts", "extracts"),
        ("Explanation", "explanation"),
        ("Takeaways", "takeaways"),
        ("Section map", "map"),
        ("Full text", "full"),
    ):
        frame.get_by_role("button", name=level, exact=True).click()
        assert (
            frame.locator('#level-controls [aria-pressed="true"]').get_attribute(
                "data-level"
            )
            == key
        )
        assert frame.locator("#position").get_attribute("data-passage") == "p0017"
        if key != "map":
            assert frame.locator("#passage-p0017").evaluate(
                "el=>{const r=el.getBoundingClientRect(),s=document.querySelector('#reading-column').getBoundingClientRect();return r.top<s.bottom&&r.bottom>s.top}"
            )
    frame.get_by_role("button", name="Find", exact=True).click()
    assert field.input_value() == "Workstream 17 will test"
    page.set_viewport_size({"width": 390, "height": 800})
    assert (
        frame.get_by_role("button", name="Open in reader", exact=True)
        .filter(visible=True)
        .is_visible()
    )
    assert frame.locator("body").evaluate(
        "()=>document.documentElement.scrollWidth<=innerWidth"
    )


def test_source_inspector_opens_current_passage_in_reader(mounted):
    _, frame, *_ = mounted
    frame.get_by_role("button", name="Find", exact=True).click()
    frame.get_by_label("Search the complete source").fill("Workstream 17 will test")
    frame.locator(".search-result").get_by_role(
        "button", name="Inspect source", exact=True
    ).click()
    frame.get_by_role("button", name="Next passage →", exact=True).click()
    frame.locator("#source-dialog").get_by_role(
        "button", name="Open in reader", exact=True
    ).click()
    assert not frame.locator("#source-dialog").is_visible()
    assert frame.locator("#position").get_attribute("data-passage") == "p0018"
    assert frame.locator("#passage-p0018").evaluate("el=>el===document.activeElement")
    frame.get_by_role("button", name="Takeaways", exact=True).click()
    assert frame.locator("#position").get_attribute("data-passage") == "p0018"


def test_found_passage_stays_selected_when_document_is_too_short_to_scroll(
    mounted, render_reader
):
    page, frame, snapshot, *_ = mounted
    snapshot["passages"] = snapshot["passages"][:3]
    snapshot["sections"] = snapshot["sections"][:1]
    snapshot["sections"][0]["passage_ids"] = [p["id"] for p in snapshot["passages"]]
    snapshot["overviews"] = []
    page.evaluate(
        "html=>document.querySelector('iframe').srcdoc=html", render_reader(snapshot)
    )
    frame.locator("#reader").wait_for(state="visible")
    frame.get_by_role("button", name="Find", exact=True).click()
    frame.get_by_label("Search the complete source").fill("Workstream 3 will test")
    frame.locator("#search-dialog").get_by_role(
        "button", name="Open in reader", exact=True
    ).click()
    for level in ("Extracts", "Explanation", "Takeaways", "Section map", "Full text"):
        frame.get_by_role("button", name=level, exact=True).click()
        assert frame.locator("#position").get_attribute("data-passage") == "p0003"
    # Choosing another section releases the search anchor.
    frame.locator("#outline-list button").first.click()
    frame.locator("#reading-column").press("Home")
    frame.locator("#reading-column").evaluate(
        "el=>{el.scrollTop=0;el.dispatchEvent(new Event('scroll'))}"
    )
    frame.locator("body").evaluate(
        "()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))"
    )
    assert frame.locator("#position").get_attribute("data-passage") == "p0001"
