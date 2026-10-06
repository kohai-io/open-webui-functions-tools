"""Opt-in local Reader tests in OWUI's opaque-origin iframe sandbox.

Run with RUN_DOCUMENT_READER_BROWSER_TESTS=1; requires Python Playwright and
Chrome. No OWUI server, provider connection, browser downloads or credentials.
Set DOCUMENT_READER_SCREENSHOTS to a local directory to save inspection images.
"""

import base64
import importlib.util
import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCUMENT_READER_BROWSER_TESTS") != "1",
    reason="opt-in local sandbox browser tests",
)


def reader_snapshot():
    """Readable synthetic business document, including exact-text edge cases."""
    sections = [
        {
            "id": "s01",
            "title": "Purpose and scope",
            "heading_kind": "source",
            "passage_ids": [],
        },
        {
            "id": "s02",
            "title": "Delivery and evidence",
            "heading_kind": "source",
            "passage_ids": [],
        },
        {
            "id": "s03",
            "title": "Decision and next steps",
            "heading_kind": "fallback",
            "passage_ids": [],
        },
    ]
    passages = []
    for index in range(24):
        pid = f"p{index + 1:04d}"
        section = sections[index // 8]
        section["passage_ids"].append(pid)
        source_units = [
            {
                "id": f"u{index * 3 + 1:04d}",
                "text": f"Workstream {index + 1} will   test   a   small assisted-service pilot with the existing support team. ",
            },
            {
                "id": f"u{index * 3 + 2:04d}",
                "text": "The £24,000 budget is conditional on approval; no permanent rollout is authorised. ",
            },
            {
                "id": f"u{index * 3 + 3:04d}",
                "text": "Review customer outcomes before expanding the trial. Café / cafe\u0301, 東京 and 🧭 remain in the source.\r\n\r\n",
            },
        ]
        generated = {
            "extract_ids": [source_units[0]["id"], source_units[2]["id"]],
            "explanation": [
                {
                    "text": f"Workstream {index + 1} is a limited test using the current support team. Its budget needs approval and the proposal does not authorise a permanent rollout.",
                    "evidence": [source_units[0]["id"], source_units[1]["id"]],
                }
            ],
            "takeaways": [
                {
                    "text": f"Test workstream {index + 1} within the proposed budget, subject to approval and a review of customer outcomes.",
                    "evidence": [source_units[1]["id"], source_units[2]["id"]],
                }
            ],
        }
        passage = {
            "id": pid,
            "section_id": section["id"],
            "units": source_units,
            "source_only": False,
            "generated": generated,
        }
        if index == 20:
            passage.update(generated=None, error="This batch reached its time limit.")
        if index == 21:
            passage.update(
                generated=None,
                source_only=True,
                reason="Table columns could not be verified.",
            )
            source_units[0][
                "text"
            ] = "Team\tPilot capacity\r\nService A\t12\r\nService B\t8\r\n"
        if index == 23:
            source_units[1][
                "text"
            ] = '</script><img src="https://invalid.test/leak" onerror="window.readerInjected=true"><script>window.readerInjected=true</script>\n'
        passages.append(passage)
    return {
        "version": 1,
        "fingerprint": "fixture-source-sha256-segmentation-1",
        "filename": "Customer support pilot — decision brief.docx",
        "model_id": "permitted-test-model",
        "created_at": "2026-10-05T12:00:00Z",
        "status": "partial",
        "warnings": [
            "One preparation batch reached its time limit. Its source text remains readable."
        ],
        "sections": sections,
        "passages": passages,
        "overviews": [
            {
                "id": f"o{i + 1}",
                "section_id": section["id"],
                "passage_ids": section["passage_ids"][:2],
                "text": "Run a bounded support pilot, keep approval conditions visible and review the evidence before considering a wider rollout.",
                "evidence": [
                    passages[i * 8]["units"][1]["id"],
                    passages[i * 8 + 1]["units"][2]["id"],
                ],
            }
            for i, section in enumerate(sections)
        ],
    }


@pytest.fixture(scope="module")
def render_reader():
    path = Path(__file__).parents[1] / "document_reader.py"
    spec = importlib.util.spec_from_file_location(
        "document_reader_browser_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_reader


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        yield browser
        browser.close()


@pytest.fixture
def mounted(browser, render_reader):
    page = browser.new_page(
        viewport={"width": 1120, "height": 850}, reduced_motion="reduce"
    )
    page.set_default_timeout(10000)
    failures, network = [], []
    page.on("pageerror", lambda error: failures.append(str(error)))
    page.on("request", lambda request: network.append(request.url))
    snapshot = reader_snapshot()
    html = render_reader(snapshot)

    def mount(width=1120):
        page.set_viewport_size({"width": width, "height": 850})
        page.set_content(
            "<!doctype html><html><head><style>body{margin:0}</style></head><body></body></html>"
        )
        page.evaluate(
            """html => {
            const iframe = document.createElement('iframe');
            iframe.title = 'Document Reader';
            iframe.style.cssText = 'display:block;width:100%;height:720px;border:0';
            iframe.sandbox = 'allow-scripts allow-downloads';
            window.readerHeightMessages = [];
            window.addEventListener('message', e => {
                if (e.source === iframe.contentWindow && e.data.type === 'iframe:height') {
                    window.readerHeightMessages.push(e.data.height);
                    iframe.style.height = e.data.height + 'px';
                }
            });
            iframe.srcdoc = html;
            document.body.append(iframe);
        }""",
            html,
        )
        frame = page.locator("iframe").content_frame
        frame.locator("#reader").wait_for(state="visible")
        # Visibility precedes the Reader's queued startup position update.
        frame.locator("body").evaluate(
            "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
        )
        return frame

    frame = mount()
    yield page, frame, snapshot, mount, network, failures
    assert failures == []
    assert network == []
    page.close()


def screenshot(page, name):
    directory = os.getenv("DOCUMENT_READER_SCREENSHOTS")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        page.locator("iframe").screenshot(path=str(target / name))


def goto_passage(frame, passage_id, offset=18):
    frame.locator("#reading-column").evaluate(
        """(scroller, {id, offset}) => {
        const passage = document.getElementById('passage-' + id);
        scroller.scrollTop += passage.getBoundingClientRect().top - scroller.getBoundingClientRect().top - offset;
    }""",
        {"id": passage_id, "offset": offset},
    )
    frame.locator(f"[data-passage='{passage_id}'][data-active='true']").wait_for()


def anchor_top(frame, passage_id):
    return frame.locator(f"#passage-{passage_id}").evaluate(
        "el => el.getBoundingClientRect().top - document.getElementById('reading-column').getBoundingClientRect().top"
    )


def test_sandbox_exact_text_partial_coverage_and_inert_input(mounted):
    page, frame, snapshot, _, _, _ = mounted
    assert (
        frame.get_by_role("button", name="Section map", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    assert frame.locator("#snapshot-status").inner_text() == "Partly prepared"
    assert frame.locator("#position").get_attribute("data-passage") == "p0001"
    assert (
        page.locator("iframe").get_attribute("sandbox")
        == "allow-scripts allow-downloads"
    )
    assert frame.locator("body").evaluate("() => window.origin") == "null"
    assert frame.locator("body").evaluate(
        """() => {
        try { localStorage.getItem('reader-test'); return false; } catch (_) { return true; }
    }"""
    )
    screenshot(page, "reader-desktop.png")

    frame.get_by_role("button", name="Full text", exact=True).click()
    actual = frame.locator("#reading-content .source-unit").evaluate_all(
        "nodes => nodes.map(n => n.textContent).join('')"
    )
    expected = "".join(
        unit["text"] for passage in snapshot["passages"] for unit in passage["units"]
    )
    assert actual == expected
    assert frame.locator("#reading-content img, #reading-content script").count() == 0
    assert frame.locator("body").evaluate("() => window.readerInjected") is None

    frame.get_by_role("button", name="Extracts", exact=True).click()
    for passage in snapshot["passages"]:
        if not passage["generated"]:
            continue
        selected = set(passage["generated"]["extract_ids"])
        expected = [unit["text"] for unit in passage["units"] if unit["id"] in selected]
        assert (
            frame.locator(
                f"#passage-{passage['id']} [data-extract]"
            ).all_text_contents()
            == expected
        )

    frame.get_by_role("button", name="Takeaways", exact=True).click()
    assert "AI level unavailable" in frame.locator("#passage-p0021").inner_text()
    assert "Source only" in frame.locator("#passage-p0022").inner_text()
    assert page.evaluate(
        "window.readerHeightMessages.every(h => h >= 480 && h <= 1000)"
    )


def test_every_text_transition_map_and_inline_expansion_preserve_place(mounted):
    _, frame, _, _, _, _ = mounted
    levels = ["Full text", "Extracts", "Explanation", "Takeaways"]
    for before in levels:
        for after in levels:
            if before == after:
                continue
            frame.get_by_role("button", name=before, exact=True).click()
            goto_passage(frame, "p0012")
            old = anchor_top(frame, "p0012")
            frame.get_by_role("button", name=after, exact=True).click()
            assert abs(anchor_top(frame, "p0012") - old) <= 24
            assert frame.locator("#position").get_attribute("data-passage") == "p0012"

    frame.get_by_role("button", name="Takeaways", exact=True).click()
    goto_passage(frame, "p0012")
    old = anchor_top(frame, "p0012")
    frame.get_by_role("button", name="Section map", exact=True).click()
    assert frame.locator(".map-card").count() == 3
    frame.get_by_role("button", name="Return to takeaways", exact=False).click()
    assert abs(anchor_top(frame, "p0012") - old) <= 24
    frame.locator("#passage-p0012").get_by_role("button", name="Expand here").click()
    assert (
        frame.locator("#passage-p0012")
        .get_by_role("button", name="Collapse passage")
        .get_attribute("aria-expanded")
        == "true"
    )
    assert abs(anchor_top(frame, "p0012") - old) <= 24
    frame.locator("#passage-p0012").get_by_role(
        "button", name="Collapse passage"
    ).click()
    assert (
        frame.locator("#passage-p0012")
        .get_by_role("button", name="Expand here")
        .evaluate("el => el === document.activeElement")
    )
    frame.get_by_role("button", name="Section map", exact=True).click()
    frame.locator(".map-card").nth(2).get_by_role(
        "button", name="Read section", exact=False
    ).click()
    assert frame.locator("#position").get_attribute("data-passage") == "p0017"


def test_pdf_word_spacing_is_readable_but_source_and_tables_stay_exact(mounted):
    _, frame, snapshot, _, _, _ = mounted
    frame.get_by_role("button", name="Full text", exact=True).click()
    prose = frame.locator("#passage-p0001 .source-text")
    assert prose.evaluate("el => getComputedStyle(el).whiteSpace") == "pre-line"
    expected = "".join(unit["text"] for unit in snapshot["passages"][0]["units"])
    assert prose.text_content() == expected
    assert "will   test   a   small" in expected
    table = frame.locator("#passage-p0022 .source-text")
    assert table.evaluate("el => getComputedStyle(el).whiteSpace") == "pre-wrap"
    frame.locator("#passage-p0001").get_by_role(
        "button", name="Inspect source", exact=False
    ).click()
    original = frame.locator("#source-text")
    assert original.evaluate("el => getComputedStyle(el).whiteSpace") == "pre-wrap"
    assert original.text_content() == expected


def test_multi_passage_source_keyboard_close_and_neighbours(mounted):
    page, frame, _, _, _, _ = mounted
    frame.get_by_role("button", name="Section map", exact=True).click()
    trigger = frame.locator(".map-card").first.get_by_role(
        "button", name="Inspect source", exact=False
    )
    trigger.click()
    dialog = frame.get_by_role("dialog", name="Source wording")
    assert dialog.is_visible()
    assert frame.locator("#source-citations button").count() == 2
    assert frame.locator("#source-text .highlight").count() == 1
    frame.locator("#source-citations button").nth(1).click()
    assert frame.locator("#source-count").inner_text() == "2 of 24"
    assert frame.locator("#source-text .highlight").count() == 1
    frame.get_by_role("button", name="Next passage", exact=False).click()
    assert frame.locator("#source-count").inner_text() == "3 of 24"
    frame.get_by_role("button", name="Previous passage", exact=False).click()
    assert frame.locator("#source-count").inner_text() == "2 of 24"
    screenshot(page, "reader-source.png")
    frame.get_by_role("button", name="Close source", exact=True).press("Escape")
    assert not dialog.is_visible()
    assert trigger.evaluate("el => el === document.activeElement")

    frame.get_by_role("button", name="Takeaways", exact=True).click()
    goto_passage(frame, "p0012")
    scroll = frame.locator("#reading-column").evaluate("el => el.scrollTop")
    trigger = frame.locator("#passage-p0012").get_by_role(
        "button", name="Inspect source", exact=False
    )
    trigger.press("Enter")
    frame.get_by_role("button", name="Close source", exact=True).press("Enter")
    assert frame.locator("#reading-column").evaluate("el => el.scrollTop") == scroll
    assert trigger.evaluate("el => el === document.activeElement")


def test_same_passage_evidence_selection_has_distinct_feedback(mounted):
    _, frame, snapshot, _, _, _ = mounted
    frame.get_by_role("button", name="Explanation", exact=True).click()
    frame.locator("#passage-p0001").get_by_role(
        "button", name="Inspect source", exact=False
    ).click()
    source = frame.locator("#source-text")
    original = "".join(u["text"] for u in snapshot["passages"][0]["units"])
    assert source.text_content() == original
    assert source.locator(".highlight").count() == 2
    assert source.locator(".selected-evidence").count() == 1
    assert source.locator(".selected-evidence").get_attribute("data-unit") == "u0001"
    assert "Evidence 1 of 2" in frame.locator("#source-location").inner_text()

    frame.get_by_role("button", name="Show evidence 2", exact=True).click()
    assert source.text_content() == original
    assert source.locator(".highlight").count() == 2
    assert source.locator(".selected-evidence").count() == 1
    assert source.locator(".selected-evidence").get_attribute("data-unit") == "u0002"
    assert "Evidence 2 of 2" in frame.locator("#source-location").inner_text()
    assert (
        frame.get_by_role("button", name="Show evidence 2").get_attribute(
            "aria-pressed"
        )
        == "true"
    )

    frame.get_by_role("button", name="Show evidence 1", exact=True).click()
    assert source.locator(".selected-evidence").get_attribute("data-unit") == "u0001"
    frame.get_by_role("button", name="Next passage", exact=False).click()
    assert source.locator(".selected-evidence").count() == 0
    assert "surrounding context" in frame.locator("#source-location").inner_text()


def test_manual_bookmark_reopen_and_reject_wrong_version(mounted):
    _, frame, _, mount, _, _ = mounted
    frame.get_by_role("button", name="Explanation", exact=True).click()
    goto_passage(frame, "p0012")
    frame.get_by_role("button", name="Save place", exact=True).click()
    token = frame.get_by_label("Bookmark token", exact=True).input_value()
    assert token.startswith("DR1.")
    payload = json.loads(base64.b64decode(token[4:]))
    assert payload["passage"] == "p0012"
    assert payload["level"] == "explanation"
    assert "text" not in payload
    frame.get_by_role("button", name="Close bookmark", exact=True).click()

    frame = mount()
    assert (
        frame.get_by_role("button", name="Section map", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    frame.get_by_role("button", name="Restore place", exact=True).click()
    frame.get_by_label("Bookmark token", exact=True).fill(token)
    frame.locator("#bookmark-action").click()
    assert (
        frame.get_by_role("button", name="Explanation", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"

    frame.get_by_role("button", name="Restore place", exact=True).click()
    payload["fingerprint"] = "other-source-version"
    invalid = "DR1." + base64.b64encode(json.dumps(payload).encode()).decode()
    frame.get_by_label("Bookmark token", exact=True).fill(invalid)
    frame.locator("#bookmark-action").click()
    assert "different document version" in frame.get_by_role("alert").inner_text()
    frame.get_by_label("Bookmark token", exact=True).fill("DR1.bad-token")
    frame.locator("#bookmark-action").click()
    assert "valid Document Reader bookmark" in frame.get_by_role("alert").inner_text()


def test_narrow_viewport_controls_dialog_and_internal_scroll(mounted):
    page, _, _, mount, _, _ = mounted
    frame = mount(390)
    assert frame.locator("body").evaluate(
        "() => document.documentElement.scrollWidth <= innerWidth"
    )
    for control in ("Section map", "Restore place", "About this reader"):
        assert frame.get_by_role("button", name=control, exact=True).evaluate(
            "el => el.getBoundingClientRect().right <= innerWidth"
        )
    assert frame.locator("#reading-column").evaluate(
        "el => el.scrollWidth === el.clientWidth"
    )
    assert frame.get_by_label("Go to section", exact=True).is_visible()
    frame.get_by_label("Go to section", exact=True).select_option("s02")
    assert frame.locator("#position").get_attribute("data-passage") == "p0009"
    screenshot(page, "reader-mobile.png")
    frame.locator("#passage-p0009").get_by_role(
        "button", name="Inspect source", exact=False
    ).click()
    assert frame.get_by_role("dialog", name="Source wording").is_visible()
    assert frame.get_by_role("dialog", name="Source wording").evaluate(
        "el => el.getBoundingClientRect().right <= innerWidth"
    )
    screenshot(page, "reader-mobile-source.png")
    frame.get_by_role("button", name="Close source", exact=True).press("Escape")
    for label in ("Full text", "Extracts", "Explanation", "Takeaways", "Section map"):
        frame.get_by_role("button", name=label, exact=True).click()
        assert page.locator("iframe").evaluate("el => el.clientHeight") == 720
    frame.get_by_role("button", name="About this reader", exact=True).click()
    assert (
        "does not remove this saved copy"
        in frame.get_by_role("dialog", name="About this reader").inner_text()
    )
