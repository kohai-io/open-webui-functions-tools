"""Opt-in live iframe resize checks; no remount, server or provider calls.

RUN_DOCUMENT_READER_BROWSER_TESTS=1 requires Playwright and installed Chrome.
"""

import base64
import json
import os

import pytest

from test_document_reader_browser import (
    anchor_top,
    browser,
    goto_passage,
    mounted,
    render_reader,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCUMENT_READER_BROWSER_TESTS") != "1",
    reason="opt-in live iframe resize tests",
)

# Keep the passage just inside the Reader's 24px capture line. At exactly 24px,
# subpixel rounding can leave the preceding passage 0.25px across that line.
ANCHOR_TOP = 20


def settle(frame):
    frame.locator("body").evaluate(
        """() => new Promise(resolve => requestAnimationFrame(() =>
            requestAnimationFrame(() => requestAnimationFrame(resolve))))"""
    )


def mark_frame(frame):
    frame.locator("body").evaluate(
        "() => { window.readerResizeTestMarker = 'original-frame'; }"
    )


def assert_frame_geometry(page, frame, width):
    assert (
        frame.locator("body").evaluate("() => window.readerResizeTestMarker")
        == "original-frame"
    )
    assert page.locator("iframe").count() == 1
    assert page.locator("iframe").evaluate("el => el.clientHeight") == 720
    measurements = frame.locator("body").evaluate(
        """() => ({
            width: innerWidth,
            height: innerHeight,
            pageWidth: document.documentElement.scrollWidth,
            contentWidth: document.getElementById('reading-column').scrollWidth,
            contentClientWidth: document.getElementById('reading-column').clientWidth
        })"""
    )
    assert abs(measurements["width"] - width) <= 1
    assert measurements["height"] == 720
    assert measurements["pageWidth"] <= measurements["width"] + 1
    assert measurements["contentWidth"] <= measurements["contentClientWidth"] + 1


def assert_reading_position(frame):
    assert (
        frame.get_by_role("button", name="Explanation", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"
    assert abs(anchor_top(frame, "p0012") - ANCHOR_TOP) <= 4


@pytest.mark.parametrize("resize_mode", ["viewport", "parent-container"])
def test_live_resize_preserves_reading_level_anchor_expansion_and_bookmark(
    mounted, resize_mode
):
    page, frame, _, _, network, failures = mounted
    if resize_mode == "parent-container":
        page.set_viewport_size({"width": 1920, "height": 850})
        page.evaluate("() => { document.body.style.width = '1120px'; }")
        settle(frame)
    mark_frame(frame)
    frame.get_by_role("button", name="Explanation", exact=True).click()
    goto_passage(frame, "p0011", offset=ANCHOR_TOP)
    frame.locator("#passage-p0011").get_by_role("button", name="Expand here").click()
    goto_passage(frame, "p0012", offset=ANCHOR_TOP)
    settle(frame)

    content_widths = {}
    for width in (1120, 1800, 390, 1120):
        if resize_mode == "viewport":
            page.set_viewport_size({"width": width, "height": 850})
        else:
            page.evaluate(
                "width => { document.body.style.width = width + 'px'; }", width
            )
        settle(frame)
        assert_frame_geometry(page, frame, width)
        assert_reading_position(frame)
        content_widths[width] = frame.locator("#reading-content").evaluate(
            "el => el.getBoundingClientRect().width"
        )
        assert (
            frame.locator("#passage-p0011")
            .get_by_role("button", name="Collapse passage")
            .get_attribute("aria-expanded")
            == "true"
        )

        frame.get_by_role("button", name="Save place", exact=True).click()
        token = frame.get_by_label("Bookmark token", exact=True).input_value()
        bookmark = json.loads(base64.b64decode(token[4:]))
        assert bookmark["passage"] == "p0012"
        assert bookmark["level"] == "explanation"
        assert abs(bookmark["offset"] - ANCHOR_TOP) <= 4
        frame.get_by_role("button", name="Close bookmark", exact=True).click()
    assert content_widths[1800] > content_widths[1120] + 150
    assert network == []
    assert failures == []


def test_source_modal_survives_live_resize_without_overflow_or_lost_evidence(mounted):
    page, frame, snapshot, _, network, failures = mounted
    mark_frame(frame)
    frame.get_by_role("button", name="Explanation", exact=True).click()
    goto_passage(frame, "p0012", offset=ANCHOR_TOP)
    frame.locator("#passage-p0012").get_by_role(
        "button", name="Inspect source", exact=False
    ).click()
    dialog = frame.get_by_role("dialog", name="Source wording")
    expected = "".join(unit["text"] for unit in snapshot["passages"][11]["units"])
    evidence = frame.locator("#source-citations button").all_text_contents()

    dialog_widths = {}
    for width in (1120, 1800, 390, 1120):
        page.set_viewport_size({"width": width, "height": 850})
        settle(frame)
        assert_frame_geometry(page, frame, width)
        assert dialog.is_visible()
        assert frame.locator("#source-text").text_content() == expected
        assert frame.locator("#source-citations button").all_text_contents() == evidence
        assert frame.locator("#source-count").inner_text() == "12 of 24"
        geometry = dialog.evaluate(
            """el => {
                const rect = el.getBoundingClientRect();
                return {left:rect.left, top:rect.top, right:rect.right, bottom:rect.bottom,
                    width:innerWidth, height:innerHeight, dialogWidth:rect.width,
                    overflow:el.scrollWidth - el.clientWidth};
            }"""
        )
        assert geometry["left"] >= -1 and geometry["top"] >= -1
        assert geometry["right"] <= geometry["width"] + 1
        assert geometry["bottom"] <= geometry["height"] + 1
        assert geometry["overflow"] <= 1
        dialog_widths[width] = geometry["dialogWidth"]
    assert dialog_widths[1800] > dialog_widths[1120] + 100
    frame.get_by_role("button", name="Close source", exact=True).click()
    settle(frame)
    assert_reading_position(frame)
    assert network == []
    assert failures == []


def test_section_map_reflows_into_columns_and_returns_to_same_reading_position(mounted):
    page, frame, _, _, network, failures = mounted
    mark_frame(frame)
    frame.get_by_role("button", name="Explanation", exact=True).click()
    goto_passage(frame, "p0012", offset=ANCHOR_TOP)
    frame.get_by_role("button", name="Section map", exact=True).click()
    for width in (1800, 390, 1120):
        page.set_viewport_size({"width": width, "height": 850})
        settle(frame)
        assert_frame_geometry(page, frame, width)
        assert (
            frame.get_by_role("button", name="Section map", exact=True).get_attribute(
                "aria-pressed"
            )
            == "true"
        )
        boxes = frame.locator(".map-card").evaluate_all(
            "nodes => nodes.map(n => { const r=n.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width}; })"
        )
        assert len(boxes) == 3
        if width == 1800:
            assert abs(boxes[0]["y"] - boxes[1]["y"]) <= 2
            assert boxes[1]["x"] > boxes[0]["x"] + boxes[0]["width"]
        elif width == 390:
            assert all(abs(box["x"] - boxes[0]["x"]) <= 2 for box in boxes)
            assert boxes[0]["y"] < boxes[1]["y"] < boxes[2]["y"]
    frame.get_by_role("button", name="Return to explanation", exact=False).click()
    settle(frame)
    assert_reading_position(frame)
    assert network == []
    assert failures == []
