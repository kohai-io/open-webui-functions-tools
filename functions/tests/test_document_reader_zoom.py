"""Opt-in semantic zoom checks in the default opaque OWUI iframe sandbox."""

import pytest

from test_document_reader_browser import (
    anchor_top,
    browser,
    goto_passage,
    mounted,
    pytestmark,
    render_reader,
)


def pause_change(frame, label):
    frame.get_by_role("button", name=label, exact=True).evaluate(
        """button => {
            button.click();
            document.getAnimations().forEach(a => { a.pause(); a.currentTime = 160; });
        }"""
    )


def finish_change(frame):
    frame.locator("body").evaluate(
        """async () => {
            document.getAnimations().forEach(a => a.finish());
            await new Promise(r => requestAnimationFrame(r));
        }"""
    )
    assert frame.locator(".zoom-layer").count() == 0
    assert frame.locator("#reading-content").get_attribute("data-zoom-state") == "idle"


def prepare(mounted):
    page, frame, *_ = mounted
    frame.get_by_role("button", name="Explanation", exact=True).click()
    goto_passage(frame, "p0012", offset=20)
    page.emulate_media(reduced_motion="no-preference")
    # Chromium dispatches the matchMedia change asynchronously. Let the Reader's
    # legitimate preference-change cancellation finish before starting the motion
    # we intend to inspect, otherwise it can cancel the paused test animation.
    frame.locator("body").evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )
    return page, frame


@pytest.mark.parametrize("label", ["Full text", "Extracts", "Takeaways"])
def test_words_travel_while_place_and_exact_evidence_survive(mounted, label):
    _, frame = prepare(mounted)
    old_top = anchor_top(frame, "p0012")
    pause_change(frame, label)
    assert frame.locator(".zoom-layer[aria-hidden='true']").count() == 1
    assert frame.locator(".zoom-word").count() > 20
    assert frame.locator(".zoom-word").count() <= 600
    counts = frame.locator(".zoom-layer").evaluate(
        "layer=>({words:layer.querySelectorAll('.zoom-word').length,animations:layer.getAnimations({subtree:true}).length})"
    )
    assert counts["animations"] < counts["words"] * 0.6
    motion = frame.locator(".zoom-layer").evaluate(
        """layer => {
            const animations = layer.getAnimations({subtree:true});
            return {
                travelling: animations.some(a => a.effect.getKeyframes().some(k =>
                    k.transform?.includes('translate(') && !k.transform.startsWith('translate(0px,0px)'))),
                fading: animations.some(a => a.effect.getKeyframes().some(k => k.opacity === '0')),
                inert: layer.inert,
                pointerEvents: getComputedStyle(layer).pointerEvents
            };
        }"""
    )
    assert motion == {
        "travelling": True,
        "fading": True,
        "inert": True,
        "pointerEvents": "none",
    }
    assert abs(anchor_top(frame, "p0012") - old_top) <= 4
    finish_change(frame)
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"
    frame.locator("#passage-p0012").get_by_role(
        "button", name="Inspect source", exact=False
    ).first.click()
    expected = "".join(u["text"] for u in mounted[2]["passages"][11]["units"])
    assert frame.locator("#source-text").text_content() == expected
    assert frame.locator("#reading-content img, #reading-content script").count() == 0


def test_rapid_reverse_resize_and_scroll_cancel_cleanly(mounted):
    page, frame = prepare(mounted)
    pause_change(frame, "Full text")
    pause_change(frame, "Takeaways")
    assert frame.locator(".zoom-layer").count() == 1
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"
    page.set_viewport_size({"width": 390, "height": 850})
    frame.locator("body").evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )
    assert frame.locator(".zoom-layer").count() == 0
    assert abs(anchor_top(frame, "p0012") - 20) <= 4
    pause_change(frame, "Explanation")
    frame.locator("#reading-column").dispatch_event("wheel")
    assert frame.locator(".zoom-layer").count() == 0
    assert frame.locator("#reading-content.zooming").count() == 0


def test_reduced_motion_and_map_use_appropriate_transitions(mounted):
    page, frame = prepare(mounted)
    pause_change(frame, "Section map")
    assert frame.locator(".zoom-layer").count() == 0
    assert (
        frame.locator("#reading-content").get_attribute("data-zoom-state") == "running"
    )
    finish_change(frame)
    pause_change(frame, "Explanation")
    finish_change(frame)
    pause_change(frame, "Full text")
    page.emulate_media(reduced_motion="reduce")
    frame.locator("body").evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )
    assert frame.locator(".zoom-layer").count() == 0
    assert frame.locator("#reading-content.zooming").count() == 0
    pause_change(frame, "Extracts")
    assert frame.locator("body").evaluate("() => document.getAnimations().length") == 0
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"


def test_inline_expand_moves_words_and_preserves_keyboard_focus(mounted):
    _, frame = prepare(mounted)
    frame.locator("#passage-p0012").get_by_role("button", name="Expand here").evaluate(
        """button => {
            button.click();
            document.getAnimations().forEach(a => { a.pause(); a.currentTime = 160; });
        }"""
    )
    assert frame.locator(".zoom-layer").count() == 1
    finish_change(frame)
    toggle = frame.locator("#passage-p0012").get_by_role(
        "button", name="Collapse passage"
    )
    assert toggle.evaluate("el => el === document.activeElement")
    assert toggle.get_attribute("aria-expanded") == "true"
    assert abs(anchor_top(frame, "p0012") - 20) <= 4


def test_dense_viewport_uses_bounded_dissolve(mounted):
    _, frame = prepare(mounted)
    frame.locator("body").evaluate(
        """() => {
            const style=document.createElement('style');
            style.textContent='.generated-text,.source-text{font-size:1px!important;line-height:1!important}.passage{padding:0!important}.passage-meta,.passage-actions{display:none}';
            document.head.append(style);
        }"""
    )
    pause_change(frame, "Full text")
    assert frame.locator(".zoom-layer").count() == 0
    assert (
        frame.locator("#reading-content").get_attribute("data-zoom-state") == "running"
    )
    finish_change(frame)


def test_browser_without_animation_support_still_changes_levels(mounted):
    _, frame = prepare(mounted)
    frame.locator("body").evaluate("() => { Element.prototype.animate=undefined; }")
    pause_change(frame, "Full text")
    assert frame.locator(".zoom-layer").count() == 0
    assert frame.locator("#passage-p0012 .source-text").is_visible()
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"
