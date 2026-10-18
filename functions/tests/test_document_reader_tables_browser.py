"""Tables exercise the same opaque sandbox and source navigation as prose."""

from test_document_reader_browser import (
    browser,
    mounted,
    pytestmark,
    render_reader,
    screenshot,
)
from test_document_reader_tables import snapshot, xlsx_bytes
import test_document_reader as base


def mount_table(mounted, render_reader, excel=False):
    page, frame, *_ = mounted
    reader = base.load_reader()
    raw = (
        xlsx_bytes()
        if excel
        else (
            "ID,Name,Value\n"
            + "".join(f"{i:05d},Example {i},{i}\n" for i in range(1, 41))
            + '09999,"</script><img src=https://invalid.test/leak>",0\n'
        ).encode()
    )
    s = snapshot(reader, raw, "Example.xlsx" if excel else "Example.csv")
    for p in s["passages"]:
        ids = [u["id"] for u in p["units"]]
        p["generated"] = {
            "extract_ids": ids[:2],
            "explanation": [
                {
                    "text": "These source rows describe the listed examples.",
                    "evidence": ids[:2],
                }
            ],
            "takeaways": [
                {
                    "text": "Inspect the named rows for their stored values.",
                    "evidence": ids[:2],
                }
            ],
        }
    page.evaluate(
        "html=>document.querySelector('iframe').srcdoc=html", render_reader(s)
    )
    frame.locator("#reader").wait_for(state="visible")
    return page, frame, s


def test_find_table_row_then_walk_all_levels_and_inspect(mounted, render_reader):
    page, frame, s = mount_table(mounted, render_reader)
    target = next(
        p for p in s["passages"] if any("Example 29\n" in u["text"] for u in p["units"])
    )
    frame.get_by_role("button", name="Find", exact=True).click()
    frame.get_by_label("Search the complete source").fill("Example 29")
    result = frame.locator(".search-result")
    assert result.count() == 1
    assert "'Table'!A30:C30" in result.inner_text()
    result.get_by_role("button", name="Open in reader", exact=True).click()
    active = frame.locator("#passage-" + target["id"])
    assert "Example 29" in active.inner_text()
    assert active.locator(".source-grid").count() == 1
    for level in ("Explanation", "Takeaways", "Extracts", "Section map", "Full text"):
        frame.get_by_role("button", name=level, exact=True).click()
        if level != "Section map":
            assert active.get_attribute("data-active") == "true"
    assert active.is_visible()
    active.get_by_role("button", name="Inspect source ↗", exact=True).click()
    assert frame.locator("#source-text .source-grid").count() == 1
    assert "Example 29" in frame.locator("#source-text").inner_text()
    frame.get_by_role("button", name="Close source", exact=True).click()
    frame.get_by_role("button", name="Full text", exact=True).click()
    assert frame.locator(".source-grid img").count() == 0
    assert (
        "</script><img src=https://invalid.test/leak>"
        in frame.locator("#reading-content").inner_text()
    )


def test_excel_formulas_types_hidden_content_evidence_and_mobile(
    mounted, render_reader
):
    page, frame, s = mount_table(mounted, render_reader, excel=True)
    frame.get_by_role("button", name="Full text", exact=True).click()
    assert "00123" in frame.locator("#reading-content").inner_text()
    assert "No saved result" in frame.locator("#reading-content").inner_text()
    assert "Formula: =SUM(B2:B2)" in frame.locator("#reading-content").inner_text()
    assert "hidden sheet" in frame.locator("#reading-content").inner_text()
    assert "hidden" in frame.locator(".source-grid thead").first.inner_text()
    screenshot(page, "tables-desktop.png")
    page.emulate_media(color_scheme="dark")
    screenshot(page, "tables-dark.png")
    page.emulate_media(color_scheme="light")
    frame.get_by_role("button", name="Explanation", exact=True).click()
    frame.locator(".generated-item").first.get_by_role("button").click()
    frame.get_by_role("button", name="Show evidence 2", exact=True).click()
    assert "'Risks'!A2:C2" in frame.locator("#source-location").inner_text()
    assert "00123" in frame.locator("#source-text .selected-evidence").inner_text()
    frame.get_by_role("button", name="Close source", exact=True).click()
    frame.get_by_role("button", name="Full text", exact=True).click()
    page.set_viewport_size({"width": 390, "height": 850})
    frame.locator("body").evaluate(
        "()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))"
    )
    assert frame.locator("html").evaluate("e=>e.scrollWidth<=e.clientWidth+1")
    grid = frame.locator(".table-scroll").first
    assert grid.evaluate("e=>e.scrollWidth>e.clientWidth")
    grid.focus()
    grid.press("ArrowRight")
    assert grid.evaluate("e=>e===document.activeElement")
    screenshot(page, "tables-mobile.png")
