"""Overview navigation, source context, concepts and durable reading positions."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from test_document_reader import load_reader, batch_response
from test_document_reader_browser import (
    browser,
    render_reader,
    reader_snapshot,
    goto_passage,
    anchor_top,
)


def test_source_derived_hierarchy_and_context_preserve_the_source_partition():
    reader = load_reader()
    text = "# 1. Scope\n\nOnly approved pilots.\n\n# 2. Uses\n\nSubject to the following guidelines.\n\n# a. Communication\n\nReview outputs before sharing.\n\n# b. Research\n\nVerify findings.\n\n# 3. Review\n\nApproval is required.\n"
    value = reader.build_snapshot(
        text, "review.md", "file", "model", reader.Pipe.Valves()
    )
    assert "".join(u["text"] for p in value["passages"] for u in p["units"]) == text
    sections = value["sections"]
    assert [s["depth"] for s in sections] == [0, 0, 1, 1, 0]
    assert sections[2]["parent_id"] == sections[3]["parent_id"] == sections[1]["id"]
    batch = next(
        b
        for b in reader.make_batches(value, reader.Pipe.Valves())
        if b["section_id"] == sections[1]["id"]
    )
    prompt = json.loads(reader.batch_messages(batch, value)[1]["content"])
    assert prompt["child_headings"] == ["a. Communication", "b. Research"]
    assert "Never turn conditional permission" in reader.SYSTEM_PROMPT
    result = batch_response(value, batch)
    result["concepts"] = [
        {"text": "Conditional approval", "evidence": result["overview"]["evidence"]}
    ]
    assert (
        reader.validate_result(json.dumps(result), batch, value)["concepts"]
        == result["concepts"]
    )
    bad = copy.deepcopy(result)
    bad["concepts"][0]["evidence"] = ["unknown-unit"]
    with pytest.raises(reader.ReaderError, match="invalid source references"):
        reader.validate_result(json.dumps(bad), batch, value)
    bad = copy.deepcopy(result)
    bad["concepts"][0]["text"] = "x" * 71
    with pytest.raises(reader.ReaderError, match="too long"):
        reader.validate_result(json.dumps(bad), batch, value)


def test_extract_lead_in_includes_examples_without_including_the_next_topic():
    reader = load_reader()
    units = [
        {"id": f"u{i}", "text": text}
        for i, text in enumerate(
            [
                "- Communication, for example:\n",
                "  - Meeting notes\n",
                "  - Email composition\n",
                "- Research:\n",
                "  - Fact checking\n",
                "Approval is required.\n",
            ]
        )
    ]
    assert reader.contextual_extract_ids(["u0", "u5"], {"units": units}) == [
        "u0",
        "u1",
        "u2",
        "u5",
    ]
    assert reader.contextual_extract_ids(["u3"], {"units": units}) == ["u3", "u4"]


@pytest.mark.skipif(
    os.getenv("RUN_DOCUMENT_READER_BROWSER_TESTS") != "1",
    reason="opt-in local browser check",
)
def test_overview_concepts_hierarchy_resume_focus_and_source_in_opaque_iframe(
    browser, render_reader
):
    value = reader_snapshot()
    value["fingerprint"] = "a" * 64
    value["resume_scope"] = "b" * 64
    value["sections"][1].update(parent_id="s01", depth=1)
    value["overviews"][0]["concepts"] = [
        {"text": "Conditional approval", "evidence": ["u0002"]}
    ]
    source = (
        Path(__file__).resolve().parents[3] / "src/lib/utils/documentReaderBridge.ts"
    )
    compiled = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            "import ts from 'typescript';import fs from 'node:fs';process.stdout.write(ts.transpileModule(fs.readFileSync(process.argv[1],'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText.replace('export function','function'));",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=source.parents[3],
    ).stdout
    page = browser.new_page(viewport={"width": 1120, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route(
        "https://reader.test/**",
        lambda route: route.fulfill(
            body='<html class="light"><body style="margin:0"></body></html>',
            content_type="text/html",
        ),
    )
    page.goto("https://reader.test/")
    page.evaluate(compiled)
    html = render_reader(value)

    def mount():
        page.evaluate(
            """html => {
            const old = document.querySelector('iframe'); if(old)old.remove();
            const iframe=document.createElement('iframe');iframe.sandbox='allow-scripts';iframe.allowFullscreen=true;iframe.style.cssText='width:100%;height:720px;border:0;color-scheme:light';
            const bridge=createDocumentReaderBridge(html,'test-user',localStorage);
            window.addEventListener('message',e=>{if(e.source!==iframe.contentWindow)return;
                const reply=bridge(e.data,innerHeight);if(reply)iframe.contentWindow.postMessage(reply,'*');
                if(e.data?.type==='iframe:height')iframe.style.height=e.data.height+'px';});
            iframe.srcdoc=html;document.body.append(iframe);
        }""",
            html,
        )
        frame = page.locator("iframe").content_frame
        frame.locator("#reader").wait_for(state="visible")
        frame.locator("#resume-note").filter(
            has_text="saved on this browser"
        ).wait_for()
        frame.locator("body").evaluate(
            "() => new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))"
        )
        return frame

    frame = mount()
    assert frame.locator("#level-controls button").all_text_contents() == [
        "1Section map",
        "2Takeaways",
        "3Explanation",
        "4Extracts",
        "5Full text",
    ]
    assert frame.locator(".document-overview").inner_text().startswith("START HERE")
    assert (
        frame.locator('.map-group .map-children [data-map-section="s02"]').count() == 1
    )
    frame.get_by_role("button", name="Conditional approval", exact=True).first.click()
    assert (
        frame.get_by_role("button", name="Takeaways", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    frame.get_by_role("button", name="Explanation", exact=True).click()
    goto_passage(frame, "p0012", offset=20)
    frame.locator("#passage-p0012").get_by_role("button", name="Expand here").click()
    goto_passage(frame, "p0012", offset=20)
    page.wait_for_function(
        'Object.keys(localStorage).some(k=>k.startsWith("owui:test-user:") && JSON.parse(localStorage[k]).passage==="p0012")'
    )
    frame = mount()
    assert (
        frame.get_by_role("button", name="Explanation", exact=True).get_attribute(
            "aria-pressed"
        )
        == "true"
    )
    assert frame.locator("#position").get_attribute("data-passage") == "p0012"
    assert abs(anchor_top(frame, "p0012") - 20) < 5
    assert (
        frame.locator("#passage-p0012")
        .get_by_role("button", name="Collapse passage")
        .is_visible()
    )
    assert frame.locator("body").evaluate(
        "() => {try {void parent.document;return false;}catch(_){return true;}}"
    )
    assert page.locator("iframe").get_attribute("sandbox") == "allow-scripts"
    frame.get_by_role("button", name="Focus reading", exact=True).click()
    frame.get_by_role("button", name="Exit focus", exact=True).wait_for()
    assert frame.locator("#reader").evaluate("el=>document.fullscreenElement===el")
    frame.get_by_role("button", name="Exit focus", exact=True).click()
    frame.get_by_role("button", name="Focus reading", exact=True).wait_for()
    assert errors == []
    page.close()
