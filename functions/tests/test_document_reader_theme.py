"""Opt-in native theme inheritance tests for OWUI's sandboxed Reader embed.

RUN_DOCUMENT_READER_BROWSER_TESTS=1 requires Playwright and installed Chrome.
Use native browser preferences: Playwright's media emulation overrides child
media queries too and would mask the browser's iframe color-scheme inheritance.
"""

import os
import re
from pathlib import Path

import pytest

from test_document_reader_browser import reader_snapshot, render_reader


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCUMENT_READER_BROWSER_TESTS") != "1",
    reason="opt-in native iframe theme tests",
)


def host_iframe_css():
    component = (
        Path(__file__).resolve().parents[3]
        / "src/lib/components/common/FullHeightIframe.svelte"
    )
    source = component.read_text(encoding="utf-8")
    blocks = re.findall(r"<style(?:\s[^>]*)?>([\s\S]*?)</style>", source)
    assert blocks, "The host iframe must propagate the resolved OWUI theme."
    # Apply the authored component CSS, removing only Svelte's global marker.
    return re.sub(r":global\(([^)]+)\)", r"\1", "\n".join(blocks))


@pytest.mark.parametrize(
    "browser_dark", [False, True], ids=["light-browser", "dark-browser"]
)
@pytest.mark.parametrize(
    "host_css,allow_same_origin",
    [(True, False), (False, True)],
    ids=["opaque-host-css", "same-origin-legacy-host"],
)
def test_reader_follows_app_theme_on_open_toggle_and_reopen(
    render_reader, browser_dark, host_css, allow_same_origin
):
    from playwright.sync_api import sync_playwright

    css = host_iframe_css()
    html = render_reader(reader_snapshot())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--force-dark-mode"] if browser_dark else [],
        )
        # Literal "null" disables Playwright's default forced-light emulation.
        page = browser.new_page(color_scheme="null")
        network, errors = [], []
        page.on("request", lambda request: network.append(request.url))
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            assert (
                page.evaluate("matchMedia('(prefers-color-scheme: dark)').matches")
                is browser_dark
            )
            page.set_content("<!doctype html><html><head></head><body></body></html>")
            if host_css:
                page.add_style_tag(content=css)

            def mount(app_dark):
                page.evaluate(
                    """({html, dark, sameOrigin}) => {
                        document.documentElement.className = dark ? 'dark' : 'light';
                        document.body.replaceChildren();
                        const iframe = document.createElement('iframe');
                        iframe.sandbox = 'allow-scripts allow-downloads' + (sameOrigin ? ' allow-same-origin' : '');
                        iframe.style.cssText = 'width:100%;height:720px;border:0';
                        iframe.srcdoc = html;
                        document.body.append(iframe);
                    }""",
                    {"html": html, "dark": app_dark, "sameOrigin": allow_same_origin},
                )
                frame = page.locator("iframe").content_frame
                frame.locator("#reader").wait_for(state="visible")
                return frame

            def check_theme(frame, dark):
                # Allow both the media query and any Reader change listener to run.
                frame.locator("body").evaluate(
                    "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
                )
                result = frame.locator("body").evaluate(
                    """() => ({
                        origin: window.origin,
                        dark: matchMedia('(prefers-color-scheme: dark)').matches,
                        readerTheme: document.documentElement.dataset.theme,
                        background: getComputedStyle(document.body).backgroundColor,
                        isolated: (() => {
                            try { void parent.document.documentElement; return false; }
                            catch (_) { return true; }
                        })()
                    })"""
                )
                assert result["origin"] == "null"
                assert result["isolated"] is not allow_same_origin
                assert result["dark"] is (dark if host_css else browser_dark)
                assert result["readerTheme"] == ("dark" if dark else "light")
                assert page.locator("iframe").evaluate(
                    "el => getComputedStyle(el).colorScheme"
                ) == (("dark" if dark else "light") if host_css else "normal")
                assert (
                    page.evaluate("matchMedia('(prefers-color-scheme: dark)').matches")
                    is browser_dark
                )  # The app toggle did not change the browser preference.
                return result["background"]

            # Start with OWUI explicitly set opposite to the native browser theme.
            app_dark = not browser_dark
            frame = mount(app_dark)
            initial_background = check_theme(frame, app_dark)
            frame.locator("body").evaluate(
                "() => { window.readerThemeTestMarker = 'same-frame'; }"
            )
            frame.get_by_role("button", name="Full text", exact=True).click()
            frame.locator("body").evaluate(
                """() => {
                    window.dispatchEvent(new PageTransitionEvent('pagehide', {persisted:true}));
                    window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted:true}));
                }"""
            )
            page.evaluate(
                "dark => document.documentElement.className = dark ? 'dark' : 'light'",
                not app_dark,
            )
            toggled_background = check_theme(frame, not app_dark)
            assert initial_background != toggled_background
            assert (
                frame.locator("body").evaluate("() => window.readerThemeTestMarker")
                == "same-frame"
            )
            assert (
                frame.get_by_role("button", name="Full text", exact=True).get_attribute(
                    "aria-pressed"
                )
                == "true"
            )

            page.evaluate(
                "dark => document.documentElement.className = dark ? 'dark' : 'light'",
                app_dark,
            )
            assert check_theme(frame, app_dark) == initial_background
            reopened = mount(app_dark)
            assert check_theme(reopened, app_dark) == initial_background
            assert (
                reopened.get_by_role(
                    "button", name="Section map", exact=True
                ).get_attribute("aria-pressed")
                == "true"
            )
            assert network == []
            assert errors == []
        finally:
            browser.close()
