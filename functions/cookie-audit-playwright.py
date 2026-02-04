"""
title: Cookie Compliance Audit (Playwright)
author: Open WebUI
version: 1.0.2
license: MIT
description: Advanced cookie compliance audit using Playwright for real browser rendering. Captures HTTP and JavaScript cookies, detects trackers using EasyPrivacy lists, analyzes third-party requests, and generates ICO PECR compliance reports. Based on EDPB Website Auditing Tool (EUPL-1.2).
requirements: playwright, aiohttp, pydantic, adblockparser
"""

# Based on EDPB Website Auditing Tool (https://code.europa.eu/edpb/website-auditing-tool)
# Original work licensed under EUPL-1.2
# Cookie analysis logic adapted from:
#   - app/handlers/collectors/har-collector.ts
#   - app/handlers/cards/cookie-card.ts
#   - app/handlers/cards/beacon-card.ts
#   - app/handlers/cards/traffic-card.ts

import asyncio
import json
import logging
import re
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Awaitable, Set
from urllib.parse import urljoin, urlparse, parse_qs
import ipaddress
import socket

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Pipe:
    """Cookie Compliance Audit Pipeline using Playwright - ICO/PECR focused
    
    Based on EDPB Website Auditing Tool analysis patterns.
    """

    # Class-level cache to prevent duplicate runs
    _active_audits = {}
    
    # EasyPrivacy patterns cache
    _easyprivacy_rules = None
    _fanboy_rules = None

    class Valves(BaseModel):
        """Configuration options for the cookie audit"""

        TIMEOUT_SECONDS: int = Field(
            default=120,
            description="Maximum time for the entire audit",
        )
        PAGE_TIMEOUT_MS: int = Field(
            default=30000,
            description="Timeout for page load in milliseconds",
        )
        WAIT_AFTER_LOAD_MS: int = Field(
            default=3000,
            description="Time to wait after page load for JS cookies to be set",
        )
        PLAYWRIGHT_WS_URL: str = Field(
            default="ws://10.100.1.144:3000",
            description="WebSocket URL for remote Playwright browser (e.g., ws://host:3000). Leave empty to use local browser.",
        )
        HEADLESS: bool = Field(
            default=True,
            description="Run browser in headless mode (only applies to local browser)",
        )
        USER_AGENT: str = Field(
            default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            description="User agent string for the browser",
        )
        ICO_GUIDANCE_URL: str = Field(
            default="https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/cookies-and-similar-technologies/",
            description="ICO guidance URL for cookie compliance citations",
        )
        ENABLE_TRACKER_DETECTION: bool = Field(
            default=True,
            description="Enable tracker detection using EasyPrivacy/Fanboy lists",
        )
        ENABLE_LLM_ANALYSIS: bool = Field(
            default=False,
            description="Use LLM to analyze cookie policy quality",
        )
        LLM_MODEL: str = Field(
            default="gpt-4o-mini",
            description="LLM model for analysis",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="OpenAI API key for LLM analysis",
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API base URL",
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable verbose logging",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Cookie Compliance Audit (Playwright)"

        # ICO PECR Cookie Requirements
        self.ico_requirements = {
            "consent": {
                "name": "Cookie Consent",
                "description": "Users must give consent before non-essential cookies are set",
                "ico_ref": "PECR Regulation 6",
                "critical": True,
            },
            "information": {
                "name": "Clear Information",
                "description": "Users must be told clearly what cookies are used and why",
                "ico_ref": "PECR Regulation 6(2)(a)",
                "critical": True,
            },
            "control": {
                "name": "User Control",
                "description": "Users must be able to refuse non-essential cookies",
                "ico_ref": "PECR Regulation 6",
                "critical": True,
            },
        }

        # Cookie classification patterns (from EDPB WAT cookie-card.ts logic)
        self.cookie_categories = {
            "essential": {
                "patterns": [
                    re.compile(r"^(session|csrf|xsrf|token|auth|login|cart|basket|checkout|security).*$", re.IGNORECASE),
                    re.compile(r"^(consent|cookie_consent|cookieconsent|cc_cookie|gdpr|accepted_cookies|cookie_notice|cookies_accepted).*$", re.IGNORECASE),
                    re.compile(r"^(phpsessid|jsessionid|asp\.net_sessionid|laravel_session|wordpress_logged_in|wp-settings).*$", re.IGNORECASE),
                    # Infrastructure/CDN cookies
                    re.compile(r"^(AWSALB|AWSALBCORS|AWSELB|AWSELBCORS).*$", re.IGNORECASE),
                    re.compile(r"^(__cf_bm|_cfuvid|cf_clearance|__cfruid|__cflb).*$", re.IGNORECASE),
                    re.compile(r"^(JSESSIONID|SERVERID|ROUTEID|BACKEND).*$", re.IGNORECASE),
                    re.compile(r"^(incap_ses|visid_incap|nlbi_).*$", re.IGNORECASE),
                    re.compile(r"^(ak_bmsc|bm_sv|bm_sz).*$", re.IGNORECASE),
                ],
                "description": "Strictly necessary for website functionality",
                "requires_consent": False,
                "ico_category": "Strictly necessary",
            },
            "analytics": {
                "patterns": [
                    re.compile(r"^(_ga|_gid|_gat|__utma|__utmb|__utmc|__utmz|__utmv).*$", re.IGNORECASE),
                    re.compile(r"^(_pk_id|_pk_ses|_pk_ref|_pk_cvar).*$", re.IGNORECASE),
                    re.compile(r"^(_hjid|_hjSessionUser|_hjSession|_hjAbsoluteSessionInProgress).*$", re.IGNORECASE),
                    re.compile(r"^(amplitude|mixpanel|segment|heap|fullstory).*$", re.IGNORECASE),
                    re.compile(r"^(ajs_user_id|ajs_anonymous_id).*$", re.IGNORECASE),
                ],
                "description": "Used to collect anonymous usage statistics",
                "requires_consent": True,
                "ico_category": "Analytics",
            },
            "marketing": {
                "patterns": [
                    re.compile(r"^(_fbp|_fbc|fr|tr)$", re.IGNORECASE),
                    re.compile(r"^(IDE|DSID|__gads|__gpi|_gcl_au|_gcl_aw).*$", re.IGNORECASE),
                    re.compile(r"^(NID|ANID|CONSENT|1P_JAR|AID|APISID|HSID|SAPISID|SID|SIDCC|SSID)$", re.IGNORECASE),
                    re.compile(r"^(_uetsid|_uetvid|MUID|_clck|_clsk).*$", re.IGNORECASE),
                    re.compile(r"^(lidc|bcookie|bscookie|li_gc|li_mc).*$", re.IGNORECASE),
                    re.compile(r"^(personalization_id|guest_id|ct0|twid).*$", re.IGNORECASE),
                ],
                "description": "Used for advertising and tracking across websites",
                "requires_consent": True,
                "ico_category": "Marketing/Advertising",
            },
            "functional": {
                "patterns": [
                    re.compile(r"^(lang|locale|language|i18n|timezone|tz).*$", re.IGNORECASE),
                    re.compile(r"^(theme|dark_mode|font_size|accessibility).*$", re.IGNORECASE),
                    re.compile(r"^(recently_viewed|wishlist|compare|preferences).*$", re.IGNORECASE),
                ],
                "description": "Enhance functionality and personalization",
                "requires_consent": True,
                "ico_category": "Functional",
            },
        }

        # Tracker detection patterns (simplified EasyPrivacy-style)
        self.tracker_patterns = [
            # Google Analytics
            re.compile(r"google-analytics\.com", re.IGNORECASE),
            re.compile(r"googletagmanager\.com", re.IGNORECASE),
            re.compile(r"doubleclick\.net", re.IGNORECASE),
            # Facebook
            re.compile(r"facebook\.com/tr", re.IGNORECASE),
            re.compile(r"connect\.facebook\.net", re.IGNORECASE),
            # Microsoft/Bing
            re.compile(r"bat\.bing\.com", re.IGNORECASE),
            re.compile(r"clarity\.ms", re.IGNORECASE),
            # Hotjar
            re.compile(r"hotjar\.com", re.IGNORECASE),
            # LinkedIn
            re.compile(r"linkedin\.com/px", re.IGNORECASE),
            re.compile(r"snap\.licdn\.com", re.IGNORECASE),
            # Twitter
            re.compile(r"ads-twitter\.com", re.IGNORECASE),
            re.compile(r"analytics\.twitter\.com", re.IGNORECASE),
            # Generic tracking patterns
            re.compile(r"/pixel\.", re.IGNORECASE),
            re.compile(r"/beacon", re.IGNORECASE),
            re.compile(r"/collect\?", re.IGNORECASE),
            re.compile(r"track(ing)?\..*\.(com|net|io)", re.IGNORECASE),
            re.compile(r"analytics\.", re.IGNORECASE),
            re.compile(r"telemetry\.", re.IGNORECASE),
        ]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Main entry point for the cookie audit"""

        messages = body.get("messages", [])
        if not messages:
            yield "Please provide a website URL to audit for cookie compliance."
            return

        # Only process if the last message is from the user
        last_message_obj = messages[-1]
        message_role = last_message_obj.get("role", "unknown")
        last_message = last_message_obj.get("content", "")

        log.info(f"[COOKIE AUDIT PW] Received message with role={message_role}, content_preview={last_message[:100]}")

        if message_role != "user":
            log.info(f"[COOKIE AUDIT PW] Ignoring non-user message (role={message_role})")
            return

        # Ignore auto-generated suggestion requests from Open WebUI
        if "### Task:" in last_message or "follow-up questions" in last_message.lower():
            log.info("[COOKIE AUDIT PW] Ignoring auto-generated suggestion request")
            return

        target_url = self._extract_url(last_message)

        if not target_url:
            yield "Please provide a valid website URL to audit. Example: `https://example.com`"
            return

        # Ensure URL has scheme
        if not target_url.startswith(("http://", "https://")):
            target_url = f"https://{target_url}"

        # Security: Block internal/private URLs (SSRF protection)
        if not self._is_safe_url(target_url):
            yield "❌ **Security Error:** Cannot audit internal, private, or localhost URLs.\n"
            return

        # Deduplication: Check if this exact audit is already running
        user_id = __user__.get("id") if __user__ else "unknown"
        audit_key = f"{user_id}:{target_url}"

        if audit_key in Pipe._active_audits:
            log.warning(f"[COOKIE AUDIT PW] Duplicate audit request detected for {target_url}, ignoring")
            return

        # Mark this audit as active
        Pipe._active_audits[audit_key] = datetime.now()
        log.info(f"[COOKIE AUDIT PW] Starting new audit for {target_url} (active_audits: {len(Pipe._active_audits)})")

        try:
            yield f"# 🍪 Cookie Compliance Audit (Playwright)\n\n"
            yield f"**Target:** {target_url}\n"
            yield f"**Standard:** ICO PECR Guidelines\n"
            yield f"**Engine:** Playwright (Real Browser)\n"
            yield f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "Launching browser...", "done": False}}
                )

            async with asyncio.timeout(self.valves.TIMEOUT_SECONDS):
                # Run the audit with Playwright
                audit_data = await self.run_playwright_audit(target_url, __event_emitter__)

                # Generate report
                async for chunk in self.generate_report(audit_data):
                    yield chunk

        except asyncio.TimeoutError:
            yield f"\n\n⏱️ **Audit timed out** after {self.valves.TIMEOUT_SECONDS} seconds.\n"
        except Exception as e:
            log.exception(f"Cookie audit error: {e}")
            yield f"\n\n❌ **Error during audit:** {str(e)}\n"
        finally:
            # Clean up: remove from active audits
            if audit_key in Pipe._active_audits:
                del Pipe._active_audits[audit_key]
                log.info(f"[COOKIE AUDIT PW] Completed audit for {target_url} (active_audits: {len(Pipe._active_audits)})")
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "Cookie audit complete", "done": True}}
                )

    async def run_playwright_audit(
        self,
        url: str,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Run the audit using Playwright with HAR capture
        
        Based on EDPB WAT mcp-server.ts auditUrl function
        
        Note: Uses sync Playwright in a thread pool to avoid Windows asyncio subprocess issues
        """
        import concurrent.futures
        
        # Run Playwright in a thread pool to avoid Windows asyncio subprocess issues
        # WindowsSelectorEventLoop doesn't support subprocess_exec which Playwright needs
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self._run_playwright_sync,
                url,
            )
        
        # Unpack result and run async post-processing
        (
            cookies, local_storage, session_storage, requests_log,
            third_party_hosts, page_title, har_data
        ) = result
        
        # Analyze trackers from requests (can be done async)
        trackers = []
        if self.valves.ENABLE_TRACKER_DETECTION:
            main_domain = urlparse(url).netloc
            for req in requests_log:
                req_url = req.get("url", "")
                for pattern in self.tracker_patterns:
                    if pattern.search(req_url):
                        trackers.append({
                            "url": req_url,
                            "pattern": pattern.pattern,
                            "type": self._get_tracker_type(req_url),
                        })
                        break
        
        # Analyze collected data
        if event_emitter:
            await event_emitter(
                {"type": "status", "data": {"description": "Analyzing cookies and trackers...", "done": False}}
            )
        
        # Classify cookies
        cookie_analysis = self._analyze_cookies(cookies)
        
        # Analyze HAR for additional insights
        har_analysis = self._analyze_har(har_data) if har_data else {}
        
        # Check for consent (from page snapshot stored in har_data)
        consent_detected = har_data.get("_consent_detected", False) if har_data else False
        cookie_policy_url = har_data.get("_cookie_policy_url") if har_data else None
        
        # Assess ICO compliance
        compliance = self._assess_compliance(
            cookies=cookies,
            cookie_analysis=cookie_analysis,
            consent_detected=consent_detected,
            cookie_policy_url=cookie_policy_url,
            trackers=trackers,
        )
        
        return {
            "url": url,
            "page_title": page_title,
            "cookies": cookies,
            "cookie_analysis": cookie_analysis,
            "local_storage": local_storage,
            "session_storage": session_storage,
            "requests": requests_log,
            "trackers": trackers,
            "third_party_hosts": list(third_party_hosts),
            "consent_detected": consent_detected,
            "cookie_policy_url": cookie_policy_url,
            "har_analysis": har_analysis,
            "compliance": compliance,
            "audit_time": datetime.now().isoformat(),
        }

    def _run_playwright_sync(self, url: str) -> tuple:
        """Run Playwright synchronously in a thread pool
        
        This avoids Windows asyncio subprocess issues with WindowsSelectorEventLoop
        """
        from playwright.sync_api import sync_playwright
        
        har_data = None
        cookies = []
        local_storage = {}
        session_storage = {}
        requests_log = []
        third_party_hosts = set()
        page_title = ""
        consent_detected = False
        cookie_policy_url = None
        
        with sync_playwright() as p:
            # Connect to remote browser or launch local
            if self.valves.PLAYWRIGHT_WS_URL:
                log.info(f"[COOKIE AUDIT PW] Connecting to remote browser: {self.valves.PLAYWRIGHT_WS_URL}")
                browser = p.chromium.connect(self.valves.PLAYWRIGHT_WS_URL)
            else:
                log.info("[COOKIE AUDIT PW] Launching local browser")
                browser = p.chromium.launch(headless=self.valves.HEADLESS)
            
            try:
                # Create temp file for HAR
                har_path = tempfile.mktemp(suffix=".har")
                
                # Create context with HAR recording (like EDPB WAT)
                context = browser.new_context(
                    user_agent=self.valves.USER_AGENT,
                    record_har_path=har_path,
                    record_har_content="omit",  # Don't record response bodies
                    ignore_https_errors=True,
                )
                
                # Track requests for tracker detection
                page = context.new_page()
                
                # Set up request interception for tracker detection
                main_domain = urlparse(url).netloc
                
                def handle_request(request):
                    request_url = request.url
                    requests_log.append({
                        "url": request_url,
                        "method": request.method,
                        "resource_type": request.resource_type,
                    })
                    
                    # Check for third-party hosts
                    request_domain = urlparse(request_url).netloc
                    if request_domain and request_domain != main_domain:
                        third_party_hosts.add(request_domain)
                
                page.on("request", handle_request)
                
                # Navigate to page with retry on failure
                log.info(f"[COOKIE AUDIT PW] Navigating to {url}")
                response = None
                navigation_error = None
                
                # Try networkidle first, fallback to load on failure
                for wait_strategy in ["networkidle", "load", "domcontentloaded"]:
                    try:
                        response = page.goto(url, wait_until=wait_strategy, timeout=self.valves.PAGE_TIMEOUT_MS)
                        if response:
                            log.info(f"[COOKIE AUDIT PW] Page loaded with status {response.status} (wait_until={wait_strategy})")
                            navigation_error = None
                            break
                    except Exception as e:
                        navigation_error = str(e)
                        log.warning(f"[COOKIE AUDIT PW] Navigation failed with {wait_strategy}: {e}")
                        if "net::ERR_" in str(e) or "timeout" in str(e).lower():
                            continue  # Try next strategy
                        else:
                            raise  # Re-raise non-network errors
                
                if navigation_error and not response:
                    raise Exception(f"Failed to load page after retries: {navigation_error}")
                
                # Wait for JS cookies to be set
                page.wait_for_timeout(self.valves.WAIT_AFTER_LOAD_MS)
                
                # Get page title
                page_title = page.title()
                
                # Collect cookies (HTTP + JS)
                cookies = context.cookies()
                log.info(f"[COOKIE AUDIT PW] Found {len(cookies)} cookies")
                
                # Collect localStorage and sessionStorage
                try:
                    local_storage = page.evaluate("() => Object.assign({}, localStorage)")
                    session_storage = page.evaluate("() => Object.assign({}, sessionStorage)")
                except Exception as e:
                    log.warning(f"[COOKIE AUDIT PW] Could not access storage: {e}")
                
                # Detect consent mechanism
                consent_detected = self._detect_consent_mechanism_sync(page)
                
                # Look for cookie policy link
                cookie_policy_url = self._find_cookie_policy_link_sync(page, url)
                
                # Close context to finalize HAR
                context.close()
                
                # Read HAR file
                if os.path.exists(har_path):
                    with open(har_path, 'r', encoding='utf-8') as f:
                        har_data = json.load(f)
                    # Store consent/policy in har_data for return
                    har_data["_consent_detected"] = consent_detected
                    har_data["_cookie_policy_url"] = cookie_policy_url
                    os.remove(har_path)
                    log.info(f"[COOKIE AUDIT PW] HAR captured with {len(har_data.get('log', {}).get('entries', []))} entries")
                
            finally:
                browser.close()
        
        return (
            cookies, local_storage, session_storage, requests_log,
            third_party_hosts, page_title, har_data
        )

    def _detect_consent_mechanism_sync(self, page) -> bool:
        """Detect if a cookie consent mechanism is present (sync version)"""
        consent_selectors = [
            '[class*="cookie-consent"]',
            '[class*="cookie-banner"]',
            '[class*="cookie-notice"]',
            '[class*="consent-banner"]',
            '[class*="gdpr"]',
            '[id*="cookie-consent"]',
            '[id*="cookie-banner"]',
            '[id*="gdpr"]',
            '[id*="onetrust"]',
            '[class*="onetrust"]',
            '[id*="cookiebot"]',
            '[class*="cookiebot"]',
            '[id*="CybotCookiebotDialog"]',
            '[class*="cc-banner"]',
            '[class*="cc-window"]',
            '[id*="sp_message_container"]',
            '[id*="cmp"]',
            '[class*="cmp-"]',
            '[data-testid*="cookie"]',
            '[aria-label*="cookie"]',
            '[aria-label*="consent"]',
        ]
        
        for selector in consent_selectors:
            try:
                element = page.query_selector(selector)
                if element:
                    is_visible = element.is_visible()
                    if is_visible:
                        log.info(f"[COOKIE AUDIT PW] Consent mechanism detected: {selector}")
                        return True
            except Exception:
                continue
        
        # Also check for consent-related text
        try:
            page_text = page.inner_text("body")
            consent_phrases = [
                "we use cookies",
                "this website uses cookies",
                "accept cookies",
                "cookie preferences",
                "manage cookies",
                "cookie settings",
            ]
            for phrase in consent_phrases:
                if phrase.lower() in page_text.lower():
                    log.info(f"[COOKIE AUDIT PW] Consent text detected: '{phrase}'")
                    return True
        except Exception:
            pass
        
        return False

    def _find_cookie_policy_link_sync(self, page, base_url: str) -> Optional[str]:
        """Find cookie policy page link (sync version)"""
        policy_patterns = [
            r"cookie.*policy",
            r"cookie.*notice",
            r"privacy.*cookie",
            r"cookies",
        ]
        
        try:
            links = page.query_selector_all("a[href]")
            for link in links:
                href = link.get_attribute("href")
                text = link.inner_text()
                
                if not href:
                    continue
                
                href_lower = href.lower()
                text_lower = text.lower()
                
                for pattern in policy_patterns:
                    if re.search(pattern, href_lower) or re.search(pattern, text_lower):
                        # Make absolute URL
                        if href.startswith("/"):
                            href = urljoin(base_url, href)
                        elif not href.startswith("http"):
                            href = urljoin(base_url, href)
                        log.info(f"[COOKIE AUDIT PW] Cookie policy found: {href}")
                        return href
        except Exception as e:
            log.warning(f"[COOKIE AUDIT PW] Error finding cookie policy: {e}")
        
        return None

    def _get_tracker_type(self, url: str) -> str:
        """Determine tracker type from URL"""
        url_lower = url.lower()
        if "google" in url_lower:
            return "Google Analytics/Ads"
        elif "facebook" in url_lower or "fb.com" in url_lower:
            return "Facebook/Meta"
        elif "bing" in url_lower or "clarity" in url_lower:
            return "Microsoft"
        elif "hotjar" in url_lower:
            return "Hotjar"
        elif "linkedin" in url_lower:
            return "LinkedIn"
        elif "twitter" in url_lower:
            return "Twitter/X"
        elif "analytics" in url_lower:
            return "Analytics"
        elif "pixel" in url_lower or "beacon" in url_lower:
            return "Tracking Pixel"
        else:
            return "Unknown Tracker"

    def _analyze_cookies(self, cookies: List[Dict]) -> Dict[str, Any]:
        """Analyze and classify cookies
        
        Based on EDPB WAT cookie-card.ts inspectCookies logic
        """
        by_category = {
            "essential": [],
            "analytics": [],
            "marketing": [],
            "functional": [],
            "unknown": [],
        }
        
        for cookie in cookies:
            name = cookie.get("name", "")
            classified = False
            
            for category, config in self.cookie_categories.items():
                for pattern in config["patterns"]:
                    if pattern.match(name):
                        by_category[category].append({
                            "name": name,
                            "domain": cookie.get("domain", ""),
                            "path": cookie.get("path", "/"),
                            "expires": cookie.get("expires", -1),
                            "httpOnly": cookie.get("httpOnly", False),
                            "secure": cookie.get("secure", False),
                            "sameSite": cookie.get("sameSite", "None"),
                            "value_length": len(str(cookie.get("value", ""))),
                            "category": category,
                            "requires_consent": config["requires_consent"],
                            "ico_category": config["ico_category"],
                        })
                        classified = True
                        break
                if classified:
                    break
            
            if not classified:
                by_category["unknown"].append({
                    "name": name,
                    "domain": cookie.get("domain", ""),
                    "path": cookie.get("path", "/"),
                    "expires": cookie.get("expires", -1),
                    "httpOnly": cookie.get("httpOnly", False),
                    "secure": cookie.get("secure", False),
                    "sameSite": cookie.get("sameSite", "None"),
                    "value_length": len(str(cookie.get("value", ""))),
                    "category": "unknown",
                    "requires_consent": True,  # Assume consent required for unknown
                    "ico_category": "Unknown",
                })
        
        # Calculate statistics
        total = len(cookies)
        requiring_consent = sum(
            len(by_category[cat]) for cat in ["analytics", "marketing", "functional", "unknown"]
        )
        
        return {
            "total": total,
            "by_category": by_category,
            "requiring_consent": requiring_consent,
            "essential_count": len(by_category["essential"]),
            "analytics_count": len(by_category["analytics"]),
            "marketing_count": len(by_category["marketing"]),
            "functional_count": len(by_category["functional"]),
            "unknown_count": len(by_category["unknown"]),
        }

    def _analyze_har(self, har_data: Dict) -> Dict[str, Any]:
        """Analyze HAR data for additional insights
        
        Based on EDPB WAT har-collector.ts parseHar logic
        """
        if not har_data:
            return {}
        
        entries = har_data.get("log", {}).get("entries", [])
        
        # Count request types
        request_types = {}
        total_size = 0
        
        for entry in entries:
            request = entry.get("request", {})
            response = entry.get("response", {})
            
            # Resource type
            resource_type = entry.get("_resourceType", "other")
            request_types[resource_type] = request_types.get(resource_type, 0) + 1
            
            # Response size
            content = response.get("content", {})
            total_size += content.get("size", 0)
        
        return {
            "total_requests": len(entries),
            "request_types": request_types,
            "total_size_bytes": total_size,
        }

    def _assess_compliance(
        self,
        cookies: List[Dict],
        cookie_analysis: Dict,
        consent_detected: bool,
        cookie_policy_url: Optional[str],
        trackers: List[Dict],
    ) -> Dict[str, Any]:
        """Assess ICO PECR compliance"""
        
        issues = []
        recommendations = []
        
        # Check for non-essential cookies without consent mechanism
        non_essential_count = cookie_analysis["requiring_consent"]
        
        if non_essential_count > 0 and not consent_detected:
            issues.append({
                "severity": "high",
                "issue": f"Found {non_essential_count} non-essential cookies but no consent mechanism detected",
                "ico_ref": "PECR Regulation 6",
            })
            recommendations.append("Implement a cookie consent mechanism before setting non-essential cookies")
        
        # Check for cookie policy
        if not cookie_policy_url:
            issues.append({
                "severity": "medium",
                "issue": "No cookie policy page detected",
                "ico_ref": "PECR Regulation 6(2)(a)",
            })
            recommendations.append("Create a clear cookie policy explaining what cookies are used and why")
        
        # Check for trackers
        if trackers:
            tracker_types = set(t["type"] for t in trackers)
            issues.append({
                "severity": "medium",
                "issue": f"Found {len(trackers)} tracking requests ({', '.join(tracker_types)})",
                "ico_ref": "PECR Regulation 6",
            })
            recommendations.append("Ensure all tracking scripts only load after user consent")
        
        # Check for marketing cookies
        if cookie_analysis["marketing_count"] > 0:
            issues.append({
                "severity": "medium",
                "issue": f"Found {cookie_analysis['marketing_count']} marketing/advertising cookies",
                "ico_ref": "PECR Regulation 6",
            })
            recommendations.append("Marketing cookies require explicit user consent before being set")
        
        # Check for unknown cookies
        if cookie_analysis["unknown_count"] > 0:
            issues.append({
                "severity": "low",
                "issue": f"Found {cookie_analysis['unknown_count']} unclassified cookies",
                "ico_ref": "PECR Regulation 6(2)(a)",
            })
            recommendations.append("Document all cookies in your cookie policy with clear explanations")
        
        # Calculate overall score
        high_issues = len([i for i in issues if i["severity"] == "high"])
        medium_issues = len([i for i in issues if i["severity"] == "medium"])
        low_issues = len([i for i in issues if i["severity"] == "low"])
        
        if high_issues > 0:
            overall = "Non-Compliant"
            score = max(0, 40 - (high_issues * 20) - (medium_issues * 10) - (low_issues * 5))
        elif medium_issues > 0:
            overall = "Needs Improvement"
            score = max(40, 70 - (medium_issues * 10) - (low_issues * 5))
        elif low_issues > 0:
            overall = "Mostly Compliant"
            score = max(70, 90 - (low_issues * 5))
        else:
            overall = "Compliant"
            score = 100
        
        return {
            "overall": overall,
            "score": score,
            "issues": issues,
            "recommendations": recommendations,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "low_issues": low_issues,
        }

    async def generate_report(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate the compliance report"""
        
        yield "---\n\n"
        yield "## 📊 Summary\n\n"
        
        compliance = data.get("compliance", {})
        cookie_analysis = data.get("cookie_analysis", {})
        
        # Overall status
        overall = compliance.get("overall", "Unknown")
        score = compliance.get("score", 0)
        
        status_emoji = {
            "Compliant": "✅",
            "Mostly Compliant": "🟡",
            "Needs Improvement": "🟠",
            "Non-Compliant": "🔴",
        }.get(overall, "❓")
        
        yield f"**Overall Compliance:** {status_emoji} {overall} ({score}/100)\n\n"
        
        # Quick stats
        yield "| Metric | Value |\n"
        yield "|--------|-------|\n"
        yield f"| Total Cookies | {cookie_analysis.get('total', 0)} |\n"
        yield f"| Essential | {cookie_analysis.get('essential_count', 0)} |\n"
        yield f"| Analytics | {cookie_analysis.get('analytics_count', 0)} |\n"
        yield f"| Marketing | {cookie_analysis.get('marketing_count', 0)} |\n"
        yield f"| Functional | {cookie_analysis.get('functional_count', 0)} |\n"
        yield f"| Unknown | {cookie_analysis.get('unknown_count', 0)} |\n"
        yield f"| Trackers Detected | {len(data.get('trackers', []))} |\n"
        yield f"| Third-Party Hosts | {len(data.get('third_party_hosts', []))} |\n"
        yield f"| Consent Mechanism | {'✅ Yes' if data.get('consent_detected') else '❌ No'} |\n"
        yield f"| Cookie Policy | {'✅ Found' if data.get('cookie_policy_url') else '❌ Not Found'} |\n\n"
        
        # Issues
        issues = compliance.get("issues", [])
        if issues:
            yield "## ⚠️ Issues Found\n\n"
            for issue in issues:
                severity_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(issue["severity"], "⚪")
                yield f"- {severity_emoji} **{issue['severity'].upper()}:** {issue['issue']}\n"
                yield f"  - *ICO Reference: {issue['ico_ref']}*\n"
            yield "\n"
        
        # Recommendations
        recommendations = compliance.get("recommendations", [])
        if recommendations:
            yield "## 💡 Recommendations\n\n"
            for rec in recommendations:
                yield f"- {rec}\n"
            yield "\n"
        
        # Cookie details
        yield "## 🍪 Cookie Details\n\n"
        by_category = cookie_analysis.get("by_category", {})
        
        for category, cookies in by_category.items():
            if cookies:
                category_config = self.cookie_categories.get(category, {})
                ico_cat = category_config.get("ico_category", category.title())
                requires = "Yes" if category_config.get("requires_consent", True) else "No"
                
                yield f"### {category.title()} ({len(cookies)})\n\n"
                yield f"*ICO Category: {ico_cat} | Requires Consent: {requires}*\n\n"
                
                yield "| Cookie Name | Domain | Expires | Secure | HttpOnly |\n"
                yield "|-------------|--------|---------|--------|----------|\n"
                
                for cookie in cookies[:10]:  # Limit to 10 per category
                    expires = cookie.get("expires", -1)
                    if expires == -1:
                        expires_str = "Session"
                    else:
                        try:
                            expires_str = datetime.fromtimestamp(expires).strftime("%Y-%m-%d")
                        except:
                            expires_str = str(expires)
                    
                    yield f"| `{cookie['name'][:30]}` | {cookie['domain'][:25]} | {expires_str} | {'✅' if cookie.get('secure') else '❌'} | {'✅' if cookie.get('httpOnly') else '❌'} |\n"
                
                if len(cookies) > 10:
                    yield f"\n*...and {len(cookies) - 10} more*\n"
                yield "\n"
        
        # Trackers
        trackers = data.get("trackers", [])
        if trackers:
            yield "## 🔍 Trackers Detected\n\n"
            yield "| Tracker Type | URL |\n"
            yield "|--------------|-----|\n"
            
            seen = set()
            for tracker in trackers[:15]:
                key = (tracker["type"], urlparse(tracker["url"]).netloc)
                if key not in seen:
                    seen.add(key)
                    yield f"| {tracker['type']} | `{urlparse(tracker['url']).netloc}` |\n"
            
            if len(trackers) > 15:
                yield f"\n*...and {len(trackers) - 15} more tracking requests*\n"
            yield "\n"
        
        # Third-party hosts
        third_party = data.get("third_party_hosts", [])
        if third_party:
            yield "## 🌐 Third-Party Hosts\n\n"
            yield f"Found **{len(third_party)}** third-party domains:\n\n"
            for host in sorted(third_party)[:20]:
                yield f"- `{host}`\n"
            if len(third_party) > 20:
                yield f"\n*...and {len(third_party) - 20} more*\n"
            yield "\n"
        
        # Storage
        local_storage = data.get("local_storage", {})
        session_storage = data.get("session_storage", {})
        
        if local_storage or session_storage:
            yield "## 💾 Web Storage\n\n"
            if local_storage:
                yield f"**localStorage:** {len(local_storage)} items\n"
                for key in list(local_storage.keys())[:5]:
                    yield f"- `{key[:40]}`\n"
                if len(local_storage) > 5:
                    yield f"- *...and {len(local_storage) - 5} more*\n"
                yield "\n"
            
            if session_storage:
                yield f"**sessionStorage:** {len(session_storage)} items\n"
                for key in list(session_storage.keys())[:5]:
                    yield f"- `{key[:40]}`\n"
                if len(session_storage) > 5:
                    yield f"- *...and {len(session_storage) - 5} more*\n"
                yield "\n"
        
        # Footer
        yield "---\n\n"
        yield f"*Audit completed at {data.get('audit_time', 'N/A')}*\n"
        yield f"*Based on [ICO PECR Guidance]({self.valves.ICO_GUIDANCE_URL})*\n"
        yield f"*Analysis patterns from [EDPB Website Auditing Tool](https://code.europa.eu/edpb/website-auditing-tool)*\n"

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from user message"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        if match:
            return match.group(0).rstrip(".,;:)")
        
        # Try to find domain-like patterns
        domain_pattern = r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}\b'
        match = re.search(domain_pattern, text)
        if match:
            return match.group(0)
        
        return None

    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to audit (not internal/private)"""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            
            if not hostname:
                return False
            
            # Block localhost
            if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return False
            
            # Block internal hostnames
            if hostname.endswith(".local") or hostname.endswith(".internal"):
                return False
            
            # Try to resolve and check for private IPs
            try:
                ip = socket.gethostbyname(hostname)
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved:
                    return False
            except socket.gaierror:
                pass  # Can't resolve, allow it
            
            return True
        except Exception:
            return False
