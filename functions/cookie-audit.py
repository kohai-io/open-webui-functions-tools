"""
title: Cookie Compliance Audit
author: Open WebUI
version: 1.0.6
license: MIT
description: Audit website cookie usage against ICO (UK Information Commissioner's Office) PECR guidelines. Detects cookies, classifies them, checks for consent mechanisms, and generates compliance reports.
requirements: aiohttp, beautifulsoup4, lxml, pydantic
"""

import asyncio
import json
import logging
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Awaitable
from urllib.parse import urljoin, urlparse
import ipaddress
import socket

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Pipe:
    """Cookie Compliance Audit Pipeline - ICO/PECR focused"""

    class Valves(BaseModel):
        """Configuration options for the cookie audit"""

        MAX_PAGES: int = Field(
            default=50,
            description="Maximum pages to crawl for cookie detection",
        )
        CRAWL_DEPTH: int = Field(
            default=2,
            description="How deep to crawl from the homepage",
        )
        CRAWL_RATE_LIMIT: float = Field(
            default=0.5,
            description="Seconds between requests to avoid overloading the server",
        )
        TIMEOUT_SECONDS: int = Field(
            default=120,
            description="Maximum time for the entire audit",
        )
        ICO_GUIDANCE_URL: str = Field(
            default="https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/cookies-and-similar-technologies/",
            description="ICO guidance URL for cookie compliance citations",
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable verbose logging",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Cookie Compliance Audit"

        # ICO PECR Cookie Requirements
        self.ico_requirements = {
            "consent": {
                "name": "Cookie Consent",
                "description": "Users must give consent before non-essential cookies are set",
                "ico_ref": "PECR Regulation 6",
                "critical": True,
                "checks": [
                    "consent_before_cookies",
                    "clear_consent_mechanism",
                    "granular_choices",
                ],
            },
            "information": {
                "name": "Clear Information",
                "description": "Users must be told clearly what cookies are used and why",
                "ico_ref": "PECR Regulation 6(2)(a)",
                "critical": True,
                "checks": [
                    "cookie_policy_exists",
                    "cookies_explained",
                    "purposes_stated",
                ],
            },
            "choice": {
                "name": "Genuine Choice",
                "description": "Users must have a genuine choice to accept or reject non-essential cookies",
                "ico_ref": "PECR Regulation 6",
                "critical": True,
                "checks": [
                    "reject_option_available",
                    "equal_prominence",
                    "no_cookie_walls",
                ],
            },
            "essential_only": {
                "name": "Essential Cookies Exemption",
                "description": "Only strictly necessary cookies can be set without consent",
                "ico_ref": "PECR Regulation 6(4)",
                "critical": True,
                "checks": [
                    "essential_cookies_identified",
                    "non_essential_require_consent",
                ],
            },
        }

        # Cookie classification patterns (pre-compiled for performance)
        self.cookie_patterns = {
            "essential": {
                "patterns": [
                    re.compile(r"^(session|csrf|xsrf|token|auth|login|cart|basket|checkout|security).*$", re.IGNORECASE),
                    re.compile(r"^(consent|cookie_consent|cookieconsent|cc_cookie|gdpr|accepted_cookies|cookie_notice|cookies_accepted).*$", re.IGNORECASE),
                    re.compile(r"^(phpsessid|jsessionid|asp\.net_sessionid|laravel_session|wordpress_logged_in|wp-settings).*$", re.IGNORECASE),
                    # Infrastructure/CDN cookies (load balancing, security, bot protection)
                    re.compile(r"^(AWSALB|AWSALBCORS|AWSELB|AWSELBCORS).*$", re.IGNORECASE),  # AWS load balancer
                    re.compile(r"^(__cf_bm|_cfuvid|cf_clearance|__cfruid|__cflb).*$", re.IGNORECASE),  # Cloudflare
                    re.compile(r"^(JSESSIONID|SERVERID|ROUTEID|BACKEND).*$", re.IGNORECASE),  # Generic load balancer
                    re.compile(r"^(incap_ses|visid_incap|nlbi_).*$", re.IGNORECASE),  # Incapsula/Imperva
                    re.compile(r"^(ak_bmsc|bm_sv|bm_sz).*$", re.IGNORECASE),  # Akamai
                ],
                "description": "Strictly necessary for website functionality",
                "requires_consent": False,
                "ico_category": "Strictly necessary",
            },
            "analytics": {
                "patterns": [
                    re.compile(r"^(_ga|_gid|_gat|_gtag|__utm|_hjid|_hjSession|_pk_id|_pk_ses).*$", re.IGNORECASE),
                    re.compile(r"^(amplitude|mixpanel|segment|heap|hotjar|clarity|plausible).*$", re.IGNORECASE),
                ],
                "description": "Used to understand how visitors use the website",
                "requires_consent": True,
                "ico_category": "Performance/Analytics",
            },
            "marketing": {
                "patterns": [
                    re.compile(r"^(_fbp|_fbc|fr|tr|_gcl|gclid|_uetsid|_uetvid|IDE|DSID|__gads|__gpi).*$", re.IGNORECASE),
                    re.compile(r"^(_rdt_uuid|_pin_unauth|li_sugr|bcookie|bscookie).*$", re.IGNORECASE),
                    re.compile(r"^(facebook|fb_|google_ads|doubleclick|adsense|adwords|remarketing).*$", re.IGNORECASE),
                ],
                "description": "Used for advertising and remarketing",
                "requires_consent": True,
                "ico_category": "Targeting/Advertising",
            },
            "functional": {
                "patterns": [
                    re.compile(r"^(lang|language|locale|timezone|theme|dark_mode|font_size|accessibility|preferences|settings).*$", re.IGNORECASE),
                ],
                "description": "Remember user preferences",
                "requires_consent": True,
                "ico_category": "Functionality",
            },
        }

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Main entry point for the cookie audit"""

        # Check if this is a system task request (title generation, follow-ups, tags, etc.)
        # These tasks should not trigger the audit
        metadata = body.get("metadata", {})
        task = metadata.get("task", "")
        
        SYSTEM_TASKS = {
            "title_generation",
            "follow_up_generation",
            "tags_generation",
            "emoji_generation",
            "query_generation",
            "autocomplete_generation",
            "moa_response_generation",
            "TITLE_GENERATION",
            "FOLLOW_UP_GENERATION",
            "TAGS_GENERATION",
            "EMOJI_GENERATION",
            "QUERY_GENERATION",
            "AUTOCOMPLETE_GENERATION",
            "MOA_RESPONSE_GENERATION",
        }
        
        if task in SYSTEM_TASKS:
            log.debug(f"[COOKIE AUDIT] Ignoring system task: {task}")
            return

        messages = body.get("messages", [])
        if not messages:
            yield "Please provide a website URL to audit for cookie compliance."
            return

        last_message = messages[-1].get("content", "")
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

        yield f"# 🍪 Cookie Compliance Audit\n\n"
        yield f"**Target:** {target_url}\n"
        yield f"**Standard:** ICO PECR Guidelines\n"
        yield f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "Starting cookie audit...", "done": False}}
            )

        try:
            async with asyncio.timeout(self.valves.TIMEOUT_SECONDS):
                # Run the audit
                audit_data = await self.run_audit(target_url, __event_emitter__)

                # Generate report
                async for chunk in self.generate_report(audit_data):
                    yield chunk

        except asyncio.TimeoutError:
            yield f"\n\n⏱️ **Audit timed out** after {self.valves.TIMEOUT_SECONDS} seconds.\n"
        except Exception as e:
            log.exception(f"Cookie audit error: {e}")
            yield f"\n\n❌ **Error during audit:** {str(e)}\n"
        finally:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "Cookie audit complete", "done": True}}
                )

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from user message"""
        # Try to find URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        if match:
            return match.group(0).rstrip(".,;:!?)")

        # Try to find domain pattern
        domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        match = re.search(domain_pattern, text)
        if match:
            return match.group(0)

        return None

    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to request (SSRF protection)"""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname

            if not hostname:
                return False

            # Block localhost variations
            if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return False

            # Block common internal hostnames
            if hostname.lower() in ("internal", "intranet", "corp", "private"):
                return False

            # Resolve hostname and check if IP is private
            try:
                ip = socket.gethostbyname(hostname)
                ip_obj = ipaddress.ip_address(ip)

                # Block private, loopback, link-local, and reserved ranges
                if (ip_obj.is_private or ip_obj.is_loopback or 
                    ip_obj.is_link_local or ip_obj.is_reserved):
                    log.warning(f"[SECURITY] Blocked private/internal IP: {hostname} -> {ip}")
                    return False
            except socket.gaierror:
                # Can't resolve - might be invalid, let aiohttp handle it
                pass

            # Only allow http/https schemes
            if parsed.scheme not in ("http", "https"):
                return False

            return True

        except Exception as e:
            log.warning(f"[SECURITY] URL validation error: {e}")
            return False

    async def run_audit(
        self,
        url: str,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Run the cookie compliance audit"""

        start_time = datetime.now()

        # Phase 1: Crawl and collect cookies
        if event_emitter:
            await event_emitter(
                {"type": "status", "data": {"description": "Crawling website and collecting cookies...", "done": False}}
            )

        crawl_result = await self.crawl_and_collect_cookies(url)
        pages = crawl_result["pages"]
        cookies = crawl_result["cookies"]

        log.info(f"[COOKIE AUDIT] Crawled {len(pages)} pages, found {len(cookies)} cookies")

        # Phase 2: Analyze cookies
        if event_emitter:
            await event_emitter(
                {"type": "status", "data": {"description": "Analyzing cookies...", "done": False}}
            )

        cookie_analysis = self.analyze_cookies(cookies)

        # Phase 3: Check consent mechanisms
        if event_emitter:
            await event_emitter(
                {"type": "status", "data": {"description": "Checking consent mechanisms...", "done": False}}
            )

        consent_analysis = self.analyze_consent_mechanisms(pages)

        # Phase 4: Check cookie policy
        policy_analysis = self.analyze_cookie_policy(pages)

        # Phase 5: Assess ICO compliance
        if event_emitter:
            await event_emitter(
                {"type": "status", "data": {"description": "Assessing ICO compliance...", "done": False}}
            )

        compliance_results = self.assess_ico_compliance(
            cookie_analysis, consent_analysis, policy_analysis
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "target_url": url,
            "elapsed": elapsed,
            "pages_crawled": len(pages),
            "cookies": cookies,
            "cookie_analysis": cookie_analysis,
            "consent_analysis": consent_analysis,
            "policy_analysis": policy_analysis,
            "compliance_results": compliance_results,
        }

    async def crawl_and_collect_cookies(self, url: str) -> Dict[str, Any]:
        """Crawl website and collect all cookies"""

        visited = set()
        to_visit = [(url, 0)]  # (url, depth)
        pages = []
        collected_cookies = {}

        domain = urlparse(url).netloc
        cookie_jar = aiohttp.CookieJar()

        # Use realistic headers to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        }

        async with aiohttp.ClientSession(cookie_jar=cookie_jar, headers=headers) as session:
            while to_visit and len(visited) < self.valves.MAX_PAGES:
                current_url, depth = to_visit.pop(0)

                if current_url in visited or depth > self.valves.CRAWL_DEPTH:
                    continue

                visited.add(current_url)

                try:
                    await asyncio.sleep(self.valves.CRAWL_RATE_LIMIT)

                    async with session.get(current_url, timeout=10, ssl=False) as response:
                        if response.status != 200:
                            log.debug(f"[CRAWL] {current_url} returned status {response.status}")
                            continue

                        content_type = response.headers.get("content-type", "")
                        if "text/html" not in content_type:
                            continue

                        html = await response.text()
                        soup = BeautifulSoup(html, "lxml")

                        title = soup.find("title")
                        title_text = title.get_text().strip() if title else ""
                        text_content = soup.get_text(separator=" ", strip=True)

                        page_data = {
                            "url": current_url,
                            "title": title_text,
                            "content": text_content[:10000],
                            "html": html[:50000],
                        }
                        pages.append(page_data)

                        # Collect cookies after each request
                        for cookie in cookie_jar:
                            if cookie.key not in collected_cookies:
                                collected_cookies[cookie.key] = {
                                    "name": cookie.key,
                                    "value": cookie.value[:8] + "***" if len(cookie.value) > 8 else "***",  # Mask for security
                                    "domain": cookie.get("domain", domain),
                                    "path": cookie.get("path", "/"),
                                    "secure": cookie.get("secure", False),
                                    "httponly": cookie.get("httponly", False),
                                    "expires": str(cookie.get("expires", "")),
                                    "samesite": cookie.get("samesite", ""),
                                    "first_seen_url": current_url,
                                }

                        # Find links for further crawling
                        # Prioritize cookie/privacy related links
                        priority_links = []
                        normal_links = []
                        
                        for link in soup.find_all("a", href=True):
                            href = link["href"]
                            absolute_url = urljoin(current_url, href)
                            if urlparse(absolute_url).netloc == domain:
                                href_lower = href.lower()
                                link_text = link.get_text().lower()
                                # Priority: cookie policy, privacy, legal pages
                                if any(kw in href_lower or kw in link_text for kw in 
                                       ["cookie", "privacy", "legal", "policy", "notice", "gdpr", "consent"]):
                                    priority_links.append((absolute_url, depth + 1))
                                elif depth < self.valves.CRAWL_DEPTH:
                                    normal_links.append((absolute_url, depth + 1))
                        
                        # Add priority links first (they get crawled first)
                        to_visit = priority_links + to_visit + normal_links

                except asyncio.TimeoutError:
                    log.warning(f"[CRAWL] Timeout fetching {current_url}")
                    continue
                except Exception as e:
                    log.warning(f"[CRAWL] Error fetching {current_url}: {type(e).__name__}: {e}")
                    continue

        log.info(f"[CRAWL] Complete: {len(pages)} pages, {len(collected_cookies)} cookies")
        return {"pages": pages, "cookies": collected_cookies}

    def analyze_cookies(self, cookies: Dict[str, Dict]) -> Dict[str, Any]:
        """Classify and analyze collected cookies"""

        if not cookies:
            return {
                "total": 0,
                "by_category": {},
                "requires_consent": 0,
                "essential_only": 0,
                "unclassified": 0,
                "cookies_list": [],
            }

        by_category = {
            "essential": [],
            "analytics": [],
            "marketing": [],
            "functional": [],
            "unknown": [],
        }

        for cookie_name, cookie_info in cookies.items():
            classified = False
            cookie_name_lower = cookie_name.lower()

            for category, config in self.cookie_patterns.items():
                for pattern in config["patterns"]:
                    if pattern.match(cookie_name_lower):
                        by_category[category].append({
                            **cookie_info,
                            "category": category,
                            "ico_category": config["ico_category"],
                            "requires_consent": config["requires_consent"],
                        })
                        classified = True
                        break
                if classified:
                    break

            if not classified:
                by_category["unknown"].append({
                    **cookie_info,
                    "category": "unknown",
                    "ico_category": "Unclassified",
                    "requires_consent": True,
                })

        requires_consent = sum(
            len(by_category[cat]) for cat in ["analytics", "marketing", "functional", "unknown"]
        )

        return {
            "total": len(cookies),
            "by_category": by_category,
            "requires_consent": requires_consent,
            "essential_only": len(by_category["essential"]),
            "unclassified": len(by_category["unknown"]),
            "cookies_list": list(cookies.values()),
        }

    def analyze_consent_mechanisms(self, pages: List[Dict]) -> Dict[str, Any]:
        """Analyze cookie consent mechanisms on the website"""

        consent_indicators = {
            "banner_detected": False,
            "accept_button": False,
            "reject_button": False,
            "preferences_option": False,
            "granular_choices": False,
            "cookie_wall_detected": False,
            "consent_before_cookies": None,  # Can't determine without JS execution
            "evidence": [],
        }

        # Patterns to detect consent mechanisms
        consent_patterns = {
            "banner": [
                r"cookie.{0,20}(banner|notice|popup|consent|policy)",
                r"we use cookies",
                r"this (site|website) uses cookies",
                r"accept.{0,10}cookies",
                r"cookie.{0,10}preferences",
            ],
            "accept": [
                r"accept.{0,10}(all|cookies)",
                r"agree.{0,10}(all|cookies)",
                r"allow.{0,10}(all|cookies)",
                r"i.{0,5}accept",
                r"got it",
                r"ok.{0,5}(button|accept)",
            ],
            "reject": [
                r"reject.{0,10}(all|cookies|non.?essential)",
                r"decline.{0,10}(all|cookies)",
                r"refuse.{0,10}(all|cookies)",
                r"only.{0,10}essential",
                r"necessary.{0,10}only",
            ],
            "preferences": [
                r"manage.{0,10}(cookies|preferences|settings)",
                r"cookie.{0,10}(settings|preferences|options)",
                r"customize.{0,10}(cookies|preferences)",
                r"more.{0,10}options",
            ],
            "cookie_wall": [
                r"must.{0,10}accept.{0,10}cookies",
                r"accept.{0,10}to.{0,10}continue",
                r"cookies.{0,10}required",
            ],
        }

        for page in pages:
            content_lower = page.get("content", "").lower()
            html_lower = page.get("html", "").lower()

            # Check for banner
            for pattern in consent_patterns["banner"]:
                if re.search(pattern, content_lower) or re.search(pattern, html_lower):
                    consent_indicators["banner_detected"] = True
                    consent_indicators["evidence"].append(f"Cookie banner detected on {page['url']}")
                    break

            # Check for accept button
            for pattern in consent_patterns["accept"]:
                if re.search(pattern, html_lower):
                    consent_indicators["accept_button"] = True
                    break

            # Check for reject button
            for pattern in consent_patterns["reject"]:
                if re.search(pattern, html_lower):
                    consent_indicators["reject_button"] = True
                    consent_indicators["evidence"].append("Reject/decline option found")
                    break

            # Check for preferences
            for pattern in consent_patterns["preferences"]:
                if re.search(pattern, html_lower):
                    consent_indicators["preferences_option"] = True
                    consent_indicators["evidence"].append("Cookie preferences option found")
                    break

            # Check for cookie wall
            for pattern in consent_patterns["cookie_wall"]:
                if re.search(pattern, content_lower):
                    consent_indicators["cookie_wall_detected"] = True
                    consent_indicators["evidence"].append("⚠️ Potential cookie wall detected")
                    break

        # Check for granular choices (multiple checkboxes/toggles)
        for page in pages:
            html = page.get("html", "")
            # Look for multiple cookie category toggles
            toggle_patterns = [
                r'type=["\']checkbox["\'].{0,100}(analytics|marketing|functional|performance)',
                r'(analytics|marketing|functional|performance).{0,100}type=["\']checkbox["\']',
                r'toggle.{0,50}(analytics|marketing|functional)',
            ]
            matches = 0
            for pattern in toggle_patterns:
                if re.search(pattern, html, re.IGNORECASE):
                    matches += 1
            if matches >= 2:
                consent_indicators["granular_choices"] = True
                consent_indicators["evidence"].append("Granular cookie choices available")
                break

        return consent_indicators

    def analyze_cookie_policy(self, pages: List[Dict]) -> Dict[str, Any]:
        """Check for cookie policy page and its completeness"""

        policy_analysis = {
            "policy_found": False,
            "policy_url": None,
            "policy_title": None,
            "explains_cookies": False,
            "lists_cookies": False,
            "explains_purposes": False,
            "explains_third_parties": False,
            "explains_how_to_manage": False,
            "completeness_score": 0,
            "issues": [],
        }

        # Find cookie policy page
        policy_keywords = ["cookie policy", "cookie notice", "cookies policy", "use of cookies", "cookie-notice", "cookie-policy"]
        policy_url_patterns = [
            r"cookie[-_]?policy",
            r"cookie[-_]?notice", 
            r"cookies[-_]?policy",
            r"privacy.*cookie",
            r"cookie.*privacy",
        ]

        # Log all crawled URLs for debugging
        log.debug(f"[COOKIE POLICY] Checking {len(pages)} pages for cookie policy")
        for page in pages:
            url_lower = page.get("url", "").lower()
            title_lower = page.get("title", "").lower()
            content_lower = page.get("content", "").lower()

            # Check if this is a cookie policy page
            is_policy_page = False
            
            # Simple check: does URL contain both "cookie" and ("policy" or "notice")
            if "cookie" in url_lower and ("policy" in url_lower or "notice" in url_lower):
                is_policy_page = True
                log.info(f"[COOKIE POLICY] Found via simple URL check: {page.get('url')}")
            
            # Check keywords in title or URL
            if not is_policy_page:
                for keyword in policy_keywords:
                    if keyword in title_lower or keyword in url_lower:
                        is_policy_page = True
                        log.info(f"[COOKIE POLICY] Found via keyword '{keyword}': {page.get('url')}")
                        break

            # Check URL patterns with regex for more flexibility
            if not is_policy_page:
                for pattern in policy_url_patterns:
                    if re.search(pattern, url_lower):
                        is_policy_page = True
                        log.info(f"[COOKIE POLICY] Found via pattern '{pattern}': {page.get('url')}")
                        break

            # Also check if URL contains both "cookie" and ("policy" or "notice") anywhere
            if not is_policy_page:
                if "cookie" in url_lower and ("policy" in url_lower or "notice" in url_lower):
                    is_policy_page = True

            if is_policy_page:
                policy_analysis["policy_found"] = True
                policy_analysis["policy_url"] = page.get("url")
                policy_analysis["policy_title"] = page.get("title")

                # Check policy completeness
                if any(term in content_lower for term in ["what is a cookie", "what are cookies", "cookies are"]):
                    policy_analysis["explains_cookies"] = True

                if any(term in content_lower for term in ["we use the following", "cookies we use", "list of cookies", "cookie name"]):
                    policy_analysis["lists_cookies"] = True

                if any(term in content_lower for term in ["purpose", "why we use", "used for", "help us"]):
                    policy_analysis["explains_purposes"] = True

                if any(term in content_lower for term in ["third party", "third-party", "google analytics", "facebook"]):
                    policy_analysis["explains_third_parties"] = True

                if any(term in content_lower for term in ["how to manage", "how to control", "browser settings", "opt out", "disable cookies"]):
                    policy_analysis["explains_how_to_manage"] = True

                break

        # Calculate completeness score
        checks = [
            policy_analysis["policy_found"],
            policy_analysis["explains_cookies"],
            policy_analysis["lists_cookies"],
            policy_analysis["explains_purposes"],
            policy_analysis["explains_third_parties"],
            policy_analysis["explains_how_to_manage"],
        ]
        policy_analysis["completeness_score"] = sum(checks) / len(checks) * 100

        # Identify issues
        if not policy_analysis["policy_found"]:
            policy_analysis["issues"].append("No dedicated cookie policy page found")
        else:
            if not policy_analysis["explains_cookies"]:
                policy_analysis["issues"].append("Policy doesn't explain what cookies are")
            if not policy_analysis["lists_cookies"]:
                policy_analysis["issues"].append("Policy doesn't list specific cookies used")
            if not policy_analysis["explains_purposes"]:
                policy_analysis["issues"].append("Policy doesn't explain purposes of cookies")
            if not policy_analysis["explains_how_to_manage"]:
                policy_analysis["issues"].append("Policy doesn't explain how to manage cookies")

        return policy_analysis

    def assess_ico_compliance(
        self,
        cookie_analysis: Dict[str, Any],
        consent_analysis: Dict[str, Any],
        policy_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess compliance against ICO PECR requirements"""

        results = {}
        total_score = 0
        max_score = 0

        # Requirement 1: Consent
        consent_score = 0
        consent_issues = []
        consent_max = 3

        if consent_analysis["banner_detected"]:
            consent_score += 1
        else:
            consent_issues.append("No cookie consent banner detected")

        if consent_analysis["reject_button"]:
            consent_score += 1
        else:
            consent_issues.append("No clear reject/decline option found")

        if not consent_analysis["cookie_wall_detected"]:
            consent_score += 1
        else:
            consent_issues.append("Cookie wall detected - users may be forced to accept")

        results["consent"] = {
            "name": self.ico_requirements["consent"]["name"],
            "status": "GREEN" if consent_score == consent_max else ("AMBER" if consent_score >= 1 else "RED"),
            "score": consent_score,
            "max_score": consent_max,
            "issues": consent_issues,
            "ico_ref": self.ico_requirements["consent"]["ico_ref"],
        }
        total_score += consent_score
        max_score += consent_max

        # Requirement 2: Information
        info_score = 0
        info_issues = []
        info_max = 3

        if policy_analysis["policy_found"]:
            info_score += 1
        else:
            info_issues.append("No cookie policy page found")

        if policy_analysis["lists_cookies"]:
            info_score += 1
        else:
            info_issues.append("Cookies not clearly listed")

        if policy_analysis["explains_purposes"]:
            info_score += 1
        else:
            info_issues.append("Cookie purposes not explained")

        results["information"] = {
            "name": self.ico_requirements["information"]["name"],
            "status": "GREEN" if info_score == info_max else ("AMBER" if info_score >= 1 else "RED"),
            "score": info_score,
            "max_score": info_max,
            "issues": info_issues,
            "ico_ref": self.ico_requirements["information"]["ico_ref"],
        }
        total_score += info_score
        max_score += info_max

        # Requirement 3: Choice
        choice_score = 0
        choice_issues = []
        choice_max = 3

        if consent_analysis["reject_button"]:
            choice_score += 1
        else:
            choice_issues.append("No reject option - users can't easily decline")

        if consent_analysis["preferences_option"] or consent_analysis["granular_choices"]:
            choice_score += 1
        else:
            choice_issues.append("No granular cookie preferences available")

        if not consent_analysis["cookie_wall_detected"]:
            choice_score += 1
        else:
            choice_issues.append("Cookie wall removes genuine choice")

        results["choice"] = {
            "name": self.ico_requirements["choice"]["name"],
            "status": "GREEN" if choice_score == choice_max else ("AMBER" if choice_score >= 1 else "RED"),
            "score": choice_score,
            "max_score": choice_max,
            "issues": choice_issues,
            "ico_ref": self.ico_requirements["choice"]["ico_ref"],
        }
        total_score += choice_score
        max_score += choice_max

        # Requirement 4: Essential Only
        essential_score = 0
        essential_issues = []
        essential_max = 2

        non_essential = cookie_analysis["requires_consent"]
        if non_essential == 0:
            essential_score += 2
        elif consent_analysis["banner_detected"] and consent_analysis["reject_button"]:
            essential_score += 1
            essential_issues.append(f"{non_essential} non-essential cookies detected - ensure consent obtained first")
        else:
            essential_issues.append(f"{non_essential} non-essential cookies may be set without proper consent")

        if cookie_analysis["unclassified"] > 0:
            essential_issues.append(f"{cookie_analysis['unclassified']} unclassified cookies need review")

        results["essential_only"] = {
            "name": self.ico_requirements["essential_only"]["name"],
            "status": "GREEN" if essential_score == essential_max else ("AMBER" if essential_score >= 1 else "RED"),
            "score": essential_score,
            "max_score": essential_max,
            "issues": essential_issues,
            "ico_ref": self.ico_requirements["essential_only"]["ico_ref"],
        }
        total_score += essential_score
        max_score += essential_max

        # Overall assessment
        overall_pct = (total_score / max_score * 100) if max_score > 0 else 0
        if overall_pct >= 80:
            overall_status = "GREEN"
        elif overall_pct >= 50:
            overall_status = "AMBER"
        else:
            overall_status = "RED"

        return {
            "requirements": results,
            "total_score": total_score,
            "max_score": max_score,
            "overall_percentage": overall_pct,
            "overall_status": overall_status,
        }

    async def generate_report(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate the compliance report"""

        yield f"**Duration:** {data['elapsed']:.1f}s\n"
        yield f"**Pages Crawled:** {data['pages_crawled']}\n"
        yield f"**Cookies Found:** {data['cookie_analysis']['total']}\n\n"

        yield f"---\n\n"

        # Overall Status
        compliance = data["compliance_results"]
        status_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(compliance["overall_status"], "⚪")

        yield f"## Overall Compliance: {status_emoji} {compliance['overall_status']}\n\n"
        yield f"**Score:** {compliance['total_score']}/{compliance['max_score']} ({compliance['overall_percentage']:.0f}%)\n\n"

        yield f"📖 [ICO Cookie Guidance]({self.valves.ICO_GUIDANCE_URL})\n\n"

        # ICO Requirements Assessment
        yield f"---\n\n"
        yield f"## ICO PECR Requirements\n\n"

        for req_id, result in compliance["requirements"].items():
            status_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(result["status"], "⚪")
            yield f"### {status_emoji} {result['name']}\n\n"
            yield f"**Reference:** {result['ico_ref']}\n"
            yield f"**Score:** {result['score']}/{result['max_score']}\n\n"

            if result["issues"]:
                yield f"**Issues:**\n"
                for issue in result["issues"]:
                    yield f"- ⚠️ {issue}\n"
                yield f"\n"
            else:
                yield f"✅ No issues detected\n\n"

        # Cookie Summary
        yield f"---\n\n"
        yield f"## Cookie Analysis\n\n"

        cookie_analysis = data["cookie_analysis"]
        yield f"| Category | Count | Requires Consent | ICO Category |\n"
        yield f"|----------|-------|------------------|---------------|\n"

        category_info = {
            "essential": ("Essential", "No", "Strictly necessary"),
            "analytics": ("Analytics", "Yes", "Performance"),
            "marketing": ("Marketing", "Yes", "Targeting"),
            "functional": ("Functional", "Yes", "Functionality"),
            "unknown": ("Unknown", "Assumed Yes", "Unclassified"),
        }

        for cat, (name, consent, ico_cat) in category_info.items():
            count = len(cookie_analysis["by_category"].get(cat, []))
            yield f"| {name} | {count} | {consent} | {ico_cat} |\n"

        yield f"| **Total** | **{cookie_analysis['total']}** | | |\n\n"

        # Consent Mechanism Analysis
        yield f"---\n\n"
        yield f"## Consent Mechanism\n\n"

        consent = data["consent_analysis"]
        yield f"| Check | Status |\n"
        yield f"|-------|--------|\n"
        yield f"| Cookie banner detected | {'✅' if consent['banner_detected'] else '❌'} |\n"
        yield f"| Accept button | {'✅' if consent['accept_button'] else '❌'} |\n"
        yield f"| Reject/decline option | {'✅' if consent['reject_button'] else '❌'} |\n"
        yield f"| Preferences/settings option | {'✅' if consent['preferences_option'] else '❌'} |\n"
        yield f"| Granular choices | {'✅' if consent['granular_choices'] else '❌'} |\n"
        yield f"| Cookie wall detected | {'❌ (Bad)' if consent['cookie_wall_detected'] else '✅ (Good)'} |\n\n"

        # Cookie Policy Analysis
        yield f"---\n\n"
        yield f"## Cookie Policy\n\n"

        policy = data["policy_analysis"]
        if policy["policy_found"]:
            yield f"✅ **Cookie policy found:** [{policy['policy_title']}]({policy['policy_url']})\n\n"
            yield f"**Completeness:** {policy['completeness_score']:.0f}%\n\n"

            yield f"| Content Check | Status |\n"
            yield f"|---------------|--------|\n"
            yield f"| Explains what cookies are | {'✅' if policy['explains_cookies'] else '❌'} |\n"
            yield f"| Lists specific cookies | {'✅' if policy['lists_cookies'] else '❌'} |\n"
            yield f"| Explains purposes | {'✅' if policy['explains_purposes'] else '❌'} |\n"
            yield f"| Mentions third parties | {'✅' if policy['explains_third_parties'] else '❌'} |\n"
            yield f"| Explains how to manage | {'✅' if policy['explains_how_to_manage'] else '❌'} |\n\n"
        else:
            yield f"❌ **No dedicated cookie policy page found**\n\n"

        if policy["issues"]:
            yield f"**Issues:**\n"
            for issue in policy["issues"]:
                yield f"- ⚠️ {issue}\n"
            yield f"\n"

        # Cookie Details (collapsible)
        if cookie_analysis["total"] > 0:
            yield f"---\n\n"
            yield f"## Cookie Details\n\n"

            yield f"<details>\n<summary>View all {cookie_analysis['total']} cookies</summary>\n\n"
            yield f"| Cookie Name | Category | Domain | Secure | HttpOnly |\n"
            yield f"|-------------|----------|--------|--------|----------|\n"

            all_cookies = []
            for cat, cookies in cookie_analysis["by_category"].items():
                all_cookies.extend(cookies)

            for cookie in all_cookies[:50]:
                name = cookie.get("name", "Unknown")[:25]
                cat = cookie.get("category", "unknown")
                domain = cookie.get("domain", "")[:20]
                secure = "✅" if cookie.get("secure") else "❌"
                httponly = "✅" if cookie.get("httponly") else "❌"
                yield f"| {name} | {cat} | {domain} | {secure} | {httponly} |\n"

            if len(all_cookies) > 50:
                yield f"\n*...and {len(all_cookies) - 50} more cookies*\n"

            yield f"</details>\n\n"

        # Recommendations
        yield f"---\n\n"
        yield f"## Recommendations\n\n"

        recommendations = []

        if not consent["banner_detected"]:
            recommendations.append("Implement a cookie consent banner that appears before non-essential cookies are set")

        if not consent["reject_button"]:
            recommendations.append("Add a clear 'Reject All' or 'Only Essential' option with equal prominence to 'Accept All'")

        if consent["cookie_wall_detected"]:
            recommendations.append("Remove cookie wall - users must have genuine choice without being forced to accept")

        if not consent["granular_choices"]:
            recommendations.append("Provide granular cookie preferences allowing users to choose specific categories")

        if not policy["policy_found"]:
            recommendations.append("Create a dedicated cookie policy page explaining all cookies used")
        elif policy["completeness_score"] < 80:
            recommendations.append("Enhance cookie policy to include: list of all cookies, their purposes, and how to manage them")

        if cookie_analysis["unclassified"] > 0:
            recommendations.append(f"Review and document the {cookie_analysis['unclassified']} unclassified cookies")

        if cookie_analysis["requires_consent"] > 0 and not consent["banner_detected"]:
            recommendations.append("Ensure consent is obtained BEFORE setting non-essential cookies")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                yield f"{i}. 💡 {rec}\n"
        else:
            yield f"✅ No major recommendations - cookie compliance appears good!\n"

        yield f"\n"
