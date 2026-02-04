"""
title: Website Compliance Audit Pipeline
author: Open WebUI
version: 1.8.0
license: MIT
description: A comprehensive website audit agent with compliance checking, RAG analysis, intelligent sitemap generation, SEO analysis, broken link detection, cookie compliance auditing (GDPR/PECR), and change tracking over time
requirements: aiohttp, beautifulsoup4, lxml, python-dateutil, pydantic, openai, openpyxl
"""

import asyncio
import csv
import json
import logging
import sys
import time
import uuid
import hashlib
import re
from io import BytesIO, StringIO
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Awaitable, AsyncGenerator, Dict, Any, List
from urllib.parse import urlparse, urljoin

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from dateutil.parser import parse as parse_date
from openai import AsyncOpenAI

# Open WebUI Knowledge Base imports
from open_webui.models.knowledge import Knowledges, KnowledgeForm
from open_webui.models.files import Files, FileModel, FileForm
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.retrieval.utils import query_collection, get_embedding_function
from open_webui.storage.provider import Storage
from open_webui.routers.retrieval import process_file, ProcessFileForm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import app to create mock Request with app.state
from starlette.requests import Request as StarletteRequest
from starlette.datastructures import Headers

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


def get_app():
    """Get the FastAPI app instance to access app.state"""
    try:
        from open_webui.main import app

        return app
    except ImportError:
        log.warning("[KB] Could not import app from open_webui.main - embeddings may not work")
        return None


class UserObject:
    """Convert user dict to object with attributes for process_file compatibility"""

    def __init__(self, user_dict: dict):
        self.id = user_dict.get("id", "unknown")
        self.role = user_dict.get("role", "user")
        self.name = user_dict.get("name", "Unknown")
        self.email = user_dict.get("email", "unknown@example.com")


class Pipe:
    """Website Compliance Audit Pipeline for Open WebUI"""

    class Valves(BaseModel):
        """Configuration options for the audit pipeline"""

        REGULATORY_FRAMEWORK: str = Field(
            default="UK_DFE_SCHOOLS",
            description="Regulatory framework to check against (UK_DFE_SCHOOLS, GDPR, WCAG_2_1)",
        )
        SCHOOL_TYPE: str = Field(
            default="auto",
            description="School type: auto (detect from website), primary, secondary, all-through, or unknown",
        )
        CRAWL_DEPTH: int = Field(default=3, description="Maximum depth to crawl (1-5)")
        MAX_PAGES: int = Field(
            default=500, description="Maximum number of pages to analyze"
        )
        CRAWL_RATE_LIMIT: float = Field(
            default=1.0, description="Seconds between requests (respect the website)"
        )
        ENABLE_PDF_EXTRACTION: bool = Field(
            default=True, description="Extract and analyze PDF documents"
        )
        KNOWLEDGE_BASE_ID: str = Field(
            default="",
            description="Knowledge base ID containing regulatory frameworks (leave empty to use hardcoded defaults)",
        )
        # Document Analysis Settings
        ENABLE_DOCUMENT_UPLOAD: bool = Field(
            default=False,
            description="Upload PDFs and documents to knowledge base with automatic embedding generation for RAG search. Uses one persistent collection per domain.",
        )
        ENABLE_RAG_ANALYSIS: bool = Field(
            default=False,
            description="Use RAG to analyze document content for compliance (requires ENABLE_DOCUMENT_UPLOAD)",
        )
        MAX_WEBPAGES_FOR_RAG: int = Field(
            default=100,
            description="Maximum number of webpages to process for RAG analysis (scored by compliance relevance). More pages = better coverage but longer processing time.",
        )
        ENABLE_LLM_ASSESSMENT: bool = Field(
            default=False,
            description="Use LLM to assess document compliance (provides better GREEN/AMBER/RED differentiation than similarity scoring alone). Requires ENABLE_RAG_ANALYSIS and API key configuration.",
        )
        LLM_MODEL_FOR_ASSESSMENT: str = Field(
            default="gpt-4o-mini",
            description="LLM model to use for compliance assessment. Examples: gpt-4o-mini, gpt-4o, llama3.1, etc.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="OpenAI API key for LLM assessment (or compatible API). Required if ENABLE_LLM_ASSESSMENT is True.",
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API base URL. Use for OpenAI-compatible APIs (e.g., http://localhost:11434/v1 for Ollama)",
        )
        LLM_TIMEOUT_SECONDS: int = Field(
            default=120, description="Timeout for LLM API calls in seconds"
        )
        DATE_FRESHNESS_MONTHS: int = Field(
            default=12, description="Months before a document is considered outdated"
        )
        TIMEOUT_SECONDS: int = Field(
            default=300, description="Maximum time for entire audit (seconds)"
        )
        ENABLE_CACHE: bool = Field(
            default=True,
            description="Cache crawl results to avoid re-crawling unchanged pages",
        )
        CACHE_TTL_HOURS: int = Field(
            default=24, description="Cache validity period in hours"
        )
        FORCE_REFRESH: bool = Field(
            default=False, description="Force fresh crawl, ignoring cache"
        )
        DEBUG_MODE: bool = Field(
            default=False, description="Enable detailed logging to backend console"
        )
        GENERATE_SITEMAP: bool = Field(
            default=True,
            description="Generate sitemap (markdown for KB, mermaid for report visualization)",
        )
        EXPORT_REPORTS: bool = Field(
            default=True,
            description="Export audit reports as downloadable CSV and Excel files (uploaded to Files section)",
        )
        GOV_UK_GUIDANCE_URL: str = Field(
            default="https://www.gov.uk/guidance/what-maintained-schools-must-publish-online",
            description="GOV.UK guidance URL for school website requirements. Used for fetching framework and report citations.",
        )
        ENABLE_COOKIE_AUDIT: bool = Field(
            default=True,
            description="Audit cookie usage and consent mechanisms (GDPR/PECR compliance)",
        )
        ICO_COOKIE_GUIDANCE_URL: str = Field(
            default="https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/cookies-and-similar-technologies/",
            description="ICO guidance URL for cookie compliance. Used for report citations.",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "website_audit_pipe"
        self.name = "Website Compliance Audit"
        self.valves = self.Valves()
        self.cache_dir = Path.home() / ".cache" / "website_audit"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Debug helper
        self._debug_log = lambda msg: (
            log.info(f"[DEBUG] {msg}") if self.valves.DEBUG_MODE else None
        )

        # UK DfE school website requirements
        # Source: https://www.gov.uk/guidance/what-maintained-schools-must-publish-online
        # Updated from official guidance (accessed November 2024)
        # "must publish" = critical=True (statutory requirement)
        # "should publish" = critical=False (recommended best practice)
        self._default_checklist = {
            "statutory_information": {
                "name": "📄 Statutory Information",
                "gov_uk_url_anchor": "#statutory-information",
                "items": [
                    {
                        "id": "ofsted",
                        "name": "Ofsted Reports",
                        "keywords": ["ofsted", "inspection", "report"],
                        "critical": True,
                    },
                    {
                        "id": "performance",
                        "name": "Test & Exam Results",
                        "keywords": [
                            "performance",
                            "results",
                            "exam",
                            "gcse",
                            "key stage",
                            "progress 8",
                            "attainment 8",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "curriculum",
                        "name": "Curriculum",
                        "keywords": [
                            "curriculum",
                            "subjects",
                            "courses",
                            "religious education",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "admissions",
                        "name": "Admission Arrangements",
                        "keywords": [
                            "admissions",
                            "admission arrangements",
                            "apply",
                            "intake",
                            "pan",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "behaviour",
                        "name": "Behaviour Policy",
                        "keywords": ["behaviour", "discipline", "conduct"],
                        "critical": True,
                    },
                    {
                        "id": "complaints",
                        "name": "Complaints Policy",
                        "keywords": ["complaints", "concerns", "feedback"],
                        "critical": True,
                    },
                    {
                        "id": "sen",
                        "name": "SEN Information Report",
                        "keywords": [
                            "sen",
                            "special educational needs",
                            "send",
                            "sen information report",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "equality",
                        "name": "Public Sector Equality Duty",
                        "keywords": [
                            "equality",
                            "diversity",
                            "inclusion",
                            "public sector equality duty",
                            "equality objectives",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "pupil_premium",
                        "name": "Pupil Premium",
                        "keywords": [
                            "pupil premium",
                            "funding",
                            "disadvantaged pupils",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "pe_sports_premium",
                        "name": "PE & Sports Premium",
                        "keywords": [
                            "pe premium",
                            "sports premium",
                            "physical education",
                            "swimming",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "governance",
                        "name": "Governance Information",
                        "keywords": [
                            "governors",
                            "governance",
                            "board",
                            "governing body",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "charging",
                        "name": "Charging & Remissions Policies",
                        "keywords": ["charging", "fees", "costs", "remissions"],
                        "critical": True,
                    },
                    {
                        "id": "contact",
                        "name": "Contact Details",
                        "keywords": [
                            "contact",
                            "address",
                            "telephone",
                            "phone",
                            "senco",
                        ],
                        "critical": True,
                    },
                    {
                        "id": "financial",
                        "name": "Financial Information",
                        "keywords": ["financial", "salary", "benchmarking"],
                        "critical": True,
                    },
                    {
                        "id": "careers",
                        "name": "Careers Programme (Secondary)",
                        "keywords": ["careers", "careers guidance", "provider access"],
                        "critical": False,
                    },
                ],
            },
            "optional_recommended": {
                "name": "📌 Optional & Recommended",
                "gov_uk_url_anchor": "#optional-information",
                "items": [
                    {
                        "id": "ethos",
                        "name": "Ethos & Values",
                        "keywords": ["ethos", "values", "vision", "mission"],
                        "critical": False,
                    },
                    {
                        "id": "remote_education",
                        "name": "Remote Education",
                        "keywords": ["remote", "online learning", "home learning"],
                        "critical": False,
                    },
                    {
                        "id": "school_hours",
                        "name": "School Opening Hours",
                        "keywords": ["hours", "opening", "times", "term dates"],
                        "critical": False,
                    },
                    {
                        "id": "uniform",
                        "name": "School Uniform",
                        "keywords": ["uniform", "dress code", "clothing"],
                        "critical": False,
                    },
                    {
                        "id": "pay_gap",
                        "name": "Pay Gap Reporting (250+ staff)",
                        "keywords": ["pay gap", "gender pay"],
                        "critical": False,
                    },
                    {
                        "id": "music_plan",
                        "name": "Music Development Plan",
                        "keywords": ["music", "music development plan"],
                        "critical": False,
                    },
                ],
            },
        }

        # This will be populated from RAG or defaults
        self.compliance_checklist = None
        self.detected_school_type = None

    # Class-level cache to prevent duplicate runs
    _active_audits = {}

    def detect_school_type(self, pages: List[Dict]) -> str:
        """
        Auto-detect school type from website content.
        Returns: 'primary', 'secondary', 'all-through', or 'unknown'
        """
        # Combine text from all pages for analysis
        # Note: crawler stores text in "content" field, not "text"
        all_text = " ".join(
            [page.get("content", "").lower() for page in pages[:20]]
        )  # First 20 pages

        # Indicators for different school types
        primary_indicators = [
            "reception",
            "year 1",
            "year 2",
            "year 3",
            "year 4",
            "year 5",
            "year 6",
            "key stage 1",
            "key stage 2",
            "ks1",
            "ks2",
            "early years",
            "eyfs",
            "primary school",
            "infant",
            "junior",
        ]

        secondary_indicators = [
            "year 7",
            "year 8",
            "year 9",
            "year 10",
            "year 11",
            "year 12",
            "year 13",
            "key stage 3",
            "key stage 4",
            "key stage 5",
            "ks3",
            "ks4",
            "ks5",
            "gcse",
            "a-level",
            "a level",
            "sixth form",
            "secondary school",
        ]

        # Count indicators
        primary_count = sum(
            1 for indicator in primary_indicators if indicator in all_text
        )
        secondary_count = sum(
            1 for indicator in secondary_indicators if indicator in all_text
        )

        log.info(
            f"[SCHOOL TYPE] Detection: primary_indicators={primary_count}, secondary_indicators={secondary_count}"
        )

        # Determine school type
        if primary_count > 0 and secondary_count > 0:
            return "all-through"
        elif primary_count > secondary_count and primary_count >= 3:
            return "primary"
        elif secondary_count > primary_count and secondary_count >= 3:
            return "secondary"
        else:
            return "unknown"

    async def detect_school_type_from_kb(self, kb_id: str) -> str:
        """
        Detect school type from KB content (works even when pages are cached).
        Queries the KB for school type indicators.
        Returns: 'primary', 'secondary', 'all-through', or 'unknown'
        """
        try:
            # Get embedding function
            app = get_app()
            if not app or not hasattr(app, "state"):
                log.warning("[SCHOOL TYPE] Cannot access app.state for embeddings")
                return "unknown"

            config = app.state.config
            ef = get_embedding_function(
                embedding_engine=config.RAG_EMBEDDING_ENGINE,
                embedding_model=config.RAG_EMBEDDING_MODEL,
                embedding_function=app.state.ef,
                url=(
                    config.RAG_OPENAI_API_BASE_URL
                    if config.RAG_EMBEDDING_ENGINE == "openai"
                    else (
                        config.RAG_OLLAMA_BASE_URL
                        if config.RAG_EMBEDDING_ENGINE == "ollama"
                        else config.RAG_AZURE_OPENAI_API_BASE_URL
                    )
                ),
                key=(
                    config.RAG_OPENAI_API_KEY
                    if config.RAG_EMBEDDING_ENGINE == "openai"
                    else (
                        config.RAG_OLLAMA_API_KEY
                        if config.RAG_EMBEDDING_ENGINE == "ollama"
                        else config.RAG_AZURE_OPENAI_API_KEY
                    )
                ),
                embedding_batch_size=config.RAG_EMBEDDING_BATCH_SIZE,
            )

            # Query KB for school type indicators
            queries = [
                "year groups reception year 1 year 2 year 3 primary school",
                "year 7 year 8 gcse a-level secondary school sixth form",
            ]

            result = query_collection(
                collection_names=[kb_id], queries=queries, embedding_function=ef, k=5
            )

            # Count indicator mentions in results
            primary_count = 0
            secondary_count = 0

            primary_indicators = [
                "reception",
                "year 1",
                "year 2",
                "year 3",
                "year 4",
                "year 5",
                "year 6",
                "ks1",
                "ks2",
                "primary",
            ]
            secondary_indicators = [
                "year 7",
                "year 8",
                "year 9",
                "year 10",
                "year 11",
                "year 12",
                "year 13",
                "ks3",
                "ks4",
                "ks5",
                "gcse",
                "a-level",
                "sixth form",
                "secondary",
            ]

            if result and "documents" in result:
                for doc_list in result["documents"]:
                    for doc in doc_list:
                        doc_lower = doc.lower()
                        primary_count += sum(
                            1 for ind in primary_indicators if ind in doc_lower
                        )
                        secondary_count += sum(
                            1 for ind in secondary_indicators if ind in doc_lower
                        )

            log.info(
                f"[SCHOOL TYPE] KB detection: primary_indicators={primary_count}, secondary_indicators={secondary_count}"
            )

            # Determine school type
            if primary_count > 0 and secondary_count > 0:
                return "all-through"
            elif primary_count > secondary_count and primary_count >= 3:
                return "primary"
            elif secondary_count > primary_count and secondary_count >= 3:
                return "secondary"
            else:
                return "unknown"

        except Exception as e:
            log.error(f"[SCHOOL TYPE] Error detecting from KB: {e}")
            return "unknown"

    def get_filtered_checklist(self, school_type: str) -> Dict:
        """
        Filter checklist based on school type.
        Returns checklist with only applicable requirements.
        """
        if school_type == "all-through":
            # All-through schools need everything
            return self._default_checklist

        # Start with full checklist
        filtered = {}

        for category_id, category in self._default_checklist.items():
            filtered_items = []

            for item in category["items"]:
                # Filter based on school type
                if item["id"] == "careers" and school_type == "primary":
                    # Careers programme only for secondary
                    log.info(
                        f"[CHECKLIST] Skipping '{item['name']}' - not applicable to primary schools"
                    )
                    continue

                filtered_items.append(item)

            if filtered_items:
                filtered[category_id] = {
                    "name": category["name"],
                    "gov_uk_url": f"{self.valves.GOV_UK_GUIDANCE_URL}{category.get('gov_uk_url_anchor', '')}",
                    "items": filtered_items,
                }

        return filtered

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Main pipeline execution"""

        # Extract the target URL from the user's message
        messages = body.get("messages", [])
        if not messages:
            yield "Error: No messages provided"
            return

        # Only process messages from user, not from assistant (prevents pipeline output from triggering pipeline)
        last_message_obj = messages[-1]
        message_role = last_message_obj.get("role", "unknown")
        last_message = last_message_obj.get("content", "")

        log.info(
            f"[AUDIT] Received message with role={message_role}, content_preview={last_message[:100]}"
        )

        if message_role != "user":
            log.info(f"[AUDIT] Ignoring non-user message (role={message_role})")
            return

        # Ignore auto-generated suggestion requests from Open WebUI
        if "### Task:" in last_message or "follow-up questions" in last_message.lower():
            log.info(f"[AUDIT] Ignoring auto-generated suggestion request")
            return

        # Extract URL from message
        url_match = re.search(r"https?://[^\s]+", last_message)
        if not url_match:
            yield "Error: No URL found in message. Please provide a website URL to audit (e.g., https://blsschool.co.uk)"
            return

        target_url = url_match.group(0).rstrip(".,;")

        # Deduplication: Check if this exact audit is already running
        user_id = __user__.get("id") if __user__ else "unknown"
        audit_key = f"{user_id}:{target_url}"

        if audit_key in Pipe._active_audits:
            log.warning(
                f"[AUDIT] Duplicate audit request detected for {target_url}, ignoring"
            )
            return

        # Mark this audit as active
        Pipe._active_audits[audit_key] = time.time()
        log.info(
            f"[AUDIT] Starting new audit for {target_url} (active_audits: {len(Pipe._active_audits)})"
        )

        try:
            # Emit status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Starting audit of {target_url}",
                            "done": False,
                        },
                    }
                )

            # Initialize compliance framework
            if self.compliance_checklist is None:
                await self.initialize_framework(__event_call__)

            # Run the audit
            try:
                log.info(f"[AUDIT] Starting audit for {target_url}")
                log.info(
                    f"[AUDIT] ENABLE_CACHE={self.valves.ENABLE_CACHE}, DEBUG_MODE={self.valves.DEBUG_MODE}"
                )

                # Collect audit data and stream markdown
                log.info(f"[AUDIT] Streaming markdown report")
                audit_data = await self.run_audit_collect(
                    target_url, __user__, __event_emitter__, __event_call__
                )

                log.info(f"[AUDIT] Data collection complete. Rendering markdown...")

                # Stream markdown report
                async for chunk in self.format_markdown_report(audit_data):
                    yield chunk

                # Export reports if enabled
                if self.valves.EXPORT_REPORTS:
                    exported_files = await self.export_audit_reports(
                        audit_data, __user__
                    )
                    if exported_files:
                        yield f"\n\n---\n\n"
                        yield f"## 📥 Download Report Exports\n\n"
                        if "csv" in exported_files:
                            csv_id = exported_files["csv"]
                            yield f"- 📊 [**Download CSV Report**](/api/v1/files/{csv_id}/content) - Compliance results in spreadsheet format\n"
                        if "excel" in exported_files:
                            excel_id = exported_files["excel"]
                            yield f"- 📈 [**Download Excel Workbook**](/api/v1/files/{excel_id}/content) - Multi-sheet report with Summary, Compliance, SEO, and Link Health\n"
                        yield f"\n💡 *Click the links above to download the files directly*\n"
            except asyncio.TimeoutError:
                yield f"\n\n⏱️ **Audit timed out** after {self.valves.TIMEOUT_SECONDS} seconds. Partial results may be available.\n"
            except Exception as e:
                log.exception(f"Audit error: {e}")
                yield f"\n\n❌ **Error during audit:** {str(e)}\n"
            finally:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Audit complete", "done": True},
                        }
                    )
        finally:
            # Clean up: remove from active audits
            if audit_key in Pipe._active_audits:
                del Pipe._active_audits[audit_key]
                log.info(
                    f"[AUDIT] Completed audit for {target_url} (active_audits: {len(Pipe._active_audits)})"
                )

    async def crawl_website(
        self, url: str, cache_key: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Crawl the website and discover pages with intelligent caching"""

        visited = set()
        to_visit = [(url, 0, None)]  # (url, depth, parent_url)
        pages = []
        page_cache = self.load_page_cache(cache_key)
        stats = {"cached": 0, "fetched": 0, "unchanged": 0}
        page_relationships = {}  # Track parent-child relationships for sitemap
        collected_cookies = {}  # Track cookies: {cookie_name: cookie_info}

        domain = urlparse(url).netloc

        # Create cookie jar to collect cookies
        cookie_jar = aiohttp.CookieJar()
        async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
            while to_visit and len(visited) < self.valves.MAX_PAGES:
                current_url, depth, parent_url = to_visit.pop(0)

                if current_url in visited or depth > self.valves.CRAWL_DEPTH:
                    continue

                visited.add(current_url)

                # Track parent-child relationship for sitemap
                if parent_url:
                    page_relationships[current_url] = parent_url

                try:
                    # Check if we have cached headers for this URL
                    cached_page = page_cache.get(current_url, {})

                    # Respect rate limit
                    await asyncio.sleep(self.valves.CRAWL_RATE_LIMIT)

                    # Build headers for conditional request
                    headers = {}
                    if cached_page.get("etag"):
                        headers["If-None-Match"] = cached_page["etag"]
                    if cached_page.get("last_modified"):
                        headers["If-Modified-Since"] = cached_page["last_modified"]

                    async with session.get(
                        current_url, timeout=10, headers=headers
                    ) as response:
                        # 304 Not Modified - use cached version
                        if response.status == 304 and cached_page.get("content"):
                            pages.append(cached_page["content"])
                            stats["unchanged"] += 1
                            log.debug(f"Using cached (304): {current_url}")

                            # Still parse links from cached content if within depth
                            if (
                                depth < self.valves.CRAWL_DEPTH
                                and cached_page["content"].get("type") == "html"
                            ):
                                # Re-parse links from cached HTML
                                soup = BeautifulSoup(
                                    cached_page["content"].get("content", ""), "lxml"
                                )
                                for link in soup.find_all("a", href=True):
                                    href = link["href"]
                                    absolute_url = urljoin(current_url, href)
                                    if urlparse(absolute_url).netloc == domain:
                                        to_visit.append(
                                            (absolute_url, depth + 1, current_url)
                                        )

                            yield {"pages": pages, "stats": stats}
                            continue

                        if response.status != 200:
                            continue

                        content_type = response.headers.get("content-type", "")
                        etag = response.headers.get("ETag")
                        last_modified = response.headers.get("Last-Modified")

                        if "text/html" in content_type:
                            html = await response.text()
                            soup = BeautifulSoup(html, "lxml")

                            # Extract page info
                            title = soup.find("title")
                            title_text = title.get_text().strip() if title else ""

                            # Extract text content
                            text_content = soup.get_text(separator=" ", strip=True)

                            page_data = {
                                "url": current_url,
                                "title": title_text,
                                "content": text_content[:5000],  # Limit content size
                                "type": "html",
                                "parent_url": parent_url,
                                "status_code": response.status,
                                "etag": etag,
                                "last_modified": last_modified,
                            }

                            pages.append(page_data)

                            # Cache this page with headers
                            page_cache[current_url] = {
                                "content": page_data,
                                "etag": etag,
                                "last_modified": last_modified,
                                "cached_at": datetime.now().isoformat(),
                            }

                            # Determine if from cache or fresh
                            if cached_page:
                                stats["cached"] += 1
                                log.debug(f"Updated cache: {current_url}")
                            else:
                                stats["fetched"] += 1
                                log.debug(f"Fresh fetch: {current_url}")

                            # Find links
                            if depth < self.valves.CRAWL_DEPTH:
                                for link in soup.find_all("a", href=True):
                                    href = link["href"]
                                    absolute_url = urljoin(current_url, href)

                                    # Only follow internal links
                                    if urlparse(absolute_url).netloc == domain:
                                        to_visit.append(
                                            (absolute_url, depth + 1, current_url)
                                        )

                        elif (
                            "application/pdf" in content_type
                            and self.valves.ENABLE_PDF_EXTRACTION
                        ):
                            page_data = {
                                "url": current_url,
                                "title": current_url.split("/")[-1],
                                "content": "",
                                "type": "pdf",
                                "parent_url": parent_url,
                                "status_code": response.status,
                                "etag": etag,
                                "last_modified": last_modified,
                                "content_length": response.headers.get(
                                    "content-length", "0"
                                ),
                            }
                            pages.append(page_data)

                            # Cache PDF metadata
                            page_cache[current_url] = {
                                "content": page_data,
                                "etag": etag,
                                "last_modified": last_modified,
                                "cached_at": datetime.now().isoformat(),
                            }
                            stats["fetched"] += 1

                except Exception as e:
                    log.debug(f"Error crawling {current_url}: {e}")
                    # Try to use fully cached version if available
                    if cached_page.get("content"):
                        pages.append(cached_page["content"])
                        stats["cached"] += 1
                    continue

                # Collect cookies from cookie jar
                if self.valves.ENABLE_COOKIE_AUDIT:
                    for cookie in cookie_jar:
                        if cookie.key not in collected_cookies:
                            collected_cookies[cookie.key] = {
                                "name": cookie.key,
                                "value": cookie.value[:50] + "..." if len(cookie.value) > 50 else cookie.value,
                                "domain": cookie.get("domain", domain),
                                "path": cookie.get("path", "/"),
                                "secure": cookie.get("secure", False),
                                "httponly": cookie.get("httponly", False),
                                "expires": str(cookie.get("expires", "")),
                                "samesite": cookie.get("samesite", ""),
                                "first_seen_url": current_url,
                            }

                # Yield progress
                yield {
                    "pages": pages,
                    "stats": stats,
                    "relationships": page_relationships,
                    "cookies": collected_cookies,
                }

            # Save page cache
            if self.valves.ENABLE_CACHE:
                self.save_page_cache(cache_key, page_cache)

    async def analyze_documents(
        self, pages: List[Dict], base_url: str
    ) -> Dict[str, Any]:
        """Analyze discovered pages and map them to compliance requirements"""

        document_map = {}

        for page in pages:
            url = page["url"]
            title = page["title"].lower()
            content = page["content"].lower()

            # Try to extract dates from URL/filename first (more reliable), then content
            dates = self.extract_dates(content, url)
            last_updated = max(dates) if dates else None

            # Check against each compliance item
            for category_id, category in self.compliance_checklist.items():
                for item in category["items"]:
                    item_id = item["id"]
                    keywords = item["keywords"]

                    # Check if any keyword matches
                    matches = any(kw in title or kw in content for kw in keywords)

                    if matches:
                        if item_id not in document_map:
                            document_map[item_id] = {
                                "urls": [],
                                "dates": [],
                                "best_match_url": url,
                                "best_match_score": 0,
                                "content": "",  # Store content for LLM validation
                            }

                        # Calculate match score
                        score = sum(1 for kw in keywords if kw in title) * 10 + sum(
                            1 for kw in keywords if kw in content
                        )

                        document_map[item_id]["urls"].append(url)
                        document_map[item_id]["dates"].append(last_updated)

                        if score > document_map[item_id]["best_match_score"]:
                            document_map[item_id]["best_match_url"] = url
                            document_map[item_id]["best_match_score"] = score
                            document_map[item_id]["content"] = page[
                                "content"
                            ]  # Store best match content

        return document_map

    async def check_compliance(
        self,
        document_map: Dict[str, Any],
        event_call: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Check compliance status for each requirement"""

        results = {}
        current_date = datetime.now()

        for category_id, category in self.compliance_checklist.items():
            for item in category["items"]:
                item_id = item["id"]
                item_name = item["name"]
                critical = item["critical"]

                if item_id not in document_map:
                    # Document not found
                    results[item_id] = {
                        "status": "RED",
                        "reason": "Document not found on website",
                        "action": (
                            "Create and publish this document"
                            if critical
                            else "Consider adding this document"
                        ),
                        "url": None,
                        "last_updated": None,
                    }
                else:
                    doc_info = document_map[item_id]
                    url = doc_info["best_match_url"]
                    dates = [d for d in doc_info["dates"] if d]
                    last_updated = max(dates) if dates else None

                    # Calculate status based on date
                    if last_updated:
                        age_months = (current_date - last_updated).days / 30.44

                        if age_months <= self.valves.DATE_FRESHNESS_MONTHS:
                            results[item_id] = {
                                "status": "GREEN",
                                "reason": "Document found and up to date",
                                "url": url,
                                "last_updated": last_updated.strftime("%Y-%m-%d"),
                                "action": None,
                            }
                        elif age_months <= self.valves.DATE_FRESHNESS_MONTHS * 2:
                            results[item_id] = {
                                "status": "AMBER",
                                "reason": f"Document is {age_months:.0f} months old",
                                "url": url,
                                "last_updated": last_updated.strftime("%Y-%m-%d"),
                                "action": "Review and update document",
                            }
                        else:
                            results[item_id] = {
                                "status": "RED",
                                "reason": f"Document is significantly outdated ({age_months:.0f} months old)",
                                "url": url,
                                "last_updated": last_updated.strftime("%Y-%m-%d"),
                                "action": "Update document urgently",
                            }
                    else:
                        # Document found but no date - use LLM to enhance if enabled
                        base_result = {
                            "status": "AMBER",
                            "reason": "Document found but last update date unclear",
                            "url": url,
                            "last_updated": None,
                            "action": "Verify document is current and add publication date",
                        }

                        results[item_id] = base_result

        return results

    def extract_dates(self, text: str, url: str = "") -> List[datetime]:
        """Extract dates from text content and URL/filename"""

        dates = []

        # First, try to extract date from URL/filename (more reliable for documents)
        if url:
            # Extract year-month from URL path (e.g., /2022/10/ or /uploads/2024/11/)
            url_date_patterns = [
                r"/(\d{4})/(\d{1,2})/",  # /YYYY/MM/
                r"/(\d{4})-(\d{1,2})-",  # /YYYY-MM-
                r"-(\d{4})-(\d{1,2})\.",  # -YYYY-MM.pdf
            ]

            for pattern in url_date_patterns:
                match = re.search(pattern, url)
                if match:
                    try:
                        year, month = int(match.group(1)), int(match.group(2))
                        date = datetime(year, month, 1)
                        if date <= datetime.now() and date.year >= 2000:
                            dates.append(date)
                            log.debug(
                                f"[DATE] Extracted {date.strftime('%Y-%m')} from URL: {url}"
                            )
                    except:
                        pass

            # Extract dates from filename (e.g., "Sept-2022", "September-2024")
            filename = url.split("/")[-1].lower()
            month_year_pattern = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-\s]?(\d{4})"
            match = re.search(month_year_pattern, filename, re.IGNORECASE)
            if match:
                try:
                    year = int(match.group(1))
                    # Extract month name
                    month_match = re.search(
                        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*",
                        filename,
                        re.IGNORECASE,
                    )
                    if month_match:
                        month_str = month_match.group(1)
                        date = parse_date(f"{month_str} {year}", fuzzy=False)
                        if date <= datetime.now() and date.year >= 2000:
                            dates.append(date)
                            log.debug(
                                f"[DATE] Extracted {date.strftime('%Y-%m')} from filename: {filename}"
                            )
                except:
                    pass

        # Then extract from page content (less reliable - might be page update date, not document date)
        patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # DD/MM/YYYY or MM/DD/YYYY
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",  # YYYY-MM-DD
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",  # Month DD, YYYY
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",  # DD Month YYYY
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:5]:  # Limit to first 5 dates to avoid noise
                try:
                    date = parse_date(match, fuzzy=True)
                    # Only include reasonable dates (not in future, not too old)
                    if date <= datetime.now() and date.year >= 2000:
                        dates.append(date)
                except:
                    pass

        return dates

    def analyze_cookies(self, cookies: Dict[str, Dict], pages: List[Dict]) -> Dict[str, Any]:
        """
        Analyze collected cookies for GDPR/PECR compliance.
        Returns cookie audit results with classifications and compliance status.
        """
        if not cookies:
            return {
                "total_cookies": 0,
                "status": "GREEN",
                "reason": "No cookies detected",
                "cookies_by_category": {},
                "consent_mechanism": None,
                "cookie_policy_found": False,
                "issues": [],
                "recommendations": [],
            }

        # Known cookie patterns for classification
        cookie_patterns = {
            "essential": {
                "patterns": [
                    r"^(session|csrf|xsrf|token|auth|login|cart|basket|checkout|security|consent|cookie_consent|cookieconsent|cc_cookie|gdpr|accepted_cookies|cookie_notice|cookies_accepted)$",
                    r"^(phpsessid|jsessionid|asp\.net_sessionid|laravel_session|wordpress_logged_in|wp-settings).*$",
                ],
                "description": "Strictly necessary for website functionality",
                "requires_consent": False,
            },
            "analytics": {
                "patterns": [
                    r"^(_ga|_gid|_gat|_gtag|__utm|_hjid|_hjSession|_pk_id|_pk_ses|amplitude|mixpanel|segment|heap|hotjar|clarity).*$",
                    r"^(google.analytics|analytics|tracking|visitor|pageview).*$",
                ],
                "description": "Used to understand how visitors use the website",
                "requires_consent": True,
            },
            "marketing": {
                "patterns": [
                    r"^(_fbp|_fbc|fr|tr|_gcl|gclid|_uetsid|_uetvid|IDE|DSID|__gads|__gpi|_rdt_uuid|_pin_unauth).*$",
                    r"^(facebook|fb_|google_ads|doubleclick|adsense|adwords|remarketing|retargeting).*$",
                ],
                "description": "Used for advertising and remarketing",
                "requires_consent": True,
            },
            "functional": {
                "patterns": [
                    r"^(lang|language|locale|timezone|theme|dark_mode|font_size|accessibility|preferences|settings).*$",
                ],
                "description": "Remember user preferences",
                "requires_consent": True,
            },
        }

        # Classify cookies
        cookies_by_category = {
            "essential": [],
            "analytics": [],
            "marketing": [],
            "functional": [],
            "unknown": [],
        }

        for cookie_name, cookie_info in cookies.items():
            classified = False
            cookie_name_lower = cookie_name.lower()

            for category, config in cookie_patterns.items():
                for pattern in config["patterns"]:
                    if re.match(pattern, cookie_name_lower, re.IGNORECASE):
                        cookies_by_category[category].append({
                            **cookie_info,
                            "category": category,
                            "requires_consent": config["requires_consent"],
                        })
                        classified = True
                        break
                if classified:
                    break

            if not classified:
                cookies_by_category["unknown"].append({
                    **cookie_info,
                    "category": "unknown",
                    "requires_consent": True,  # Assume consent required for unknown
                })

        # Check for cookie consent mechanism
        consent_keywords = [
            "cookie consent", "cookie banner", "cookie notice", "cookie policy",
            "accept cookies", "cookie preferences", "manage cookies", "cookie settings",
            "we use cookies", "this site uses cookies", "gdpr", "privacy preferences"
        ]

        consent_mechanism_found = False
        cookie_policy_found = False
        cookie_policy_url = None

        for page in pages:
            content_lower = page.get("content", "").lower()
            title_lower = page.get("title", "").lower()
            url_lower = page.get("url", "").lower()

            # Check for consent mechanism in page content
            if any(kw in content_lower for kw in consent_keywords):
                consent_mechanism_found = True

            # Check for cookie policy page
            if "cookie" in url_lower and ("policy" in url_lower or "notice" in url_lower):
                cookie_policy_found = True
                cookie_policy_url = page.get("url")
            elif "cookie policy" in title_lower or "cookie notice" in title_lower:
                cookie_policy_found = True
                cookie_policy_url = page.get("url")

        # Determine compliance status and issues
        issues = []
        recommendations = []

        non_essential_count = (
            len(cookies_by_category["analytics"]) +
            len(cookies_by_category["marketing"]) +
            len(cookies_by_category["functional"]) +
            len(cookies_by_category["unknown"])
        )

        # Check for issues
        if non_essential_count > 0 and not consent_mechanism_found:
            issues.append("Non-essential cookies set without visible consent mechanism")

        if non_essential_count > 0 and not cookie_policy_found:
            issues.append("No cookie policy page found")

        if len(cookies_by_category["unknown"]) > 0:
            issues.append(f"{len(cookies_by_category['unknown'])} unclassified cookies detected")

        if len(cookies_by_category["marketing"]) > 0:
            issues.append(f"Marketing/tracking cookies detected ({len(cookies_by_category['marketing'])})")

        # Generate recommendations
        if not cookie_policy_found:
            recommendations.append("Create a dedicated cookie policy page explaining what cookies are used and why")

        if not consent_mechanism_found and non_essential_count > 0:
            recommendations.append("Implement a cookie consent banner that allows users to accept/reject non-essential cookies")

        if len(cookies_by_category["unknown"]) > 0:
            recommendations.append("Review and document all cookies, ensuring each has a clear purpose")

        if len(cookies_by_category["marketing"]) > 0:
            recommendations.append("Ensure marketing cookies are only set after explicit user consent")

        # Determine overall status
        if len(issues) == 0:
            status = "GREEN"
            reason = "Cookie usage appears compliant"
        elif len(issues) <= 2 and "Marketing" not in str(issues):
            status = "AMBER"
            reason = "Minor cookie compliance issues detected"
        else:
            status = "RED"
            reason = "Significant cookie compliance issues detected"

        return {
            "total_cookies": len(cookies),
            "status": status,
            "reason": reason,
            "cookies_by_category": cookies_by_category,
            "consent_mechanism_found": consent_mechanism_found,
            "cookie_policy_found": cookie_policy_found,
            "cookie_policy_url": cookie_policy_url,
            "issues": issues,
            "recommendations": recommendations,
            "essential_count": len(cookies_by_category["essential"]),
            "analytics_count": len(cookies_by_category["analytics"]),
            "marketing_count": len(cookies_by_category["marketing"]),
            "functional_count": len(cookies_by_category["functional"]),
            "unknown_count": len(cookies_by_category["unknown"]),
        }

    async def _fetch_framework_from_govuk(self) -> Optional[Dict[str, Any]]:
        """Fetch and parse UK DfE Schools framework from GOV.UK"""
        try:
            url = self.valves.GOV_UK_GUIDANCE_URL
            log.info(f"[FRAMEWORK] Fetching from GOV.UK: {url}")

            # Fetch page
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        log.warning(
                            f"[FRAMEWORK] GOV.UK returned status {response.status}"
                        )
                        return None

                    html = await response.text()

            # Parse HTML
            soup = BeautifulSoup(html, "lxml")

            # Extract main content
            main_content = soup.find("div", {"class": "govuk-govspeak"})
            if not main_content:
                log.warning("[FRAMEWORK] Could not find main content in GOV.UK page")
                return None

            # Build framework structure
            framework = {
                "source": "govuk",
                "url": url,
                "fetched_date": datetime.now().isoformat(),
                "categories": {},
            }

            current_category = None
            current_item = None

            # Parse headings and content
            for element in main_content.find_all(["h2", "h3", "p", "ul"]):
                if element.name == "h2":
                    # Main category
                    category_name = element.get_text(strip=True)
                    # Skip navigation/meta sections
                    if any(
                        skip in category_name.lower()
                        for skip in ["contents", "print", "related"]
                    ):
                        current_category = None
                        continue

                    category_id = (
                        category_name.lower().replace(" ", "_").replace("&", "and")
                    )
                    current_category = category_id
                    framework["categories"][category_id] = {
                        "name": category_name,
                        "items": [],
                    }
                    current_item = None

                elif element.name == "h3" and current_category:
                    # Requirement item
                    item_name = element.get_text(strip=True)
                    current_item = {
                        "name": item_name,
                        "keywords": [],
                        "critical": True,  # Default to MUST (can be refined)
                        "description": "",
                    }
                    framework["categories"][current_category]["items"].append(
                        current_item
                    )

                elif element.name == "p" and current_item:
                    # Description text
                    text = element.get_text(strip=True)
                    if text:
                        current_item["description"] += " " + text

                        # Detect if it's "should" vs "must"
                        text_lower = text.lower()
                        if (
                            "should publish" in text_lower
                            or "should include" in text_lower
                        ):
                            current_item["critical"] = False

                        # Extract keywords from description
                        # Look for quoted terms or important phrases
                        import re

                        quoted = re.findall(r"'([^']+)'", text)
                        current_item["keywords"].extend([q.lower() for q in quoted])

            # Generate content hash for change detection
            import json

            content_str = json.dumps(framework["categories"], sort_keys=True)
            framework["content_hash"] = hashlib.md5(content_str.encode()).hexdigest()

            log.info(
                f"[FRAMEWORK] ✅ Fetched {len(framework['categories'])} categories from GOV.UK"
            )
            return framework

        except Exception as e:
            log.error(f"[FRAMEWORK] Error fetching from GOV.UK: {e}")
            import traceback

            log.error(traceback.format_exc())
            return None

    def _convert_framework_to_markdown(self, framework: Dict[str, Any]) -> str:
        """Convert framework dict to markdown format"""
        lines = []

        # Header
        lines.append("# UK DfE Schools Compliance Framework")
        lines.append("")
        lines.append(f"**Source:** {framework.get('url', 'GOV.UK')}")
        lines.append(f"**Fetched:** {framework.get('fetched_date', 'Unknown')}")
        lines.append(f"**Hash:** {framework.get('content_hash', 'N/A')}")
        lines.append("")
        lines.append(
            "This framework is automatically fetched from official GOV.UK guidance."
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # Categories
        for category_id, category in framework.get("categories", {}).items():
            lines.append(f"## {category['name']}")
            lines.append("")

            # Items
            for item in category.get("items", []):
                lines.append(f"### {item['name']}")
                lines.append(
                    f"- **Type:** {'MUST' if item.get('critical', True) else 'SHOULD'}"
                )

                if item.get("keywords"):
                    keywords = list(set(item["keywords"]))[:10]  # Dedupe and limit
                    lines.append(f"- **Keywords:** {', '.join(keywords)}")

                if item.get("description"):
                    desc = item["description"].strip()
                    if desc:
                        lines.append(
                            f"- **Details:** {desc[:200]}..."
                        )  # Truncate long descriptions

                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_framework_markdown(self) -> str:
        """Generate markdown content from hardcoded _default_checklist"""
        lines = []

        # Header
        lines.append("# UK DfE Schools Compliance Framework")
        lines.append("")
        lines.append("This framework is based on official GOV.UK guidance:")
        lines.append(
            self.valves.GOV_UK_GUIDANCE_URL
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # Generate categories
        for category_id, category in self._default_checklist.items():
            # Category header
            lines.append(f"## {category['name']}")
            lines.append("")

            if "gov_uk_url" in category:
                lines.append(f"**Reference:** {category['gov_uk_url']}")
                lines.append("")

            # Items in this category
            for item in category.get("items", []):
                lines.append(f"### {item['name']}")
                lines.append(
                    f"- **Type:** {'MUST' if item.get('critical', False) else 'SHOULD'}"
                )

                # Keywords
                keywords = item.get("keywords", [])
                if keywords:
                    lines.append(f"- **Keywords:** {', '.join(keywords)}")

                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _get_or_create_frameworks_kb(self) -> Optional[Any]:
        """Get or create the 'Compliance Frameworks' KB collection"""
        try:
            collection_name = "Compliance Frameworks"

            # Check if collection exists
            all_knowledge = Knowledges.get_knowledge_bases()
            for kb in all_knowledge:
                if kb.name == collection_name:
                    log.info(f"[FRAMEWORK] Found existing KB: {kb.id}")
                    return kb

            # Create new collection
            log.info(f"[FRAMEWORK] Creating new KB: {collection_name}")
            knowledge_form = KnowledgeForm(
                name=collection_name,
                description="System collection for compliance framework definitions (UK DfE, GDPR, WCAG, etc.)",
                data={"source": "system", "managed_by": "website_audit_pipeline"},
            )

            kb = Knowledges.insert_new_knowledge("system", knowledge_form)
            if kb:
                log.info(f"[FRAMEWORK] Created KB: {kb.id}")
                return kb
            else:
                log.error("[FRAMEWORK] Failed to create KB")
                return None

        except Exception as e:
            log.error(f"[FRAMEWORK] Error managing KB: {e}")
            return None

    async def _upload_framework_to_kb(self, kb_id: str, framework_name: str) -> bool:
        """Upload framework to KB (fetches from GOV.UK, falls back to hardcoded)"""
        try:
            content = None
            content_hash = None
            source = "hardcoded"
            fetched_date = datetime.now().isoformat()

            # Try fetching from GOV.UK first
            if framework_name == "uk_dfe_schools":
                log.info(f"[FRAMEWORK] Attempting to fetch from GOV.UK...")
                govuk_framework = await self._fetch_framework_from_govuk()

                if govuk_framework:
                    # Successfully fetched from GOV.UK
                    content = self._convert_framework_to_markdown(govuk_framework)
                    content_hash = govuk_framework.get("content_hash")
                    source = "govuk"
                    fetched_date = govuk_framework.get("fetched_date", fetched_date)
                    log.info(
                        f"[FRAMEWORK] ✅ Using GOV.UK source (hash: {content_hash[:8]}...)"
                    )
                else:
                    log.warning(
                        f"[FRAMEWORK] GOV.UK fetch failed, using hardcoded fallback"
                    )

            # Fallback: Generate from hardcoded checklist
            if not content:
                log.info(f"[FRAMEWORK] Generating from hardcoded checklist")
                content = self._generate_framework_markdown()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                source = "hardcoded"

            # Create file record
            file_id = str(uuid.uuid4())
            filename = f"{framework_name}.md"

            # Upload to storage
            file_bytes, file_path = Storage.upload_file(
                BytesIO(content.encode("utf-8")), filename, []
            )
            log.info(f"[FRAMEWORK] Uploaded to storage: {file_path}")

            # Create file record with metadata for update tracking
            file_form = FileForm(
                id=file_id,
                filename=filename,
                path=file_path,
                data={
                    "framework": framework_name,
                    "source": source,
                    "content_hash": content_hash,
                    "fetched_date": fetched_date,
                },
                meta={
                    "name": filename,
                    "content_type": "text/markdown",
                    "size": len(content),
                    "framework_type": framework_name,
                    "source": source,
                    "content_hash": content_hash,
                    "fetched_date": fetched_date,
                },
            )

            Files.insert_new_file("system", file_form)
            log.info(f"[FRAMEWORK] Created file record: {file_id}")

            # Process with embeddings so users can query the framework via RAG
            app = get_app()
            if app:
                try:
                    # Create mock request
                    scope = {
                        "type": "http",
                        "method": "POST",
                        "headers": [],
                        "query_string": b"",
                        "app": app,
                    }
                    mock_request = StarletteRequest(scope)

                    # Create system user for processing
                    system_user = UserObject(
                        {
                            "id": "system",
                            "role": "admin",
                            "name": "System",
                            "email": "system@local",
                        }
                    )

                    log.info(f"[FRAMEWORK] Step 1: Extracting text from {filename}")
                    result1 = process_file(
                        mock_request, ProcessFileForm(file_id=file_id), user=system_user
                    )

                    if result1 and result1.get("status"):
                        log.info(f"[FRAMEWORK] Step 2: Adding to KB with embeddings")
                        result2 = process_file(
                            mock_request,
                            ProcessFileForm(file_id=file_id, collection_name=kb_id),
                            user=system_user,
                        )

                        if result2 and result2.get("status"):
                            # Add file_id to KB's file_ids list (process_file doesn't do this automatically)
                            knowledge = Knowledges.get_knowledge_by_id(kb_id)
                            if knowledge:
                                existing_data = knowledge.data if knowledge.data else {}
                                file_ids = existing_data.get("file_ids", [])
                                if file_id not in file_ids:
                                    file_ids.append(file_id)
                                    Knowledges.update_knowledge_data_by_id(
                                        kb_id, {"file_ids": file_ids}
                                    )
                                    log.info(
                                        f"[FRAMEWORK] ✅ Framework processed with embeddings and added to KB"
                                    )
                                    log.info(
                                        f"[FRAMEWORK] KB now has {len(file_ids)} file(s)"
                                    )
                        else:
                            log.warning(
                                f"[FRAMEWORK] Embedding generation failed, but file is in KB"
                            )
                    else:
                        log.warning(
                            f"[FRAMEWORK] Text extraction failed, but file is in KB"
                        )

                except Exception as e:
                    log.error(f"[FRAMEWORK] Error processing embeddings: {e}")
                    log.info(f"[FRAMEWORK] File saved but without embeddings")
            else:
                log.warning(
                    f"[FRAMEWORK] App.state not available - file saved but no embeddings"
                )

            return True

        except Exception as e:
            log.error(f"[FRAMEWORK] Error uploading to KB: {e}")
            import traceback

            log.error(traceback.format_exc())
            return False

    async def _check_framework_updates(
        self, kb_id: str, framework_name: str, file_record: Any
    ) -> bool:
        """Check if framework needs updating (returns True if updated)"""
        try:
            # Get current metadata
            fetched_date_str = (
                file_record.data.get("fetched_date") if file_record.data else None
            )
            old_hash = (
                file_record.data.get("content_hash") if file_record.data else None
            )
            source = file_record.data.get("source") if file_record.data else "unknown"

            if not fetched_date_str or source != "govuk":
                # Not from GOV.UK or no date, skip update check
                return False

            # Calculate age
            from dateutil.parser import parse as parse_date

            fetched_date = parse_date(fetched_date_str)
            age_days = (datetime.now() - fetched_date).days

            # Only check if older than 30 days
            if age_days < 30:
                log.info(
                    f"[FRAMEWORK] Framework is {age_days} days old, no update check needed"
                )
                return False

            log.info(
                f"[FRAMEWORK] Framework is {age_days} days old, checking for updates..."
            )

            # Fetch latest from GOV.UK
            if framework_name == "uk_dfe_schools":
                govuk_framework = await self._fetch_framework_from_govuk()

                if govuk_framework:
                    new_hash = govuk_framework.get("content_hash")

                    if new_hash != old_hash:
                        log.info(
                            f"[FRAMEWORK] 🔄 GOV.UK has updates! Old: {old_hash[:8]}... New: {new_hash[:8]}..."
                        )

                        # Delete old file
                        Files.delete_file_by_id(file_record.id)

                        # Upload new version
                        await self._upload_framework_to_kb(kb_id, framework_name)

                        log.info(f"[FRAMEWORK] ✅ Framework updated from GOV.UK")
                        return True
                    else:
                        log.info(
                            f"[FRAMEWORK] ✅ No changes detected (hash: {old_hash[:8]}...)"
                        )
                        return False

            return False

        except Exception as e:
            log.warning(f"[FRAMEWORK] Update check failed: {e}")
            return False

    def _load_framework_from_kb(self, kb_id: str, framework_name: str) -> Optional[str]:
        """Load framework markdown content from KB"""
        try:
            knowledge = Knowledges.get_knowledge_by_id(kb_id)
            if not knowledge:
                return None

            # Get file IDs from collection
            file_ids = knowledge.data.get("file_ids", []) if knowledge.data else []
            log.info(
                f"[FRAMEWORK] KB has {len(file_ids)} file(s) in collection: {file_ids}"
            )
            if not file_ids:
                log.info(f"[FRAMEWORK] No files in KB collection yet")
                return None

            # Find the framework file
            for file_id in file_ids:
                file_record = Files.get_file_by_id(file_id)
                if file_record and file_record.filename == f"{framework_name}.md":
                    # Read file from storage
                    try:
                        with open(file_record.path, "r", encoding="utf-8") as f:
                            content = f.read()
                        log.info(f"[FRAMEWORK] Loaded from KB: {file_record.filename}")

                        # Store file_record for update checking
                        self._framework_file_record = file_record

                        return content
                    except Exception as e:
                        log.warning(
                            f"[FRAMEWORK] Could not read file from storage: {e}"
                        )
                        return None

            log.info(f"[FRAMEWORK] Framework '{framework_name}.md' not found in KB")
            return None

        except Exception as e:
            log.error(f"[FRAMEWORK] Error loading from KB: {e}")
            return None

    async def initialize_framework(
        self, event_call: Optional[Callable[[dict], Awaitable[dict]]] = None
    ) -> None:
        """Initialize compliance framework with multiple sources (priority order: KB → User RAG → Hardcoded)"""

        # Determine framework filename based on regulatory framework
        framework_map = {
            "UK_DFE_SCHOOLS": "uk_dfe_schools",
            "GDPR": "gdpr",
            "WCAG_2_1": "wcag",
        }
        framework_name = framework_map.get(
            self.valves.REGULATORY_FRAMEWORK, "uk_dfe_schools"
        )

        # 1. Try loading from dedicated 'Compliance Frameworks' KB collection (highest priority)
        try:
            frameworks_kb = self._get_or_create_frameworks_kb()
            if frameworks_kb:
                # Try to load framework from KB
                framework_content = self._load_framework_from_kb(
                    frameworks_kb.id, framework_name
                )

                if framework_content:
                    # Check for updates (async background check)
                    if (
                        hasattr(self, "_framework_file_record")
                        and self._framework_file_record
                    ):
                        updated = await self._check_framework_updates(
                            frameworks_kb.id,
                            framework_name,
                            self._framework_file_record,
                        )
                        if updated:
                            # Reload after update
                            framework_content = self._load_framework_from_kb(
                                frameworks_kb.id, framework_name
                            )

                    # Parse markdown content
                    log.info(f"[FRAMEWORK] Parsing content from KB")
                    kb_checklist = self.load_framework_from_markdown(
                        content=framework_content
                    )
                    if kb_checklist:
                        self.compliance_checklist = kb_checklist
                        log.info(
                            f"[FRAMEWORK] ✅ Loaded {len(kb_checklist)} categories from KB"
                        )
                        return
                else:
                    # Framework not in KB yet, fetch and upload
                    log.info(f"[FRAMEWORK] Not found in KB, fetching and uploading...")
                    uploaded = await self._upload_framework_to_kb(
                        frameworks_kb.id, framework_name
                    )
                    if uploaded:
                        # Now load it
                        framework_content = self._load_framework_from_kb(
                            frameworks_kb.id, framework_name
                        )
                        if framework_content:
                            kb_checklist = self.load_framework_from_markdown(
                                content=framework_content
                            )
                            if kb_checklist:
                                self.compliance_checklist = kb_checklist
                                log.info(
                                    f"[FRAMEWORK] ✅ Loaded {len(kb_checklist)} categories from KB (uploaded)"
                                )
                                return
        except Exception as e:
            log.warning(f"[FRAMEWORK] KB loading failed: {e}, trying next source")

        # 2. Try loading from user-specified RAG KB (user customization)
        if self.valves.KNOWLEDGE_BASE_ID and event_call:
            log.info(f"[FRAMEWORK] Trying user RAG KB: {self.valves.KNOWLEDGE_BASE_ID}")
            try:
                rag_checklist = await self.load_framework_from_rag(
                    self.valves.REGULATORY_FRAMEWORK, event_call
                )
                if rag_checklist:
                    self.compliance_checklist = rag_checklist
                    log.info(
                        f"[FRAMEWORK] ✅ Loaded {len(rag_checklist)} categories from user RAG"
                    )
                    return
            except Exception as e:
                log.warning(
                    f"[FRAMEWORK] User RAG failed: {e}, using hardcoded defaults"
                )

        # 3. Fallback to hardcoded defaults (always works)
        self.compliance_checklist = self._default_checklist
        log.info(
            f"[FRAMEWORK] ✅ Using hardcoded {self.valves.REGULATORY_FRAMEWORK} framework"
        )

    def load_framework_from_markdown(
        self, file_path: Optional[Path] = None, content: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load compliance framework from markdown file or content string"""

        try:
            # Read from file if path provided, otherwise use content
            if file_path:
                with open(file_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
            elif content:
                markdown_content = content
            else:
                log.error("[FRAMEWORK] No file_path or content provided")
                return None

            checklist = {}
            current_category = None
            current_item = {}

            for line in markdown_content.split("\n"):
                line = line.strip()

                # Detect category headers (##)
                if line.startswith("## ") and not line.startswith("### "):
                    category_name = line[3:].strip()
                    # Skip meta sections
                    if category_name in [
                        "Compliance Notes",
                        "Legislation References",
                        "Additional Information",
                    ]:
                        current_category = None
                        continue

                    # Create category ID
                    category_id = (
                        category_name.lower().replace(" ", "_").replace("&", "and")
                    )
                    current_category = category_id
                    checklist[category_id] = {
                        "name": f"📋 {category_name}",
                        "items": [],
                    }

                # Detect requirement headers (###)
                elif line.startswith("### ") and current_category:
                    # Save previous item if exists
                    if current_item.get("id"):
                        checklist[current_category]["items"].append(current_item)

                    # Start new item
                    item_name = line[4:].strip()
                    item_id = item_name.lower().replace(" ", "_").replace("&", "and")
                    current_item = {
                        "id": item_id,
                        "name": item_name,
                        "keywords": [],
                        "critical": False,
                    }

                # Parse metadata
                elif line.startswith("- **Type:**") and current_item:
                    current_item["critical"] = "MUST" in line.upper()

                elif line.startswith("- **Keywords:**") and current_item:
                    keywords_str = line.split(":", 1)[1].strip()
                    current_item["keywords"] = [
                        kw.strip() for kw in keywords_str.split(",")
                    ]

            # Add last item
            if current_item.get("id") and current_category:
                checklist[current_category]["items"].append(current_item)

            # Add emoji icons to categories
            icon_map = {
                "statutory_information": "📄",
                "governance": "👥",
                "safeguarding_and_wellbeing": "🛡️",
                "financial_information": "💰",
                "performance_data": "📊",
                "staff_information": "👔",
                "data_protection_and_privacy": "🔒",
                "accessibility": "♿",
            }

            for cat_id, cat_data in checklist.items():
                if cat_id in icon_map:
                    cat_data["name"] = (
                        f"{icon_map[cat_id]} {cat_data['name'].replace('📋 ', '')}"
                    )

            return checklist if checklist else None

        except Exception as e:
            log.error(f"Error parsing markdown framework: {e}")
            return None

    async def load_framework_from_rag(
        self, framework: str, event_call: Callable[[dict], Awaitable[dict]]
    ) -> Optional[Dict[str, Any]]:
        """Load compliance framework from knowledge base"""

        try:
            # Query the knowledge base for requirements
            query = f"""
            List all {framework} school website compliance requirements.
            For each requirement include:
            - Category (e.g., Statutory Information, Safeguarding, etc.)
            - Requirement name
            - Whether it's critical/required
            - Keywords to search for on website
            """

            result = await event_call(
                {
                    "type": "call",
                    "data": {
                        "function": "knowledge_query",
                        "params": {
                            "knowledge_id": self.valves.KNOWLEDGE_BASE_ID,
                            "query": query,
                            "top_k": 50,
                        },
                    },
                }
            )

            # Parse the RAG results into checklist format
            if result and result.get("documents"):
                return self._parse_rag_documents(result["documents"], framework)

            return None

        except Exception as e:
            log.error(f"Error loading from RAG: {e}")
            return None

    def _parse_rag_documents(
        self, documents: List[str], framework: str
    ) -> Dict[str, Any]:
        """Parse RAG documents into compliance checklist format"""

        # This is a simplified parser - in production, you'd want more sophisticated parsing
        # Could use LLM to structure the documents into the required format

        checklist = {}

        # For now, merge RAG content with defaults
        # In a full implementation, you'd parse the documents to extract:
        # - Categories
        # - Requirements
        # - Keywords
        # - Criticality

        # Example: Use LLM to structure the RAG content
        # structured = await self._structure_with_llm(documents)

        # Return None to fallback to defaults if parsing fails
        # This ensures the system always works
        return None

    async def collect_document_urls(self, pages: List[Dict]) -> List[Dict[str, str]]:
        """Collect all document URLs (PDFs, DOCs, etc.) from crawled pages"""

        documents = []
        seen_urls = set()

        # Document extensions to look for
        doc_extensions = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]

        for page in pages:
            url = page["url"]

            # Check if page itself is a document
            if any(url.lower().endswith(ext) for ext in doc_extensions):
                if url not in seen_urls:
                    documents.append(
                        {
                            "url": url,
                            "type": url.split(".")[-1].upper(),
                            "source_page": url,
                            "title": page.get("title", ""),
                        }
                    )
                    seen_urls.add(url)

            # TODO: Parse page content for document links
            # Would need to extract links from HTML content
            # For now, relying on crawled document URLs

        log.info(f"[DOCUMENTS] Found {len(documents)} documents")
        return documents  # Return all documents - no artificial limit

    def collect_relevant_webpages(
        self, pages: List[Dict], max_pages: int = 50
    ) -> List[Dict[str, str]]:
        """
        Collect relevant webpages for KB upload based on compliance keywords.
        Returns pages scored by relevance to compliance requirements.
        """

        # All compliance keywords from checklist
        compliance_keywords = []
        for category in self._default_checklist.values():
            for item in category["items"]:
                compliance_keywords.extend(item.get("keywords", []))

        # Add school type indicators for better detection
        compliance_keywords.extend(
            [
                "reception",
                "year 1",
                "year 2",
                "year 3",
                "year 4",
                "year 5",
                "year 6",
                "year 7",
                "year 8",
                "year 9",
                "year 10",
                "year 11",
                "year 12",
                "year 13",
                "key stage",
                "ks1",
                "ks2",
                "ks3",
                "ks4",
                "ks5",
                "primary",
                "secondary",
                "contact",
                "address",
                "telephone",
                "email",
                "opening hours",
                "term dates",
            ]
        )

        scored_pages = []
        seen_urls = set()  # Track base URLs (without anchors) to avoid duplicates

        for page in pages:
            # Skip document URLs (already handled separately)
            url = page["url"]
            if any(
                url.lower().endswith(ext)
                for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]
            ):
                continue

            # Strip anchor links to get base URL (e.g., remove #content, #masthead)
            base_url = url.split("#")[0] if "#" in url else url

            # Skip if we've already processed this base URL
            if base_url in seen_urls:
                continue

            # Note: crawler stores text in "content" field, not "text"
            text = page.get("content", "").lower()
            title = page.get("title", "").lower()

            if not text:
                continue  # Skip pages without text content

            # Mark this base URL as seen
            seen_urls.add(base_url)

            # Score page by keyword relevance
            score = 0
            matched_keywords = set()

            # Check title (higher weight)
            for keyword in compliance_keywords:
                if keyword.lower() in title:
                    score += 3
                    matched_keywords.add(keyword)

            # Check text content
            for keyword in compliance_keywords:
                if keyword.lower() in text:
                    score += 1
                    matched_keywords.add(keyword)

            # Bonus for important pages (based on URL patterns)
            important_patterns = [
                "about",
                "contact",
                "information",
                "policies",
                "governance",
                "curriculum",
                "admissions",
                "safeguarding",
                "sen",
                "equality",
                "pupil-premium",
                "pe-sport",
                "complaints",
                "staff",
                "uniform",
                "term-dates",
                "calendar",
                "hours",
                "ofsted",
                "inspection",
                "values",
                "ethos",
            ]
            for pattern in important_patterns:
                if pattern in base_url.lower():
                    score += 5
                    break

            if score > 0:
                scored_pages.append(
                    {
                        "url": base_url,  # Use base URL without anchor
                        "type": "WEBPAGE",
                        "title": page.get("title", ""),
                        "text": text,
                        "score": score,
                        "matched_keywords": len(matched_keywords),
                    }
                )

        # Sort by score and return top pages
        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        top_pages = scored_pages[:max_pages]

        log.info(
            f"[WEBPAGES] Scored {len(scored_pages)} pages, selected top {len(top_pages)} for KB"
        )
        if top_pages:
            log.info(
                f"[WEBPAGES] Top page: '{top_pages[0]['title']}' (score: {top_pages[0]['score']}, keywords: {top_pages[0]['matched_keywords']})"
            )

        return top_pages

    def generate_sitemap_markdown(self, pages: List[Dict], target_url: str) -> str:
        """
        Generate a hierarchical markdown sitemap based on URL structure.
        Returns formatted markdown text suitable for KB storage.
        """
        from collections import defaultdict
        from urllib.parse import urlparse

        root_url = urlparse(target_url).scheme + "://" + urlparse(target_url).netloc

        # Filter out anchor links and deduplicate
        unique_pages = {}
        filtered_count = 0
        for page in pages:
            url = page["url"]
            # Skip anchor links (#content, #masthead, etc.)
            if "#" in url:
                filtered_count += 1
                continue
            # Deduplicate by URL
            if url not in unique_pages:
                unique_pages[url] = page

        pages = list(unique_pages.values())
        log.info(
            f"[SITEMAP] Filtered {filtered_count} anchor links, {len(pages)} unique pages remain"
        )

        # Separate HTML pages and documents
        html_pages = [p for p in pages if p.get("type") == "html"]
        documents = [p for p in pages if p.get("type") == "pdf"]

        # Generate markdown
        lines = []
        lines.append(f"# Website Sitemap - {urlparse(target_url).netloc}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(
            f"\n**Total Pages:** {len(pages)} ({len(html_pages)} HTML, {len(documents)} PDFs)"
        )
        lines.append("\n---\n")

        # Build URL path hierarchy for HTML pages
        lines.append("## 📄 Website Pages\n")

        def get_path_parts(url):
            """Extract path components from URL"""
            path = url.replace(root_url, "").strip("/")
            if not path:
                return []
            return path.split("/")

        # Group pages by path hierarchy
        path_tree = defaultdict(list)
        for page in html_pages:
            path = page["url"].replace(root_url, "").strip("/") or "home"
            parts = get_path_parts(page["url"])
            if len(parts) == 0:
                path_tree["__root__"].append(page)
            elif len(parts) == 1:
                path_tree[parts[0]].append(page)
            else:
                path_tree[parts[0]].append(page)

        # Sort sections
        for section in sorted(path_tree.keys()):
            section_pages = path_tree[section]

            if section == "__root__":
                lines.append("### Home\n")
            else:
                section_title = section.replace("-", " ").title()
                lines.append(f"### {section_title}\n")

            for page in sorted(section_pages, key=lambda x: x["url"]):
                title = page.get("title", "Untitled")
                rel_path = page["url"].replace(root_url, "")
                # Shorten overly long titles
                if len(title) > 60:
                    title = title[:57] + "..."
                lines.append(f"- [{title}]({rel_path})")

            lines.append("")  # Blank line between sections

        # Group documents by year/topic
        lines.append("\n---\n")
        lines.append("## 📎 Documents & Policies\n")

        # Group PDFs by category based on path
        doc_categories = defaultdict(list)
        for doc in documents:
            url = doc["url"]
            filename = url.split("/")[-1]

            # Extract year from path or filename
            if "/2025/" in url:
                category = "2025 Documents"
            elif "/2024/" in url:
                category = "2024 Documents"
            elif "/2023/" in url:
                category = "2023 Documents"
            elif "/2022/" in url:
                category = "2022 Documents"
            elif "/2021/" in url:
                category = "2021 Documents"
            else:
                category = "Older Documents"

            doc_categories[category].append(doc)

        # Output by category
        for category in sorted(doc_categories.keys(), reverse=True):
            lines.append(f"### {category} ({len(doc_categories[category])} files)\n")
            for doc in sorted(doc_categories[category], key=lambda x: x["url"]):
                filename = doc["url"].split("/")[-1]
                # Clean up filename for display
                display_name = filename.replace(".pdf", "").replace("-", " ")
                if len(display_name) > 50:
                    display_name = display_name[:47] + "..."
                rel_path = doc["url"].replace(root_url, "")
                lines.append(f"- `{display_name}` - [Download]({rel_path})")
            lines.append("")  # Blank line

        return "\n".join(lines)

    def analyze_site_structure(
        self, pages: List[Dict], target_url: str
    ) -> Dict[str, Any]:
        """
        Analyze site structure for SEO and accessibility issues.
        Returns metrics on orphaned pages, depth, duplicate titles, etc.
        """
        from collections import defaultdict

        root_url = urlparse(target_url).scheme + "://" + urlparse(target_url).netloc

        # Filter anchor links
        clean_pages = [p for p in pages if "#" not in p["url"]]
        html_pages = [p for p in clean_pages if p.get("type") == "html"]

        # Track all links between pages
        inbound_links = defaultdict(set)  # URL -> set of pages that link to it

        for page in html_pages:
            page_url = page["url"]
            # Track parent links (from crawl)
            parent = page.get("parent_url")
            if parent and "#" not in parent:
                inbound_links[page_url].add(parent)

        # Identify orphaned pages (no inbound links except from root)
        orphaned_pages = []
        for page in html_pages:
            url = page["url"]
            if url == root_url or url == root_url + "/":
                continue  # Skip homepage
            if len(inbound_links[url]) == 0:
                orphaned_pages.append(
                    {"url": url, "title": page.get("title", "Untitled")}
                )

        # Calculate page depth
        def get_url_depth(url):
            path = url.replace(root_url, "").strip("/")
            if not path:
                return 0
            return len(path.split("/"))

        depth_distribution = defaultdict(int)
        deep_pages = []  # Pages >3 levels deep

        for page in html_pages:
            depth = get_url_depth(page["url"])
            depth_distribution[depth] += 1
            if depth > 3:
                deep_pages.append(
                    {
                        "url": page["url"],
                        "title": page.get("title", "Untitled"),
                        "depth": depth,
                    }
                )

        # Find duplicate titles
        title_map = defaultdict(list)
        missing_titles = []

        for page in html_pages:
            title = page.get("title", "").strip()
            if not title or title.lower() in ["untitled", ""]:
                missing_titles.append(page["url"])
            else:
                title_map[title].append(page["url"])

        duplicate_titles = {
            title: urls for title, urls in title_map.items() if len(urls) > 1
        }

        log.info(
            f"[SEO] Analysis: {len(orphaned_pages)} orphaned, {len(deep_pages)} deep, {len(duplicate_titles)} duplicate titles"
        )

        return {
            "total_pages": len(clean_pages),
            "html_pages": len(html_pages),
            "orphaned_pages": orphaned_pages,
            "orphaned_count": len(orphaned_pages),
            "deep_pages": deep_pages,
            "deep_pages_count": len(deep_pages),
            "depth_distribution": dict(depth_distribution),
            "missing_titles": missing_titles,
            "missing_titles_count": len(missing_titles),
            "duplicate_titles": duplicate_titles,
            "duplicate_titles_count": len(duplicate_titles),
        }

    def detect_broken_links(self, pages: List[Dict]) -> Dict[str, Any]:
        """
        Analyze pages for broken links, 404s, and missing documents.
        Uses status codes captured during crawl.
        """
        broken_links = []
        redirects = []
        missing_documents = []

        for page in pages:
            url = page["url"]
            # Check for error status codes (captured during crawl)
            status = page.get("status_code", 200)

            if status == 404:
                broken_links.append(
                    {"url": url, "title": page.get("title", "Untitled"), "status": 404}
                )
            elif status >= 400:
                broken_links.append(
                    {
                        "url": url,
                        "title": page.get("title", "Untitled"),
                        "status": status,
                    }
                )
            elif status in [301, 302, 307, 308]:
                redirects.append(
                    {
                        "url": url,
                        "title": page.get("title", "Untitled"),
                        "status": status,
                        "redirect_to": page.get("redirect_url", "Unknown"),
                    }
                )

            # Check for PDFs that might be missing
            if page.get("type") == "pdf" and status != 200:
                missing_documents.append(
                    {"url": url, "filename": url.split("/")[-1], "status": status}
                )

        log.info(
            f"[LINKS] Found {len(broken_links)} broken links, {len(redirects)} redirects, {len(missing_documents)} missing docs"
        )

        return {
            "broken_links": broken_links,
            "broken_count": len(broken_links),
            "redirects": redirects,
            "redirect_count": len(redirects),
            "missing_documents": missing_documents,
            "missing_documents_count": len(missing_documents),
            "total_issues": len(broken_links) + len(missing_documents),
        }

    def compare_sitemaps(
        self, current_pages: List[Dict], previous_sitemap: Dict
    ) -> Dict[str, Any]:
        """
        Compare current sitemap to previous crawl to detect changes.
        Returns added/removed pages and documents.
        """
        if not previous_sitemap:
            return {
                "has_previous": False,
                "message": "First audit - no previous sitemap to compare",
            }

        # Extract URLs from current and previous
        current_urls = {p["url"] for p in current_pages if "#" not in p["url"]}
        previous_urls = set(previous_sitemap.get("urls", []))

        # Calculate changes
        added_urls = current_urls - previous_urls
        removed_urls = previous_urls - current_urls
        unchanged_urls = current_urls & previous_urls

        # Separate by type
        added_pages = [
            p
            for p in current_pages
            if p["url"] in added_urls and p.get("type") == "html"
        ]
        added_docs = [
            p
            for p in current_pages
            if p["url"] in added_urls and p.get("type") == "pdf"
        ]

        removed_pages = [url for url in removed_urls if not url.endswith(".pdf")]
        removed_docs = [url for url in removed_urls if url.endswith(".pdf")]

        # Check for updated documents (same URL, different hash)
        updated_docs = []
        previous_hashes = previous_sitemap.get("document_hashes", {})
        for page in current_pages:
            if page.get("type") == "pdf" and page["url"] in unchanged_urls:
                current_hash = page.get("content_hash", "")
                previous_hash = previous_hashes.get(page["url"], "")
                if current_hash and previous_hash and current_hash != previous_hash:
                    updated_docs.append(
                        {"url": page["url"], "filename": page["url"].split("/")[-1]}
                    )

        log.info(
            f"[CHANGES] +{len(added_urls)} added, -{len(removed_urls)} removed, ~{len(updated_docs)} updated since {previous_sitemap.get('generated_date', 'previous audit')}"
        )

        return {
            "has_previous": True,
            "previous_date": previous_sitemap.get("generated_date", "Unknown"),
            "total_added": len(added_urls),
            "total_removed": len(removed_urls),
            "total_unchanged": len(unchanged_urls),
            "added_pages": [
                {"url": p["url"], "title": p.get("title", "Untitled")}
                for p in added_pages
            ],
            "added_docs": [
                {"url": p["url"], "filename": p["url"].split("/")[-1]}
                for p in added_docs
            ],
            "removed_pages": [{"url": url} for url in removed_pages],
            "removed_docs": [
                {"url": url, "filename": url.split("/")[-1]} for url in removed_docs
            ],
            "updated_docs": updated_docs,
            "updated_count": len(updated_docs),
        }

    def save_sitemap_snapshot(self, pages: List[Dict], target_url: str) -> None:
        """Save current sitemap snapshot for future change tracking"""
        from datetime import datetime

        domain = urlparse(target_url).netloc
        snapshot_file = self.cache_dir / f"{domain}_sitemap_snapshot.json"

        # Build snapshot data
        snapshot = {
            "generated_date": datetime.now().isoformat(),
            "url": target_url,
            "urls": [p["url"] for p in pages if "#" not in p["url"]],
            "document_hashes": {
                p["url"]: p.get("etag", p.get("content_hash", ""))
                for p in pages
                if p.get("type") == "pdf" and "#" not in p["url"]
            },
        }

        try:
            with open(snapshot_file, "w") as f:
                json.dump(snapshot, f, indent=2)
            log.info(f"[SITEMAP] Saved snapshot: {len(snapshot['urls'])} URLs")
        except Exception as e:
            log.warning(f"[SITEMAP] Failed to save snapshot: {e}")

    def load_sitemap_snapshot(self, target_url: str) -> Optional[Dict]:
        """Load previous sitemap snapshot for change comparison"""
        domain = urlparse(target_url).netloc
        snapshot_file = self.cache_dir / f"{domain}_sitemap_snapshot.json"

        if not snapshot_file.exists():
            return None

        try:
            with open(snapshot_file, "r") as f:
                snapshot = json.load(f)
            log.info(
                f"[SITEMAP] Loaded previous snapshot from {snapshot.get('generated_date', 'unknown date')}"
            )
            return snapshot
        except Exception as e:
            log.warning(f"[SITEMAP] Failed to load snapshot: {e}")
            return None

    def generate_sitemap_mermaid(self, pages: List[Dict], max_nodes: int = 50) -> str:
        """
        Generate a Mermaid.js flowchart for visual sitemap.
        Limits to top-level pages to avoid overwhelming diagrams.
        """
        from collections import defaultdict

        # Filter out anchor links and deduplicate
        unique_pages = {}
        for page in pages:
            url = page["url"]
            # Skip anchor links
            if "#" in url:
                continue
            if url not in unique_pages:
                unique_pages[url] = page

        pages = list(unique_pages.values())

        # Build parent-child relationships
        tree = defaultdict(list)
        for page in pages:
            url = page["url"]
            parent = page.get("parent_url")
            # Also skip parent if it has an anchor
            if parent and "#" in parent:
                parent = parent.split("#")[0]  # Use base URL without anchor
            tree[parent if parent else "root"].append(page)

        # Generate Mermaid syntax
        lines = ["graph TD"]
        node_id_map = {}  # URL -> node_id
        node_counter = 0

        def sanitize_label(text: str, max_len: int = 30) -> str:
            """Sanitize text for Mermaid labels"""
            if len(text) > max_len:
                text = text[: max_len - 3] + "..."
            # Escape special characters
            text = text.replace('"', "'").replace("[", "(").replace("]", ")")
            return text

        def add_node(page: Dict, parent_id: str = None) -> str:
            """Add a node and its connection"""
            nonlocal node_counter

            if node_counter >= max_nodes:
                return None

            url = page["url"]
            if url in node_id_map:
                return node_id_map[url]

            # Create node ID
            node_id = f"N{node_counter}"
            node_id_map[url] = node_id
            node_counter += 1

            # Format label
            title = page.get("title", "Untitled")
            page_type = page.get("type", "html").upper()
            label = sanitize_label(title)

            # Different shapes for different types
            if page_type == "PDF":
                lines.append(f'  {node_id}["{label} 📄"]')
            elif url == page.get("parent_url", "") or parent_id is None:
                lines.append(f'  {node_id}("{label}")')  # Root node (rounded)
            else:
                lines.append(f'  {node_id}["{label}"]')

            # Add connection from parent
            if parent_id:
                lines.append(f"  {parent_id} --> {node_id}")

            return node_id

        # Build tree starting from root
        def build_tree(parent_key: str, parent_node_id: str = None, depth: int = 0):
            """Recursively build Mermaid tree"""
            if node_counter >= max_nodes or depth > 2:  # Limit depth for readability
                return

            children = tree.get(parent_key, [])
            children.sort(key=lambda x: x["url"])

            for page in children:
                node_id = add_node(page, parent_node_id)
                if node_id:
                    build_tree(page["url"], node_id, depth + 1)

        # Start from root
        root_pages = tree.get("root", [])
        if root_pages:
            root_node_id = add_node(root_pages[0])
            build_tree(root_pages[0]["url"], root_node_id, depth=0)

            # Add remaining root-level pages
            for page in root_pages[1:]:
                if node_counter >= max_nodes:
                    break
                add_node(page, root_node_id)

        # Add truncation notice if needed
        if node_counter >= max_nodes:
            lines.append(f'  MORE["... {len(pages) - max_nodes} more pages"]')
            lines.append("  style MORE fill:#f9f9f9,stroke:#ccc,stroke-dasharray: 5 5")

        return "\n".join(lines)

    async def export_audit_reports(
        self, audit_data: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Export audit reports as CSV and Excel files, uploaded to user's Files.
        Returns dict with file_ids and download info.
        """
        if not self.valves.EXPORT_REPORTS:
            return {}

        # Generate filename base
        domain = urlparse(audit_data["target_url"]).netloc
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"audit_{domain}_{timestamp}"

        exported_files = {}

        try:
            # Generate CSV
            csv_content = self._generate_csv_report(audit_data)
            csv_file_id = await self._upload_file_to_system(
                content=csv_content.encode("utf-8"),
                filename=f"{base_filename}.csv",
                user=user,
                content_type="text/csv",
            )
            if csv_file_id:
                exported_files["csv"] = csv_file_id
                log.info(f"[EXPORT] Uploaded CSV report: {csv_file_id}")

            # Generate Excel
            excel_content = await self._generate_excel_report(audit_data)
            excel_file_id = await self._upload_file_to_system(
                content=excel_content,
                filename=f"{base_filename}.xlsx",
                user=user,
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if excel_file_id:
                exported_files["excel"] = excel_file_id
                log.info(f"[EXPORT] Uploaded Excel report: {excel_file_id}")

            log.info(f"[EXPORT] Successfully exported {len(exported_files)} files")
            return exported_files

        except Exception as e:
            log.error(f"[EXPORT] Failed to export reports: {e}")
            import traceback

            log.error(traceback.format_exc())
            return exported_files

    def _generate_csv_report(self, audit_data: Dict[str, Any]) -> str:
        """Generate CSV report with compliance results"""
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([f"Website Audit Report - {audit_data['target_url']}"])
        writer.writerow([f"Generated: {audit_data['started']}"])
        writer.writerow([f"Framework: {audit_data['framework']}"])
        writer.writerow([f"Pages Analyzed: {audit_data['page_count']}"])
        writer.writerow([])

        # Summary
        summary = audit_data["summary"]
        writer.writerow(["Summary", "Count", "Percentage"])
        writer.writerow(
            ["Green (Compliant)", summary["green"], f"{summary['green_pct']:.1f}%"]
        )
        writer.writerow(
            ["Amber (Partial)", summary["amber"], f"{summary['amber_pct']:.1f}%"]
        )
        writer.writerow(["Red (Missing)", summary["red"], f"{summary['red_pct']:.1f}%"])
        writer.writerow([])

        # Compliance results
        writer.writerow(
            ["Category", "Requirement", "Status", "Reason", "URL", "Last Updated"]
        )

        checklist = audit_data.get("checklist", {})
        results = audit_data.get("results", {})

        for category_id, category in checklist.items():
            for item in category["items"]:
                item_id = item["id"]
                result = results.get(item_id, {})

                writer.writerow(
                    [
                        category["name"],
                        item["name"],
                        result.get("status", "UNKNOWN"),
                        result.get("reason", "Not checked"),
                        result.get("url", "N/A"),
                        result.get("last_updated", "N/A"),
                    ]
                )

        # SEO Analysis
        if audit_data.get("seo_analysis"):
            writer.writerow([])
            writer.writerow(["SEO & Structure Analysis"])
            seo = audit_data["seo_analysis"]
            writer.writerow(["Metric", "Count"])
            writer.writerow(["Orphaned Pages", seo.get("orphaned_count", 0)])
            writer.writerow(["Deep Pages (>3 levels)", seo.get("deep_pages_count", 0)])
            writer.writerow(["Duplicate Titles", seo.get("duplicate_titles_count", 0)])
            writer.writerow(["Missing Titles", seo.get("missing_titles_count", 0)])

        # Link Analysis
        if audit_data.get("link_analysis"):
            writer.writerow([])
            writer.writerow(["Link Health Analysis"])
            links = audit_data["link_analysis"]
            writer.writerow(["Metric", "Count"])
            writer.writerow(["Broken Links (404)", links.get("broken_count", 0)])
            writer.writerow(["Redirects", links.get("redirect_count", 0)])
            writer.writerow(
                ["Missing Documents", links.get("missing_documents_count", 0)]
            )

        # Change Tracking
        if audit_data.get("change_tracking", {}).get("has_previous"):
            writer.writerow([])
            writer.writerow(["Changes Since Last Audit"])
            changes = audit_data["change_tracking"]
            writer.writerow(
                ["Previous Audit Date", changes.get("previous_date", "Unknown")]
            )
            writer.writerow(["Pages/Docs Added", changes.get("total_added", 0)])
            writer.writerow(["Pages/Docs Removed", changes.get("total_removed", 0)])
            writer.writerow(["Documents Updated", changes.get("updated_count", 0)])

        return output.getvalue()

    async def _generate_excel_report(self, audit_data: Dict[str, Any]) -> bytes:
        """Generate Excel report with multiple sheets"""
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            log.error("[EXPORT] openpyxl not installed, cannot generate Excel")
            raise

        wb = Workbook()

        # Sheet 1: Summary
        ws_summary = wb.active
        ws_summary.title = "Summary"

        ws_summary["A1"] = f"Website Audit Report"
        ws_summary["A1"].font = Font(size=16, bold=True)
        ws_summary["A2"] = audit_data["target_url"]
        ws_summary["A3"] = f"Generated: {audit_data['started']}"
        ws_summary["A4"] = f"Framework: {audit_data['framework']}"
        ws_summary["A5"] = f"Pages Analyzed: {audit_data['page_count']}"

        summary = audit_data["summary"]
        ws_summary["A7"] = "Status"
        ws_summary["B7"] = "Count"
        ws_summary["C7"] = "Percentage"
        ws_summary["A7"].font = Font(bold=True)
        ws_summary["B7"].font = Font(bold=True)
        ws_summary["C7"].font = Font(bold=True)

        ws_summary["A8"] = "Green (Compliant)"
        ws_summary["B8"] = summary["green"]
        ws_summary["C8"] = f"{summary['green_pct']:.1f}%"
        ws_summary["A8"].fill = PatternFill(
            start_color="C8E6C9", end_color="C8E6C9", fill_type="solid"
        )

        ws_summary["A9"] = "Amber (Partial)"
        ws_summary["B9"] = summary["amber"]
        ws_summary["C9"] = f"{summary['amber_pct']:.1f}%"
        ws_summary["A9"].fill = PatternFill(
            start_color="FFE0B2", end_color="FFE0B2", fill_type="solid"
        )

        ws_summary["A10"] = "Red (Missing)"
        ws_summary["B10"] = summary["red"]
        ws_summary["C10"] = f"{summary['red_pct']:.1f}%"
        ws_summary["A10"].fill = PatternFill(
            start_color="FFCDD2", end_color="FFCDD2", fill_type="solid"
        )

        # Sheet 2: Compliance Details
        ws_compliance = wb.create_sheet("Compliance Results")
        headers = ["Category", "Requirement", "Status", "Reason", "URL", "Last Updated"]
        ws_compliance.append(headers)

        for cell in ws_compliance[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                start_color="DDDDDD", end_color="DDDDDD", fill_type="solid"
            )

        checklist = audit_data.get("checklist", {})
        results = audit_data.get("results", {})

        for category_id, category in checklist.items():
            for item in category["items"]:
                item_id = item["id"]
                result = results.get(item_id, {})
                status = result.get("status", "UNKNOWN")

                row = [
                    category["name"],
                    item["name"],
                    status,
                    result.get("reason", "Not checked"),
                    result.get("url", "N/A"),
                    result.get("last_updated", "N/A"),
                ]
                ws_compliance.append(row)

                # Color code status cell
                status_cell = ws_compliance.cell(row=ws_compliance.max_row, column=3)
                if status == "GREEN":
                    status_cell.fill = PatternFill(
                        start_color="C8E6C9", end_color="C8E6C9", fill_type="solid"
                    )
                elif status == "AMBER":
                    status_cell.fill = PatternFill(
                        start_color="FFE0B2", end_color="FFE0B2", fill_type="solid"
                    )
                elif status == "RED":
                    status_cell.fill = PatternFill(
                        start_color="FFCDD2", end_color="FFCDD2", fill_type="solid"
                    )

        # Auto-size columns
        for column_cells in ws_compliance.columns:
            length = max(len(str(cell.value or "")) for cell in column_cells)
            ws_compliance.column_dimensions[column_cells[0].column_letter].width = min(
                length + 2, 50
            )

        # Sheet 3: SEO Analysis
        if audit_data.get("seo_analysis"):
            ws_seo = wb.create_sheet("SEO Analysis")
            seo = audit_data["seo_analysis"]

            ws_seo["A1"] = "SEO & Structure Metrics"
            ws_seo["A1"].font = Font(size=14, bold=True)

            ws_seo.append(["Metric", "Count"])
            ws_seo.append(["Orphaned Pages", seo.get("orphaned_count", 0)])
            ws_seo.append(["Deep Pages (>3 levels)", seo.get("deep_pages_count", 0)])
            ws_seo.append(["Duplicate Titles", seo.get("duplicate_titles_count", 0)])
            ws_seo.append(["Missing Titles", seo.get("missing_titles_count", 0)])

        # Sheet 4: Link Health
        if audit_data.get("link_analysis"):
            ws_links = wb.create_sheet("Link Health")
            links = audit_data["link_analysis"]

            ws_links["A1"] = "Link Health Metrics"
            ws_links["A1"].font = Font(size=14, bold=True)

            ws_links.append(["Metric", "Count"])
            ws_links.append(["Broken Links (404)", links.get("broken_count", 0)])
            ws_links.append(["Redirects", links.get("redirect_count", 0)])
            ws_links.append(
                ["Missing Documents", links.get("missing_documents_count", 0)]
            )

        # Save to bytes
        excel_buffer = BytesIO()
        wb.save(excel_buffer)
        return excel_buffer.getvalue()

    async def _upload_file_to_system(
        self, content: bytes, filename: str, user: Dict[str, Any], content_type: str
    ) -> Optional[str]:
        """Upload file to Files system for user download"""
        try:
            file_id = str(uuid.uuid4())

            # Upload file content to storage first (Storage.upload_file takes file, filename, and tags)
            file_bytes, file_path = Storage.upload_file(BytesIO(content), filename, [])
            log.info(f"[EXPORT] Uploaded file to storage: {file_path}")

            # Create file record with path
            file_form = FileForm(
                **{
                    "id": file_id,
                    "filename": filename,
                    "path": file_path,
                    "meta": {
                        "name": filename,
                        "content_type": content_type,
                        "size": len(content),
                        "source": "website_audit_export",
                    },
                }
            )

            file = Files.insert_new_file(user.get("id", "unknown"), file_form)
            log.info(f"[EXPORT] Created file record: {file_id}")

            return file_id

        except Exception as e:
            log.error(f"[EXPORT] Error uploading file: {e}")
            import traceback

            log.error(traceback.format_exc())
            return None

    async def upload_documents_to_kb(
        self,
        documents: List[Dict[str, str]],
        target_url: str,
        user: Dict[str, Any],
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """Upload documents to knowledge base with incremental updates"""

        try:
            # Extract user_id from user object
            user_id = user.get("id", "unknown") if user else "unknown"

            # Create stable collection name from domain
            domain = urlparse(target_url).netloc
            collection_name = f"Website Audit: {domain}"

            # Check if collection already exists
            existing_knowledge = None
            all_knowledge = Knowledges.get_knowledge_bases()
            for kb in all_knowledge:
                if kb.name == collection_name:
                    existing_knowledge = kb
                    break

            if existing_knowledge:
                log.info(f"[KB] Using existing collection: {collection_name}")
                knowledge = existing_knowledge

                # Get existing files in the collection
                existing_file_ids = knowledge.data.get("file_ids", [])
                existing_files = (
                    Files.get_files_by_ids(existing_file_ids)
                    if existing_file_ids
                    else []
                )

                # Verify files actually exist and filter out stale records
                valid_files = []
                for f in existing_files:
                    if f and f.id and f.hash:  # Basic validation
                        valid_files.append(f)
                    else:
                        log.warning(
                            f"[KB] Skipping invalid file record: {f.id if f else 'None'}"
                        )

                # If we lost files (deleted but KB still references them), reset the KB
                if existing_file_ids and len(valid_files) < len(existing_file_ids):
                    log.warning(
                        f"[KB] Found {len(existing_file_ids)} file IDs but only {len(valid_files)} valid files - resetting KB data"
                    )
                    Knowledges.update_knowledge_data_by_id(
                        knowledge.id, {"file_ids": []}
                    )
                    valid_files = []

                # Build map of URL -> file info for comparison
                existing_files_map = {}
                for f in valid_files:
                    source_url = f.meta.get("source") if f.meta else None
                    if source_url:
                        existing_files_map[source_url] = {
                            "id": f.id,
                            "hash": (
                                f.meta.get("file_hash") if f.meta else None
                            ),  # Use file bytes hash, not text hash
                            "filename": f.filename,
                        }

                log.info(
                    f"[KB] Found {len(existing_files_map)} valid existing files in collection"
                )
            else:
                log.info(f"[KB] Creating new collection: {collection_name}")

                # Create new knowledge base
                kb_form = KnowledgeForm(
                    name=collection_name,
                    description=f"Website audit documents from {target_url}",
                    data={"file_ids": []},
                )

                knowledge = Knowledges.insert_new_knowledge(user_id, kb_form)
                if not knowledge:
                    log.error("[KB] Failed to create knowledge base")
                    return None

                existing_files_map = {}
                log.info(f"[KB] Created knowledge base: {knowledge.id}")

            # Process documents: identify new, changed, and unchanged files
            current_urls = {
                doc["url"] for doc in documents
            }  # Process all documents for KB
            to_update = []  # Changed files (different hash)
            to_add = []  # New files
            to_keep = []  # Unchanged files
            to_delete = []  # Removed files

            # First pass: identify what needs to be done
            for doc in documents:  # Process all documents for KB
                url = doc["url"]
                if url in existing_files_map:
                    # File exists - we'll check hash after download
                    doc["existing_file"] = existing_files_map[url]
                    to_update.append(doc)
                else:
                    # New file
                    to_add.append(doc)

            # Find files to delete (in collection but not in current crawl)
            for url, file_info in existing_files_map.items():
                if url not in current_urls:
                    to_delete.append(file_info)

            log.info(
                f"[KB] Planning: {len(to_add)} new, {len(to_update)} to check, {len(to_delete)} to remove"
            )

            if event_emitter:
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Analyzing: {len(to_add)} new, {len(to_update)} to check, {len(to_delete)} to remove",
                            "done": False,
                        },
                    }
                )

            # Delete removed files
            for idx, file_info in enumerate(to_delete, 1):
                try:
                    file_id = file_info["id"]
                    filename = file_info["filename"]
                    log.info(f"[KB] Deleting removed file: {filename}")

                    # Delete from vector DB
                    file_collection = f"file-{file_id}"
                    if VECTOR_DB_CLIENT.has_collection(collection_name=file_collection):
                        VECTOR_DB_CLIENT.delete_collection(
                            collection_name=file_collection
                        )

                    # Remove from open_webui.main collection
                    if VECTOR_DB_CLIENT.has_collection(collection_name=knowledge.id):
                        VECTOR_DB_CLIENT.delete(
                            collection_name=knowledge.id, ids=[file_id]
                        )

                    # Delete file record
                    Files.delete_file_by_id(file_id)
                    log.info(f"[KB] Deleted: {filename}")

                    if event_emitter:
                        await event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Removing old files... ({idx}/{len(to_delete)})",
                                    "done": False,
                                },
                            }
                        )
                except Exception as e:
                    log.error(
                        f"[KB] Error deleting file {file_info.get('filename')}: {e}"
                    )

            # Process new and potentially changed files
            file_ids = []
            all_docs = to_add + to_update

            processed_count = 0
            unchanged_count = 0
            updated_count = 0
            new_count = 0

            for idx, doc in enumerate(all_docs, 1):
                try:
                    # Create unique filename using URL path to avoid collisions
                    url_path = urlparse(doc["url"]).path
                    base_filename = url_path.split("/")[-1]

                    # Create prefix from parent path (e.g., "2021-09" from "/uploads/2021/09/file.pdf")
                    path_parts = url_path.split("/")[:-1]  # All parts except filename
                    if len(path_parts) >= 2:
                        # Use last 2 path components as prefix (e.g., "2021-09")
                        prefix = "-".join(path_parts[-2:]).replace("/", "-")
                        filename = f"{prefix}_{base_filename}"[:255]
                    else:
                        filename = base_filename[:255]

                    # Only reuse file_id if file is truly unchanged (hash match)
                    # For new files or changed files, always create new file_id
                    file_id = str(uuid.uuid4())

                    # Handle webpages vs documents differently
                    if doc.get("type") == "WEBPAGE":
                        # For webpages, create a text file from existing content
                        log.info(
                            f"[KB] Processing webpage: {doc['title']} ({doc['url']})"
                        )

                        # Create text content with metadata
                        webpage_content = f"""Title: {doc['title']}
URL: {doc['url']}
Type: Webpage

{doc['text']}
"""
                        content = webpage_content.encode("utf-8")
                        content_type = "text/plain"

                        # Use title as filename (sanitized)
                        safe_title = re.sub(r"[^\w\s-]", "", doc["title"])[:100]
                        filename = f"webpage_{safe_title}.txt"
                    else:
                        # For documents, download the file
                        log.info(f"[KB] Downloading: {doc['url']}")

                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                doc["url"], timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                if response.status != 200:
                                    log.warning(
                                        f"[KB] Failed to download {doc['url']}: HTTP {response.status}"
                                    )
                                    # Keep existing file if download fails
                                    if "existing_file" in doc:
                                        file_ids.append(doc["existing_file"]["id"])
                                    continue

                                content = await response.read()
                                content_type = response.headers.get(
                                    "content-type", "application/pdf"
                                )

                    # Check if file has changed (for updates)
                    file_hash = hashlib.sha256(content).hexdigest()
                    if "existing_file" in doc:
                        if doc["existing_file"]["hash"] == file_hash:
                            # Hash matches - file unchanged, just reuse existing file_id
                            log.info(f"[KB] Unchanged: {filename}")
                            file_ids.append(doc["existing_file"]["id"])
                            unchanged_count += 1
                            processed_count += 1

                            if event_emitter and processed_count % 5 == 0:
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Processing files... ({processed_count}/{len(all_docs)}) [✓{unchanged_count} ↻{updated_count} +{new_count}]",
                                            "done": False,
                                        },
                                    }
                                )
                            continue
                        else:
                            # File changed - delete old embeddings only (using old file_id from existing_file)
                            old_file_id = doc["existing_file"]["id"]
                            log.info(
                                f"[KB] Updating changed file: {filename} (hash changed, old_id: {old_file_id})"
                            )
                            updated_count += 1
                            try:
                                # Delete old embeddings from vector DB
                                file_collection = f"file-{old_file_id}"
                                if VECTOR_DB_CLIENT.has_collection(
                                    collection_name=file_collection
                                ):
                                    VECTOR_DB_CLIENT.delete_collection(
                                        collection_name=file_collection
                                    )
                                    log.info(
                                        f"[KB] Deleted old embeddings for: {filename}"
                                    )

                                # Remove from open_webui.main KB collection
                                if VECTOR_DB_CLIENT.has_collection(
                                    collection_name=knowledge.id
                                ):
                                    VECTOR_DB_CLIENT.delete(
                                        collection_name=knowledge.id, ids=[old_file_id]
                                    )

                                # Delete old file record (will be replaced with new one with new file_id)
                                Files.delete_file_by_id(old_file_id)
                                log.info(f"[KB] Deleted old file record: {old_file_id}")

                                # Wait for deletions to propagate
                                await asyncio.sleep(0.2)
                            except Exception as e:
                                log.warning(
                                    f"[KB] Error deleting old file/embeddings: {e}"
                                )
                    else:
                        log.info(f"[KB] Adding new file: {filename}")
                        new_count += 1

                    # Check if a file with this hash already exists and has content
                    # Query all files and find one with matching hash
                    existing_file_by_hash = None
                    try:
                        all_files = Files.get_files()
                        for f in all_files:
                            if f.hash == file_hash:
                                existing_file_by_hash = f
                                break
                    except Exception as e:
                        log.warning(
                            f"[KB] Error checking for existing file by hash: {e}"
                        )
                        existing_file_by_hash = None

                    if (
                        existing_file_by_hash
                        and existing_file_by_hash.data
                        and existing_file_by_hash.data.get("content")
                    ):
                        # File exists with extracted content - reuse it!
                        log.info(
                            f"[KB] Reusing existing file with same content: {filename} (id: {existing_file_by_hash.id})"
                        )
                        file_id = existing_file_by_hash.id

                        # Just add to KB collection (Step 2 only)
                        try:
                            app = get_app()
                            if app and hasattr(app, "state"):
                                scope = {
                                    "type": "http",
                                    "method": "POST",
                                    "path": "/api/v1/retrieval/process/file",
                                    "headers": [],
                                    "query_string": b"",
                                    "server": ("localhost", 8080),
                                    "app": app,
                                }
                                mock_request = StarletteRequest(scope)

                                log.info(
                                    f"[KB] Adding existing file to KB collection: {filename}"
                                )
                                result = process_file(
                                    mock_request,
                                    ProcessFileForm(
                                        file_id=file_id, collection_name=knowledge.id
                                    ),
                                    user=UserObject(user),
                                )

                                if result and result.get("status"):
                                    file_ids.append(file_id)
                                    log.info(
                                        f"[KB] Successfully added existing file: {filename}"
                                    )
                                    processed_count += 1

                                    if event_emitter:
                                        await event_emitter(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": f"Processing files... ({processed_count}/{len(all_docs)}) [✓{unchanged_count} ↻{updated_count} +{new_count}]",
                                                    "done": False,
                                                },
                                            }
                                        )
                                else:
                                    log.warning(
                                        f"[KB] Failed to add existing file: {filename}"
                                    )
                                    file_ids.append(file_id)

                        except Exception as e:
                            error_msg = str(e)
                            log.warning(
                                f"[KB] Error adding existing file to KB: {error_msg}"
                            )
                            file_ids.append(file_id)

                        continue  # Skip to next file

                    elif existing_file_by_hash:
                        # File exists but has no content - delete the broken record
                        log.warning(
                            f"[KB] Found broken file record with no content, deleting: {existing_file_by_hash.id}"
                        )
                        try:
                            Files.delete_file_by_id(existing_file_by_hash.id)
                        except Exception as e:
                            log.warning(f"[KB] Error deleting broken file: {e}")

                    # Upload file to storage (wrap bytes in BytesIO)
                    file_bytes, file_path = Storage.upload_file(
                        BytesIO(content), filename, []
                    )
                    log.info(f"[KB] Saved to storage: {filename} at path: {file_path}")

                    # Verify file was actually saved
                    import os

                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        log.info(f"[KB] File exists on disk: {file_size} bytes")
                    else:
                        log.error(
                            f"[KB] WARNING: File NOT found on disk at {file_path}"
                        )

                    # Store original URL in 'data' field (persists in file record)
                    # The 'source' metadata field will be overwritten by embedding system
                    file_form = FileForm(
                        id=file_id,
                        hash=file_hash,
                        filename=filename,
                        path=file_path,  # Set path here
                        data={
                            "source_url": doc["url"]
                        },  # Store URL in data field for lookup by file_id
                        meta={
                            "source": doc["url"],
                            "content_type": content_type,
                            "size": len(content),
                            "name": filename,
                            "file_hash": file_hash,  # Store file bytes hash for comparison on next run
                            "item_type": doc.get(
                                "type", "DOCUMENT"
                            ),  # Mark as WEBPAGE or DOCUMENT
                            "title": doc.get("title", ""),  # Store original title
                        },
                    )

                    log.info(
                        f"[KB] DEBUG FileForm: id={file_id}, path={file_path}, filename={filename}"
                    )

                    # Save to database (always insert with fresh file_id)
                    result = Files.insert_new_file(user_id, file_form)
                    if not result:
                        log.error(f"[KB] Failed to create file record: {file_id}")
                        continue

                    # Verify what was actually saved
                    saved_file = Files.get_file_by_id(file_id)
                    if saved_file:
                        log.info(
                            f"[KB] DEBUG Saved file: id={saved_file.id}, path={saved_file.path}, data_keys={list(saved_file.data.keys()) if saved_file.data else []}"
                        )

                    log.info(f"[KB] Created new file record: {file_id}")

                    # Small delay to ensure file is fully written to storage and DB is committed
                    await asyncio.sleep(0.1)

                    # Process file with embeddings
                    try:

                        # Get app instance to access app.state
                        app = get_app()
                        if app and hasattr(app, "state"):
                            # Create a mock Request with app.state
                            scope = {
                                "type": "http",
                                "method": "POST",
                                "path": "/api/v1/retrieval/process/file",
                                "headers": [],
                                "query_string": b"",
                                "server": ("localhost", 8080),
                                "app": app,  # CRITICAL: Must be in scope, not as _app attribute
                            }
                            mock_request = StarletteRequest(scope)

                            # Process file with embeddings - TWO STEP like manual upload
                            try:
                                log.info(
                                    f"[KB] Step 1: Extracting text from {filename}"
                                )
                                # Step 1: Extract text (no collection_name)
                                result1 = process_file(
                                    mock_request,
                                    ProcessFileForm(
                                        file_id=file_id
                                        # NO collection_name - this extracts text
                                    ),
                                    user=UserObject(user),
                                )

                                if not (result1 and result1.get("status")):
                                    log.error(
                                        f"[KB] Text extraction failed for {filename}"
                                    )
                                    continue

                                log.info(
                                    f"[KB] Step 2: Adding to KB collection: {filename}"
                                )
                                # Step 2: Add to KB collection
                                result2 = process_file(
                                    mock_request,
                                    ProcessFileForm(
                                        file_id=file_id,
                                        collection_name=knowledge.id,  # Now add to KB
                                    ),
                                    user=UserObject(user),
                                )

                                if result2 and result2.get("status"):
                                    file_ids.append(file_id)
                                    log.info(f"[KB] Successfully processed: {filename}")
                                    processed_count += 1

                                    if event_emitter:
                                        await event_emitter(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": f"Processing files... ({processed_count}/{len(all_docs)}) [✓{unchanged_count} ↻{updated_count} +{new_count}]",
                                                    "done": False,
                                                },
                                            }
                                        )
                                else:
                                    log.warning(f"[KB] Processing failed: {filename}")
                                    file_ids.append(file_id)  # Still add to list

                            except Exception as e:
                                error_msg = str(e)
                                # Handle duplicate hash error gracefully
                                if (
                                    "Duplicate content" in error_msg
                                    or "already exists" in error_msg
                                ):
                                    log.warning(
                                        f"[KB] Duplicate hash detected for {filename} - likely a timing issue"
                                    )
                                    # File is already in DB, just add the ID to our list
                                    file_ids.append(file_id)
                                    processed_count += 1
                                elif "empty" in error_msg.lower():
                                    log.warning(
                                        f"[KB] Empty content in {filename}, skipping embeddings"
                                    )
                                    file_ids.append(file_id)  # Still add file record
                                else:
                                    log.error(
                                        f"[KB] Error processing embeddings for {filename}: {e}"
                                    )
                                    file_ids.append(
                                        file_id
                                    )  # Still add file even if embedding fails
                        else:
                            log.warning(
                                f"[KB] App.state not available - file saved but no embeddings: {filename}"
                            )
                            file_ids.append(file_id)

                    except Exception as e:
                        log.error(f"[KB] Error saving document {filename}: {e}")
                        continue

                except Exception as e:
                    log.error(f"[KB] Error downloading {doc['url']}: {e}")
                    continue

            # Update KB with file IDs (keep all files, including unchanged ones)
            if file_ids:
                Knowledges.update_knowledge_data_by_id(
                    knowledge.id, {"file_ids": file_ids}
                )
                added = len(to_add)
                updated = len(
                    [
                        d
                        for d in to_update
                        if "existing_file" in d and d.get("_was_updated")
                    ]
                )
                unchanged = len(
                    [
                        d
                        for d in to_update
                        if "existing_file" in d and not d.get("_was_updated")
                    ]
                )
                deleted = len(to_delete)

                log.info(
                    f"[KB] Collection updated: +{added} new, ~{updated} updated, ={unchanged} unchanged, -{deleted} deleted"
                )
                log.info(f"[KB] Total files in collection: {len(file_ids)}")
                log.info(f"[KB] KB ID: {knowledge.id}")
                return knowledge.id
            else:
                log.warning("[KB] No documents in collection after processing")
                # Only delete KB if it's a new empty collection
                if not existing_knowledge:
                    try:
                        Knowledges.delete_knowledge_by_id(knowledge.id)
                        log.info(f"[KB] Deleted empty knowledge base: {knowledge.id}")
                    except Exception as e:
                        log.error(f"[KB] Failed to delete empty KB: {e}")
                return knowledge.id if existing_knowledge else None

        except Exception as e:
            log.error(f"[KB] Error uploading documents: {e}")
            return None

    async def analyze_documents_with_rag(
        self, collection_id: str, compliance_checklist: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Use RAG to analyze documents for compliance requirements"""

        if not collection_id:
            return {}

        results = {}

        try:
            log.info(f"[RAG] Starting analysis on collection {collection_id}")

            for category_id, category in compliance_checklist.items():
                for item in category["items"]:
                    item_id = item["id"]
                    item_name = item["name"]
                    keywords = item.get("keywords", [])

                    log.info(f"[RAG] Querying KB for: {item_name}")

                    # Create search query
                    query = f"Find information about {item_name}. Keywords: {', '.join(keywords)}"

                    try:
                        # Get embedding function for queries
                        app = get_app()
                        embedding_func = get_embedding_function(
                            embedding_engine=app.state.config.RAG_EMBEDDING_ENGINE,
                            embedding_model=app.state.config.RAG_EMBEDDING_MODEL,
                            embedding_function=app.state.ef,
                            url=(
                                app.state.config.RAG_OPENAI_API_BASE_URL
                                if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                                else (
                                    app.state.config.RAG_OLLAMA_BASE_URL
                                    if app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                                    else app.state.config.RAG_AZURE_OPENAI_BASE_URL
                                )
                            ),
                            key=(
                                app.state.config.RAG_OPENAI_API_KEY
                                if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                                else (
                                    app.state.config.RAG_OLLAMA_API_KEY
                                    if app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                                    else app.state.config.RAG_AZURE_OPENAI_API_KEY
                                )
                            ),
                            embedding_batch_size=app.state.config.RAG_EMBEDDING_BATCH_SIZE,
                            azure_api_version=(
                                app.state.config.RAG_AZURE_OPENAI_API_VERSION
                                if app.state.config.RAG_EMBEDDING_ENGINE
                                == "azure_openai"
                                else None
                            ),
                        )

                        # Search the vector database using query_collection
                        # Run in executor since query_collection is synchronous
                        loop = asyncio.get_running_loop()
                        search_results = await loop.run_in_executor(
                            None,
                            lambda: query_collection(
                                collection_names=[collection_id],
                                queries=[query],
                                embedding_function=embedding_func,
                                k=3,  # Top 3 results
                            ),
                        )

                        # Process results
                        if search_results and search_results.get("documents"):
                            # Get documents and scores
                            docs = search_results.get("documents", [[]])[
                                0
                            ]  # Extract first query results
                            distances = search_results.get("distances", [[]])[0]
                            metadatas = search_results.get("metadatas", [[]])[0]

                            if docs and distances:
                                # Convert distance to similarity score (lower distance = higher similarity)
                                # ChromaDB returns cosine distance: 0 (identical) to 2 (opposite)
                                # Convert to similarity: 1 (identical) to 0 (opposite)
                                distance = distances[0]
                                relevance_score = max(
                                    0, min(1, (2 - distance) / 2)
                                )  # Normalize 0-2 to 1-0

                                # Use LLM assessment if enabled
                                if (
                                    self.valves.ENABLE_LLM_ASSESSMENT
                                    and relevance_score > 0.4
                                ):  # Only assess if somewhat relevant
                                    log.info(
                                        f"[RAG] {item_name}: Using LLM assessment (similarity: {relevance_score:.2f})"
                                    )
                                    llm_result = await self.assess_compliance_with_llm(
                                        item=item,
                                        documents=docs[:3],  # Top 3 docs
                                        metadatas=metadatas[:3],
                                        similarity_score=relevance_score,
                                    )

                                    if llm_result.get("assessed"):
                                        results[item_id] = llm_result
                                        log.info(
                                            f"[RAG] {item_name}: {llm_result['status']} (LLM confidence: {llm_result.get('confidence', 0):.0f}%)"
                                        )
                                    else:
                                        # Fallback to similarity scoring
                                        log.warning(
                                            f"[RAG] {item_name}: LLM assessment failed, using similarity"
                                        )
                                        results[item_id] = (
                                            self._similarity_based_result(
                                                item_id,
                                                item_name,
                                                relevance_score,
                                                docs,
                                                metadatas,
                                            )
                                        )
                                else:
                                    # Use similarity-based scoring
                                    results[item_id] = self._similarity_based_result(
                                        item_id,
                                        item_name,
                                        relevance_score,
                                        docs,
                                        metadatas,
                                    )
                                    log.info(
                                        f"[RAG] {item_name}: {results[item_id]['status']} (similarity: {relevance_score:.2f})"
                                    )
                            else:
                                results[item_id] = {
                                    "status": "RED",
                                    "found": False,
                                    "reason": "No relevant documents found",
                                    "evidence": "",
                                }
                        else:
                            results[item_id] = {
                                "status": "RED",
                                "found": False,
                                "reason": "No relevant documents found in knowledge base",
                                "evidence": "",
                            }
                            log.info(f"[RAG] {item_name}: Not found")

                    except Exception as e:
                        log.error(f"[RAG] Error querying for {item_name}: {e}")
                        continue

            log.info(f"[RAG] Analyzed {len(results)} requirements")
            return results

        except Exception as e:
            log.error(f"[RAG] Error in RAG analysis: {e}")
            return {}

    def _get_file_url(self, file_id: str) -> Optional[str]:
        """Look up the original URL from file record using file_id"""
        try:
            file_record = Files.get_file_by_id(file_id)
            if file_record:
                # Try data.source_url first (webpages and documents)
                if file_record.data and file_record.data.get("source_url"):
                    return file_record.data.get("source_url")
                # Try meta.source as fallback
                if file_record.meta and file_record.meta.get("source"):
                    source = file_record.meta.get("source")
                    # Only use if it looks like a URL (not a filename)
                    if source and (
                        source.startswith("http://") or source.startswith("https://")
                    ):
                        return source
        except Exception as e:
            log.debug(f"[URL_LOOKUP] Could not get URL for file_id {file_id}: {e}")
        return None

    def _similarity_based_result(
        self,
        item_id: str,
        item_name: str,
        relevance_score: float,
        docs: List[str],
        metadatas: List[Dict],
    ) -> Dict[str, Any]:
        """Fallback: Use similarity scoring when LLM not available"""

        if relevance_score > 0.7:
            status = "GREEN"
            reason = f"Found relevant document with high confidence ({int(relevance_score * 100)}%)"
        elif relevance_score > 0.5:
            status = "AMBER"
            reason = f"Found potentially relevant document ({int(relevance_score * 100)}% confidence)"
        else:
            status = "AMBER"
            reason = (
                f"Found document with low confidence ({int(relevance_score * 100)}%)"
            )

        # Build source references with URLs
        sources = []
        for idx, meta in enumerate(metadatas[:3]):  # Top 3 sources
            # Look up URL from file record using file_id
            file_id = meta.get("file_id")
            url = self._get_file_url(file_id) if file_id else None
            # Fallback to metadata fields if lookup fails
            if not url:
                url = meta.get("url") or meta.get("source")
            # Final fallback: extract URL from document content (format: "URL: https://...")
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                doc_text = docs[idx] if idx < len(docs) else ""
                url_match = re.search(
                    r"^URL:\s*(https?://[^\s]+)", doc_text, re.MULTILINE
                )
                if url_match:
                    url = url_match.group(1)

            source_ref = {
                "name": meta.get("name", "Unknown"),
                "page": meta.get("page_label", meta.get("page", "?")),
                "url": url,
            }
            sources.append(source_ref)

        # Extract first URL for backward compatibility
        first_url = None
        for idx, meta in enumerate(metadatas):
            file_id = meta.get("file_id")
            url = self._get_file_url(file_id) if file_id else None
            if not url:
                url = meta.get("url") or meta.get("source")
            # Final fallback: extract URL from document content
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                doc_text = docs[idx] if idx < len(docs) else ""
                url_match = re.search(
                    r"^URL:\s*(https?://[^\s]+)", doc_text, re.MULTILINE
                )
                if url_match:
                    url = url_match.group(1)
            if url:
                first_url = url
                break

        return {
            "status": status,
            "found": True,
            "reason": reason,
            "evidence": docs[0][:300] if docs else "",
            "relevance_score": relevance_score,
            "document_count": len(docs),
            "document_name": metadatas[0].get("name") if metadatas else None,
            "sources": sources,
            "url": first_url,
        }

    async def assess_compliance_with_llm(
        self,
        item: Dict[str, Any],
        documents: List[str],
        metadatas: List[Dict],
        similarity_score: float,
    ) -> Dict[str, Any]:
        """Use LLM to assess if documents meet compliance requirements via direct API"""

        # Check if API key is configured
        if not self.valves.OPENAI_API_KEY:
            log.warning("[LLM] OPENAI_API_KEY not configured, skipping LLM assessment")
            return {"assessed": False, "error": "api_key_missing"}

        try:
            # Build context from retrieved documents
            context_parts = []
            source_urls = []  # Track source URLs for reference
            for idx, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                doc_name = meta.get("name", "Unknown")
                page = meta.get("page_label", meta.get("page", "?"))
                # Look up URL from file record using file_id
                file_id = meta.get("file_id")
                source_url = self._get_file_url(file_id) if file_id else None
                # Fallback to metadata fields if lookup fails
                if not source_url:
                    source_url = meta.get("url") or meta.get("source")
                # Final fallback: extract URL from webpage content (format: "URL: https://...")
                if not source_url or not (
                    source_url.startswith("http://")
                    or source_url.startswith("https://")
                ):
                    url_match = re.search(
                        r"^URL:\s*(https?://[^\s]+)", doc, re.MULTILINE
                    )
                    if url_match:
                        source_url = url_match.group(1)
                if source_url and source_url not in source_urls:
                    source_urls.append(source_url)

                # Use full chunk - RAG already returned the most relevant content
                # Chunks are pre-sized by the text splitter (typically 500-1500 chars)
                context_parts.append(f"Document {idx}: {doc_name} (Page {page})\n{doc}")

            context = "\n\n---\n\n".join(context_parts)

            # Debug: Log what content is being sent to LLM
            log.info(
                f"[LLM] Context for '{item['name']}' ({len(context)} chars):\n{context[:800]}..."
            )

            # Build compliance assessment prompt
            school_type_display = (
                self.detected_school_type.replace("-", " ").title()
                if self.detected_school_type
                else "Unknown"
            )
            prompt = f"""You are a compliance auditor for UK maintained schools.

REGULATORY AUTHORITY: UK Department for Education (DfE)
SOURCE: {self.valves.GOV_UK_GUIDANCE_URL}
LEGAL BASIS: School Information (England) Regulations 2008 (as amended), Equality Act 2010, Children and Families Act 2014

SCHOOL TYPE: {school_type_display}

REQUIREMENT: {item['name']}
CRITICAL: {'Yes - legally required (MUST publish)' if item.get('critical') else 'No - recommended best practice (SHOULD publish)'}
KEYWORDS: {', '.join(item.get('keywords', []))}

RETRIEVED DOCUMENTS:
{context}

ASSESSMENT TASK:
Determine if the school is compliant with this requirement based on the retrieved documents.
Consider the school type when assessing (e.g., careers programmes are for secondary schools only).

EVALUATION CRITERIA:
1. Does a relevant, current document exist that addresses this requirement?
2. Is the document dated within the last 2-3 years (if applicable)?
3. Does the content specifically cover the required areas per GOV.UK guidance?
4. Are there any gaps or concerns?

Respond in this EXACT JSON format:
{{
  "status": "GREEN|AMBER|RED",
  "confidence": 0-100,
  "evidence": "Specific quotes or details from the document",
  "reason": "Brief explanation of the status",
  "action": "What's needed (if AMBER/RED) or 'None' if GREEN",
  "document_date": "Extracted date if found, or 'Unknown'"
}}

STATUS GUIDELINES:
- GREEN: Fully compliant, current document exists, no concerns
- AMBER: Partially compliant, outdated, or minor gaps
- RED: Missing, non-compliant, or critical issues

Respond ONLY with valid JSON, no additional text."""

            # Call LLM via direct OpenAI SDK
            log.info(
                f"[LLM] Calling {self.valves.LLM_MODEL_FOR_ASSESSMENT} for {item['name']}"
            )

            try:
                client = AsyncOpenAI(
                    api_key=self.valves.OPENAI_API_KEY,
                    base_url=self.valves.OPENAI_BASE_URL,
                    timeout=self.valves.LLM_TIMEOUT_SECONDS,
                )

                response = await client.chat.completions.create(
                    model=self.valves.LLM_MODEL_FOR_ASSESSMENT,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=500,  # Use max_completion_tokens for newer OpenAI models
                )

                log.debug(f"[LLM] API call successful, model: {response.model}")

            except Exception as call_error:
                log.error(
                    f"[LLM] API call failed: {type(call_error).__name__}: {call_error}"
                )
                import traceback

                log.error(f"[LLM] Traceback:\n{traceback.format_exc()}")
                return {
                    "assessed": False,
                    "error": f"api_call_failed: {str(call_error)}",
                }

            # Parse LLM response (OpenAI SDK format)
            try:
                llm_text = response.choices[0].message.content

                if not llm_text:
                    log.error(f"[LLM] No content in response from {response.model}")
                    return {"assessed": False, "error": "empty response"}

                log.debug(f"[LLM] Got response: {llm_text[:200]}...")

            except (AttributeError, IndexError) as parse_error:
                log.error(
                    f"[LLM] Failed to extract content from response: {parse_error}"
                )
                return {"assessed": False, "error": "response parsing failed"}

            # Extract JSON from response
            import json

            # Try to find JSON in the response
            json_start = llm_text.find("{")
            json_end = llm_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    json_str = llm_text[json_start:json_end]
                    llm_result = json.loads(json_str)

                    log.info(
                        f"[LLM] Assessment: {llm_result.get('status')} - {llm_result.get('reason', '')[:100]}"
                    )

                    # Build source references with URLs
                    sources = []
                    for idx, meta in enumerate(metadatas[:3]):  # Top 3 sources
                        # Look up URL from file record using file_id
                        file_id = meta.get("file_id")
                        url = self._get_file_url(file_id) if file_id else None
                        # Fallback to metadata fields if lookup fails
                        if not url:
                            url = meta.get("url") or meta.get("source")
                        # Final fallback: extract URL from document content (format: "URL: https://...")
                        if not url or not (
                            url.startswith("http://") or url.startswith("https://")
                        ):
                            doc_text = documents[idx] if idx < len(documents) else ""
                            url_match = re.search(
                                r"^URL:\s*(https?://[^\s]+)", doc_text, re.MULTILINE
                            )
                            if url_match:
                                url = url_match.group(1)

                        source_ref = {
                            "name": meta.get("name", "Unknown"),
                            "page": meta.get("page_label", meta.get("page", "?")),
                            "url": url,
                        }
                        sources.append(source_ref)

                    return {
                        "assessed": True,
                        "status": llm_result.get("status", "AMBER"),
                        "found": True,
                        "confidence": llm_result.get("confidence", 50),
                        "evidence": llm_result.get("evidence", ""),
                        "reason": llm_result.get("reason", "LLM assessment completed"),
                        "action": llm_result.get("action"),
                        "document_date": llm_result.get("document_date"),
                        "document_name": (
                            metadatas[0].get("name") if metadatas else None
                        ),
                        "similarity_score": similarity_score,
                        "sources": sources,  # List of source references with URLs
                        "url": (
                            source_urls[0] if source_urls else None
                        ),  # Primary URL for backward compatibility
                    }
                except json.JSONDecodeError as json_error:
                    log.error(f"[LLM] JSON parse error: {json_error}")
                    log.error(f"[LLM] JSON string: {json_str[:200]}")
                    return {"assessed": False, "error": "json parse error"}
            else:
                log.error(f"[LLM] Could not find JSON in response: {llm_text[:200]}")
                return {"assessed": False, "error": "no json found"}

        except Exception as e:
            log.error(
                f"[LLM] Unexpected error in LLM assessment: {type(e).__name__}: {e}"
            )
            import traceback

            log.error(f"[LLM] Full traceback:\n{traceback.format_exc()}")
            return {"assessed": False, "error": str(e)}

    def get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached audit data"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        log.info(f"[CACHE] Checking cache file: {cache_file}")

        if not cache_file.exists():
            log.info(f"[CACHE] Cache file not found")
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            log.info(f"[CACHE] Cache file loaded successfully")

            # Check if cache is still valid
            cache_time = datetime.fromisoformat(data["timestamp"])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600

            log.info(
                f"[CACHE] Cache age: {age_hours:.1f}h (TTL: {self.valves.CACHE_TTL_HOURS}h)"
            )

            if age_hours > self.valves.CACHE_TTL_HOURS:
                log.info(f"[CACHE] ⏰ Cache EXPIRED ({age_hours:.1f}h old)")
                return None

            log.info(f"[CACHE] ✅ Cache is valid")
            return data
        except Exception as e:
            log.error(f"[CACHE] ❌ Error loading cache: {e}")
            return None

    def save_to_cache(self, cache_key: str, pages: List[Dict[str, Any]]) -> None:
        """Save audit data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        log.info(f"[CACHE] Saving to cache file: {cache_file}")

        try:
            data = {"timestamp": datetime.now().isoformat(), "pages": pages}

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            file_size = cache_file.stat().st_size / 1024  # KB
            log.info(
                f"[CACHE] ✅ Saved {len(pages)} pages to cache ({file_size:.1f}KB)"
            )
        except Exception as e:
            log.error(f"[CACHE] ❌ Error saving cache: {e}")

    def load_page_cache(self, cache_key: str) -> Dict[str, Any]:
        """Load individual page cache with HTTP headers"""
        cache_file = self.cache_dir / f"{cache_key}_pages.json"

        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading page cache: {e}")
            return {}

    def save_page_cache(self, cache_key: str, page_cache: Dict[str, Any]) -> None:
        """Save individual page cache with HTTP headers"""
        cache_file = self.cache_dir / f"{cache_key}_pages.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(page_cache, f, ensure_ascii=False, indent=2)

            log.debug(f"Saved page cache with {len(page_cache)} entries")
        except Exception as e:
            log.error(f"Error saving page cache: {e}")

    async def run_audit_collect(
        self,
        target_url: str,
        user: Optional[dict] = None,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
        event_call: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Dict[str, Any]:
        """Run audit and collect all data (for HTML rendering)"""

        start_time = datetime.now()

        # Phase 1: Crawl
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": "Crawling website...", "done": False},
                }
            )

        cache_key = self.get_cache_key(target_url)
        log.info(f"[CACHE] Cache key: {cache_key}")
        log.info(f"[CACHE] Cache directory: {self.cache_dir}")

        cached_data = None
        if self.valves.ENABLE_CACHE and not self.valves.FORCE_REFRESH:
            log.info(f"[CACHE] Attempting to load from cache...")
            cached_data = self.load_from_cache(cache_key)
        else:
            log.info(f"[CACHE] Cache disabled or force refresh enabled")

        if cached_data:
            pages = cached_data["pages"]
            cache_age = (
                datetime.now() - datetime.fromisoformat(cached_data["timestamp"])
            ).total_seconds() / 3600
            crawl_stats = {
                "cached": len(pages),
                "unchanged": 0,
                "fetched": 0,
                "used_cache": True,
                "cache_age": cache_age,
            }
            log.info(
                f"[CACHE] ✅ Cache HIT! Loaded {len(pages)} pages (age: {cache_age:.1f}h)"
            )
        else:
            log.info(f"[CACHE] ❌ Cache MISS. Starting fresh crawl...")
            pages = []
            crawl_stats = {
                "cached": 0,
                "fetched": 0,
                "unchanged": 0,
                "used_cache": False,
            }
            collected_cookies = {}
            async for update in self.crawl_website(target_url, cache_key):
                pages = update.get("pages", [])
                crawl_stats = update.get("stats", crawl_stats)
                collected_cookies = update.get("cookies", {})
                if self.valves.DEBUG_MODE and len(pages) % 20 == 0:
                    log.info(
                        f"[CRAWL] Progress: {len(pages)} pages, Stats: {crawl_stats}"
                    )

            log.info(f"[CRAWL] Complete: {len(pages)} pages discovered, {len(collected_cookies)} cookies collected")
            log.info(f"[CRAWL] Final stats: {crawl_stats}")

            if self.valves.ENABLE_CACHE:
                log.info(f"[CACHE] Saving {len(pages)} pages to cache...")
                self.save_to_cache(cache_key, pages)
                log.info(f"[CACHE] ✅ Cache saved successfully")

        # Phase 2: Analyze
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": "Analyzing documents...", "done": False},
                }
            )

        document_map = await self.analyze_documents(pages, target_url)

        # Phase 2.5: Upload documents and webpages, perform RAG analysis (optional)
        rag_results = {}
        if self.valves.ENABLE_DOCUMENT_UPLOAD or self.valves.ENABLE_RAG_ANALYSIS:
            if event_emitter:
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": "Collecting content for KB...",
                            "done": False,
                        },
                    }
                )

            # Collect document URLs
            documents = await self.collect_document_urls(pages)
            log.info(f"[DOCUMENTS] Collected {len(documents)} documents")

            # Collect relevant webpages (scored by compliance relevance)
            webpages = self.collect_relevant_webpages(
                pages, max_pages=self.valves.MAX_WEBPAGES_FOR_RAG
            )
            log.info(f"[WEBPAGES] Collected {len(webpages)} relevant webpages")

            # Generate sitemap and run analyses if enabled
            sitemap_item = None
            seo_analysis = None
            link_analysis = None
            change_tracking = None

            if self.valves.GENERATE_SITEMAP:
                log.info(f"[SITEMAP] Generating sitemap for {len(pages)} pages")
                sitemap_markdown = self.generate_sitemap_markdown(pages, target_url)

                # Create sitemap as a webpage item for KB
                sitemap_item = {
                    "url": f"{target_url}#sitemap",
                    "type": "WEBPAGE",
                    "title": f"Site Map - {urlparse(target_url).netloc}",
                    "text": sitemap_markdown,
                }
                log.info(
                    f"[SITEMAP] Generated sitemap markdown ({len(sitemap_markdown)} chars)"
                )

                # Run SEO/structure analysis
                seo_analysis = self.analyze_site_structure(pages, target_url)

                # Detect broken links
                link_analysis = self.detect_broken_links(pages)

                # Load previous sitemap and compare changes
                previous_sitemap = self.load_sitemap_snapshot(target_url)
                change_tracking = self.compare_sitemaps(pages, previous_sitemap)

                # Save current snapshot for future comparisons
                self.save_sitemap_snapshot(pages, target_url)

            # Combine documents, webpages, and sitemap for KB upload
            all_content = documents + webpages
            if sitemap_item:
                all_content.append(sitemap_item)
            log.info(
                f"[KB] Total content to process: {len(documents)} documents + {len(webpages)} webpages + {'1 sitemap' if sitemap_item else '0 sitemaps'} = {len(all_content)} items"
            )

            if self.valves.ENABLE_DOCUMENT_UPLOAD and all_content:
                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Uploading {len(all_content)} items to KB ({len(documents)} docs, {len(webpages)} pages)...",
                                "done": False,
                            },
                        }
                    )

                # Upload to knowledge base (handles both documents and webpages)
                collection_id = await self.upload_documents_to_kb(
                    all_content, target_url, user, event_emitter
                )

                # Give ChromaDB a moment to finalize the collection
                if collection_id:
                    await asyncio.sleep(0.5)

                # Detect school type from KB content (more reliable than cached pages)
                if self.valves.SCHOOL_TYPE == "auto" and collection_id:
                    self.detected_school_type = await self.detect_school_type_from_kb(
                        collection_id
                    )
                    log.info(
                        f"[SCHOOL TYPE] Auto-detected from KB: {self.detected_school_type}"
                    )
                elif self.valves.SCHOOL_TYPE == "auto":
                    # Fallback to page-based detection
                    self.detected_school_type = self.detect_school_type(pages)
                    log.info(
                        f"[SCHOOL TYPE] Auto-detected from pages: {self.detected_school_type}"
                    )
                else:
                    self.detected_school_type = self.valves.SCHOOL_TYPE.lower()
                    log.info(
                        f"[SCHOOL TYPE] Using configured: {self.detected_school_type}"
                    )

                # Filter checklist based on school type
                if self.detected_school_type in ["primary", "secondary"]:
                    log.info(
                        f"[CHECKLIST] Filtering requirements for {self.detected_school_type} school"
                    )
                    self.compliance_checklist = self.get_filtered_checklist(
                        self.detected_school_type
                    )
                else:
                    log.info(
                        f"[CHECKLIST] Using full checklist (school type: {self.detected_school_type})"
                    )
                    self.compliance_checklist = self._default_checklist

                if collection_id and self.valves.ENABLE_RAG_ANALYSIS:
                    if event_emitter:
                        await event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "description": "Analyzing documents with RAG...",
                                    "done": False,
                                },
                            }
                        )

                    # Analyze with RAG (LLM assessment via direct API if enabled)
                    rag_results = await self.analyze_documents_with_rag(
                        collection_id, self.compliance_checklist
                    )
                    log.info(f"[RAG] Analyzed {len(rag_results)} requirements with RAG")

        # Fallback: Ensure checklist is set even if KB is disabled
        if not self.compliance_checklist:
            if self.valves.SCHOOL_TYPE == "auto":
                self.detected_school_type = self.detect_school_type(pages)
                log.info(
                    f"[SCHOOL TYPE] Auto-detected (no KB): {self.detected_school_type}"
                )
            else:
                self.detected_school_type = self.valves.SCHOOL_TYPE.lower()
                log.info(f"[SCHOOL TYPE] Using configured: {self.detected_school_type}")

            if self.detected_school_type in ["primary", "secondary"]:
                log.info(
                    f"[CHECKLIST] Filtering requirements for {self.detected_school_type} school"
                )
                self.compliance_checklist = self.get_filtered_checklist(
                    self.detected_school_type
                )
            else:
                log.info(
                    f"[CHECKLIST] Using full checklist (school type: {self.detected_school_type})"
                )
                self.compliance_checklist = self._default_checklist

        # Phase 3: Check compliance
        if event_emitter:
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": "Checking compliance...", "done": False},
                }
            )

        results = await self.check_compliance(document_map, event_call)

        # Merge RAG results if available
        if rag_results:
            log.info(f"[RAG] Merging RAG results with keyword-based results")
            # RAG results take precedence over keyword matching
            for item_id, rag_result in rag_results.items():
                if rag_result.get("found"):
                    results[item_id] = rag_result

        # Calculate summary
        total_items = sum(
            len(cat["items"]) for cat in self.compliance_checklist.values()
        )
        green_count = sum(1 for r in results.values() if r["status"] == "GREEN")
        amber_count = sum(1 for r in results.values() if r["status"] == "AMBER")
        red_count = sum(1 for r in results.values() if r["status"] == "RED")

        elapsed = (datetime.now() - start_time).total_seconds()

        # Generate mermaid sitemap for visual report
        sitemap_mermaid = None
        if self.valves.GENERATE_SITEMAP:
            sitemap_mermaid = self.generate_sitemap_mermaid(pages, max_nodes=50)
            log.info(
                f"[SITEMAP] Generated mermaid diagram ({len(sitemap_mermaid)} chars)"
            )

        # Analyze cookies if enabled and collected
        cookie_audit = None
        if self.valves.ENABLE_COOKIE_AUDIT:
            cookies_to_analyze = collected_cookies if 'collected_cookies' in locals() else {}
            cookie_audit = self.analyze_cookies(cookies_to_analyze, pages)
            log.info(f"[COOKIES] Audit complete: {cookie_audit.get('total_cookies', 0)} cookies, status: {cookie_audit.get('status', 'N/A')}")

        return {
            "target_url": target_url,
            "framework": self.valves.REGULATORY_FRAMEWORK,
            "school_type": self.detected_school_type,
            "started": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": elapsed,
            "pages": pages,
            "page_count": len(pages),
            "crawl_stats": crawl_stats,
            "sitemap_mermaid": sitemap_mermaid,
            "seo_analysis": seo_analysis,
            "link_analysis": link_analysis,
            "change_tracking": change_tracking,
            "results": results,
            "summary": {
                "total": total_items,
                "green": green_count,
                "amber": amber_count,
                "red": red_count,
                "green_pct": (green_count / total_items * 100) if total_items else 0,
                "amber_pct": (amber_count / total_items * 100) if total_items else 0,
                "red_pct": (red_count / total_items * 100) if total_items else 0,
            },
            "checklist": self.compliance_checklist,
            "cookie_audit": cookie_audit,
        }

    async def format_markdown_report(
        self, data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Format audit data as markdown"""
        summary = data["summary"]
        results = data["results"]
        checklist = data["checklist"]
        crawl_stats = data.get("crawl_stats", {})

        yield f"# 🔍 Website Compliance Audit\n\n"
        yield f"**Target:** {data['target_url']}\n"
        yield f"**Framework:** {data['framework']}\n"
        yield f"**Source:** [GOV.UK - What maintained schools must publish online]({self.valves.GOV_UK_GUIDANCE_URL})\n"

        # Show school type if detected
        school_type = data.get("school_type", "unknown")
        if school_type and school_type != "unknown":
            school_type_emoji = {
                "primary": "🎒",
                "secondary": "🎓",
                "all-through": "🏫",
            }.get(school_type, "🏫")
            school_type_display = school_type.replace("-", " ").title()
            yield f"**School Type:** {school_type_emoji} {school_type_display}\n"

        yield f"**Duration:** {data['elapsed']:.1f}s\n"
        yield f"**Pages:** {data['page_count']}\n"

        # Show cache stats if available
        if crawl_stats.get("used_cache"):
            yield f"**Cache:** ✅ Used (age: {crawl_stats.get('cache_age', 0):.1f}h)\n"
        elif crawl_stats:
            yield f"**Crawl Stats:** {crawl_stats.get('fetched', 0)} fetched, {crawl_stats.get('unchanged', 0)} unchanged\n"

        yield f"\n---\n\n"

        yield f"## Overall Status\n\n"
        yield f"- 🟢 **GREEN:** {summary['green']} ({summary['green_pct']:.0f}%)\n"
        yield f"- 🟡 **AMBER:** {summary['amber']} ({summary['amber_pct']:.0f}%)\n"
        yield f"- 🔴 **RED:** {summary['red']} ({summary['red_pct']:.0f}%)\n\n"

        # Add sitemap and site analysis sections if available
        sitemap_mermaid = data.get("sitemap_mermaid")
        seo_analysis = data.get("seo_analysis")
        link_analysis = data.get("link_analysis")
        change_tracking = data.get("change_tracking")

        if sitemap_mermaid or seo_analysis or link_analysis or change_tracking:
            yield f"---\n\n"
            yield f"## 🗺️ Site Map & Analysis\n\n"

            # Change tracking summary (if available)
            if change_tracking and change_tracking.get("has_previous"):
                yield f"### 📊 Changes Since Last Audit\n\n"
                yield f"**Previous audit:** {change_tracking.get('previous_date', 'Unknown')}\n\n"

                total_added = change_tracking.get("total_added", 0)
                total_removed = change_tracking.get("total_removed", 0)
                updated_count = change_tracking.get("updated_count", 0)

                if total_added == 0 and total_removed == 0 and updated_count == 0:
                    yield f"✅ **No changes detected** - site structure is stable\n\n"
                else:
                    if total_added > 0:
                        yield f"- ➕ **{total_added} new** pages/documents added\n"
                    if total_removed > 0:
                        yield f"- ➖ **{total_removed}** pages/documents removed\n"
                    if updated_count > 0:
                        yield f"- 🔄 **{updated_count}** documents updated (new versions)\n"
                    yield f"\n"

                    # Show details in collapsible sections
                    if change_tracking.get("added_pages") or change_tracking.get(
                        "added_docs"
                    ):
                        yield f"<details>\n<summary>View added items ({total_added} total)</summary>\n\n"
                        if change_tracking.get("added_pages"):
                            yield f"**New Pages:**\n"
                            for page in change_tracking["added_pages"][:10]:
                                yield f"- {page['title']}\n"
                            if len(change_tracking["added_pages"]) > 10:
                                yield f"- *...and {len(change_tracking['added_pages']) - 10} more*\n"
                            yield f"\n"
                        if change_tracking.get("added_docs"):
                            yield f"**New Documents:**\n"
                            for doc in change_tracking["added_docs"][:10]:
                                yield f"- {doc['filename']}\n"
                            if len(change_tracking["added_docs"]) > 10:
                                yield f"- *...and {len(change_tracking['added_docs']) - 10} more*\n"
                        yield f"</details>\n\n"

                    if change_tracking.get("removed_pages") or change_tracking.get(
                        "removed_docs"
                    ):
                        yield f"<details>\n<summary>⚠️ View removed items ({total_removed} total)</summary>\n\n"
                        if change_tracking.get("removed_pages"):
                            yield f"**Removed Pages:**\n"
                            for page in change_tracking["removed_pages"][:10]:
                                yield f"- {page['url']}\n"
                            if len(change_tracking["removed_pages"]) > 10:
                                yield f"- *...and {len(change_tracking['removed_pages']) - 10} more*\n"
                            yield f"\n"
                        if change_tracking.get("removed_docs"):
                            yield f"**Removed Documents:**\n"
                            for doc in change_tracking["removed_docs"][:10]:
                                yield f"- {doc['filename']}\n"
                            if len(change_tracking["removed_docs"]) > 10:
                                yield f"- *...and {len(change_tracking['removed_docs']) - 10} more*\n"
                        yield f"</details>\n\n"

            # SEO/Structure analysis
            if seo_analysis:
                yield f"### 🔍 SEO & Structure Analysis\n\n"

                issues_found = []
                if seo_analysis.get("orphaned_count", 0) > 0:
                    issues_found.append(
                        f"🔴 **{seo_analysis['orphaned_count']} orphaned pages** (no inbound links)"
                    )
                if seo_analysis.get("deep_pages_count", 0) > 0:
                    issues_found.append(
                        f"🟡 **{seo_analysis['deep_pages_count']} pages buried deep** (>3 levels)"
                    )
                if seo_analysis.get("duplicate_titles_count", 0) > 0:
                    issues_found.append(
                        f"🟡 **{seo_analysis['duplicate_titles_count']} duplicate page titles**"
                    )
                if seo_analysis.get("missing_titles_count", 0) > 0:
                    issues_found.append(
                        f"🔴 **{seo_analysis['missing_titles_count']} pages missing titles**"
                    )

                if not issues_found:
                    yield f"✅ **No structural issues detected** - good site organization\n\n"
                else:
                    for issue in issues_found:
                        yield f"- {issue}\n"
                    yield f"\n"

                    # Show details in collapsible sections
                    if seo_analysis.get("orphaned_pages"):
                        yield f"<details>\n<summary>View orphaned pages ({len(seo_analysis['orphaned_pages'])})</summary>\n\n"
                        for page in seo_analysis["orphaned_pages"][:15]:
                            yield f"- {page['title']} - `{page['url']}`\n"
                        if len(seo_analysis["orphaned_pages"]) > 15:
                            yield f"- *...and {len(seo_analysis['orphaned_pages']) - 15} more*\n"
                        yield f"\n*Recommendation: Add internal links to these pages for better navigation*\n"
                        yield f"</details>\n\n"

                    if seo_analysis.get("deep_pages"):
                        yield f"<details>\n<summary>View deeply nested pages ({len(seo_analysis['deep_pages'])})</summary>\n\n"
                        for page in seo_analysis["deep_pages"][:15]:
                            yield f"- {page['title']} (depth: {page['depth']})\n"
                        if len(seo_analysis["deep_pages"]) > 15:
                            yield f"- *...and {len(seo_analysis['deep_pages']) - 15} more*\n"
                        yield f"\n*Recommendation: Consider flattening navigation structure*\n"
                        yield f"</details>\n\n"

            # Broken links analysis
            if link_analysis and link_analysis.get("total_issues", 0) > 0:
                yield f"### 🔗 Link Health Check\n\n"

                broken_count = link_analysis.get("broken_count", 0)
                redirect_count = link_analysis.get("redirect_count", 0)
                missing_docs = link_analysis.get("missing_documents_count", 0)

                if broken_count > 0:
                    yield f"- 🔴 **{broken_count} broken links** (404 or errors)\n"
                if redirect_count > 0:
                    yield f"- 🟡 **{redirect_count} redirects** (consider updating links)\n"
                if missing_docs > 0:
                    yield f"- 🔴 **{missing_docs} missing documents**\n"
                yield f"\n"

                if link_analysis.get("broken_links"):
                    yield f"<details>\n<summary>View broken links ({broken_count})</summary>\n\n"
                    for link in link_analysis["broken_links"][:20]:
                        yield f"- [{link.get('title', 'Untitled')}]({link['url']}) - **HTTP {link['status']}**\n"
                    if len(link_analysis["broken_links"]) > 20:
                        yield f"- *...and {len(link_analysis['broken_links']) - 20} more*\n"
                    yield f"</details>\n\n"

            # Sitemap visualization
            if sitemap_mermaid:
                yield f"### 📍 Site Structure Diagram\n\n"
                yield f"<details>\n"
                yield f"<summary>Click to view interactive sitemap ({data['page_count']} pages)</summary>\n\n"
                yield f"```mermaid\n"
                yield f"{sitemap_mermaid}\n"
                yield f"```\n\n"
                yield f"*Full sitemap available in Knowledge Base for semantic search*\n"
                yield f"</details>\n\n"

        # Cookie Audit Section
        cookie_audit = data.get("cookie_audit")
        if cookie_audit and cookie_audit.get("total_cookies", 0) > 0:
            yield f"---\n\n"
            yield f"## 🍪 Cookie Compliance Audit\n\n"
            yield f"📖 [ICO Cookie Guidance]({self.valves.ICO_COOKIE_GUIDANCE_URL})\n\n"

            status_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(cookie_audit.get("status"), "⚪")
            yield f"**Status:** {status_emoji} {cookie_audit.get('reason', 'Unknown')}\n\n"

            yield f"### Cookie Summary\n\n"
            yield f"| Category | Count | Requires Consent |\n"
            yield f"|----------|-------|------------------|\n"
            yield f"| Essential | {cookie_audit.get('essential_count', 0)} | No |\n"
            yield f"| Analytics | {cookie_audit.get('analytics_count', 0)} | Yes |\n"
            yield f"| Marketing | {cookie_audit.get('marketing_count', 0)} | Yes |\n"
            yield f"| Functional | {cookie_audit.get('functional_count', 0)} | Yes |\n"
            yield f"| Unknown | {cookie_audit.get('unknown_count', 0)} | Assumed Yes |\n"
            yield f"| **Total** | **{cookie_audit.get('total_cookies', 0)}** | |\n\n"

            # Consent mechanism status
            if cookie_audit.get("consent_mechanism_found"):
                yield f"✅ **Cookie consent mechanism detected**\n"
            else:
                yield f"⚠️ **No cookie consent mechanism detected**\n"

            if cookie_audit.get("cookie_policy_found"):
                policy_url = cookie_audit.get("cookie_policy_url", "")
                if policy_url:
                    yield f"✅ **Cookie policy found:** [{policy_url}]({policy_url})\n"
                else:
                    yield f"✅ **Cookie policy page found**\n"
            else:
                yield f"⚠️ **No dedicated cookie policy page found**\n"

            yield f"\n"

            # Issues
            issues = cookie_audit.get("issues", [])
            if issues:
                yield f"### Issues Detected\n\n"
                for issue in issues:
                    yield f"- 🔴 {issue}\n"
                yield f"\n"

            # Recommendations
            recommendations = cookie_audit.get("recommendations", [])
            if recommendations:
                yield f"### Recommendations\n\n"
                for rec in recommendations:
                    yield f"- 💡 {rec}\n"
                yield f"\n"

            # Cookie details (collapsible)
            cookies_by_cat = cookie_audit.get("cookies_by_category", {})
            all_cookies = []
            for cat, cookies in cookies_by_cat.items():
                all_cookies.extend(cookies)

            if all_cookies:
                yield f"<details>\n<summary>View all cookies ({len(all_cookies)} total)</summary>\n\n"
                yield f"| Cookie Name | Category | Domain | Secure | HttpOnly |\n"
                yield f"|-------------|----------|--------|--------|----------|\n"
                for cookie in all_cookies[:30]:
                    name = cookie.get("name", "Unknown")[:30]
                    cat = cookie.get("category", "unknown")
                    domain = cookie.get("domain", "")[:20]
                    secure = "✅" if cookie.get("secure") else "❌"
                    httponly = "✅" if cookie.get("httponly") else "❌"
                    yield f"| {name} | {cat} | {domain} | {secure} | {httponly} |\n"
                if len(all_cookies) > 30:
                    yield f"\n*...and {len(all_cookies) - 30} more cookies*\n"
                yield f"</details>\n\n"

        elif cookie_audit:
            yield f"---\n\n"
            yield f"## 🍪 Cookie Compliance Audit\n\n"
            yield f"✅ **No cookies detected** - website does not appear to set cookies\n\n"

        yield f"---\n\n"

        for category_id, category in checklist.items():
            yield f"## {category['name']}\n\n"
            # Add GOV.UK guidance link if available
            if category.get("gov_uk_url"):
                yield f"📖 [View GOV.UK guidance for this section]({category['gov_uk_url']})\n\n"
            for item in category["items"]:
                item_id = item["id"]
                result = results.get(
                    item_id, {"status": "RED", "reason": "Not checked"}
                )
                status_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(
                    result["status"], "⚪"
                )

                yield f"{status_emoji} **{item['name']}**"
                if item.get("critical"):
                    yield f" *(Required)*"
                yield f"\n"

                # Show source references with URLs
                if result.get("sources"):
                    sources = result.get("sources", [])
                    for source in sources:
                        source_name = source.get("name", "Unknown")
                        source_page = source.get("page", "?")
                        source_url = source.get("url")
                        if source_url:
                            yield f"  - 📄 [{source_name} (Page {source_page})]({source_url})\n"
                        else:
                            yield f"  - 📄 {source_name} (Page {source_page})\n"
                elif result.get("url"):
                    # Fallback to single URL if sources not available
                    yield f"  - 📍 {result['url']}\n"

                if result.get("last_updated"):
                    yield f"  - 📅 Updated: {result['last_updated']}\n"
                if result.get("reason"):
                    yield f"  - ℹ️ {result['reason']}\n"
                if result.get("action"):
                    yield f"  - 🔔 **Action:** {result['action']}\n"
                yield f"\n"
            yield f"\n"