"""
title: Website Compliance Audit Pipeline
author: Open WebUI
version: 1.5.0
license: MIT
description: A comprehensive website audit agent for regulatory compliance checking with RAG (Red/Amber/Green) status reporting and LLM-based document assessment
requirements: aiohttp, beautifulsoup4, lxml, python-dateutil, pydantic, openai
"""

import asyncio
import json
import logging
import sys
import time
import uuid
import hashlib
import re
from io import BytesIO
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
        USE_LLM_FOR_VALIDATION: bool = Field(
            default=False,
            description="[DEPRECATED] Legacy setting - use ENABLE_LLM_ASSESSMENT instead",
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
        MAX_DOCUMENTS_TO_ANALYZE: int = Field(
            default=50,
            description="Maximum number of documents to upload and analyze per audit",
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
                "gov_uk_url": "https://www.gov.uk/guidance/what-maintained-schools-must-publish-online#statutory-information",
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
                "gov_uk_url": "https://www.gov.uk/guidance/what-maintained-schools-must-publish-online#optional-information",
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
                    "gov_uk_url": category.get("gov_uk_url"),  # Preserve GOV.UK link
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
        to_visit = [(url, 0)]  # (url, depth)
        pages = []
        page_cache = self.load_page_cache(cache_key)
        stats = {"cached": 0, "fetched": 0, "unchanged": 0}

        domain = urlparse(url).netloc

        async with aiohttp.ClientSession() as session:
            while to_visit and len(visited) < self.valves.MAX_PAGES:
                current_url, depth = to_visit.pop(0)

                if current_url in visited or depth > self.valves.CRAWL_DEPTH:
                    continue

                visited.add(current_url)

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
                                        to_visit.append((absolute_url, depth + 1))

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
                                        to_visit.append((absolute_url, depth + 1))

                        elif (
                            "application/pdf" in content_type
                            and self.valves.ENABLE_PDF_EXTRACTION
                        ):
                            page_data = {
                                "url": current_url,
                                "title": current_url.split("/")[-1],
                                "content": "",
                                "type": "pdf",
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

                # Yield progress
                yield {"pages": pages, "stats": stats}

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

                        # Enhanced validation with LLM
                        if self.valves.USE_LLM_FOR_VALIDATION and event_call:
                            log.info(f"[LLM] Validating {item_name} at {url}")
                            llm_result = await self.validate_document_with_llm(
                                document_text=doc_info.get("content", ""),
                                requirement_name=item_name,
                                event_call=event_call,
                            )

                            if llm_result.get("validated"):
                                # LLM provided additional insights
                                if llm_result.get("last_updated"):
                                    base_result["last_updated"] = llm_result[
                                        "last_updated"
                                    ]
                                    base_result["reason"] = (
                                        f"Document found. LLM extracted date: {llm_result['last_updated']}"
                                    )

                                if llm_result.get("meets_requirement") is False:
                                    base_result["status"] = "RED"
                                    base_result["reason"] = (
                                        f"Document found but may not meet requirements: {llm_result.get('summary', 'incomplete')}"
                                    )
                                elif llm_result.get("completeness", 0) > 80:
                                    base_result["status"] = "GREEN"
                                    base_result["reason"] = (
                                        f"Document found and validated by LLM (completeness: {llm_result['completeness']}%)"
                                    )

                                log.info(
                                    f"[LLM] Result: {base_result['status']} - {llm_result.get('summary', '')}"
                                )

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

    async def initialize_framework(
        self, event_call: Optional[Callable[[dict], Awaitable[dict]]] = None
    ) -> None:
        """Initialize compliance framework with multiple sources"""

        # Try loading from RAG first (highest priority - user customization)
        if self.valves.KNOWLEDGE_BASE_ID and event_call:
            log.info(
                f"Loading framework from knowledge base: {self.valves.KNOWLEDGE_BASE_ID}"
            )
            try:
                rag_checklist = await self.load_framework_from_rag(
                    self.valves.REGULATORY_FRAMEWORK, event_call
                )
                if rag_checklist:
                    self.compliance_checklist = rag_checklist
                    log.info(
                        f"Successfully loaded {len(rag_checklist)} categories from RAG"
                    )
                    return
            except Exception as e:
                log.warning(f"Failed to load from RAG: {e}, trying markdown framework")

        # Try loading from markdown framework file (official guidance)
        try:
            framework_file = Path(__file__).parent / "frameworks" / "uk_dfe_schools.md"
            if framework_file.exists():
                log.info(f"Loading framework from markdown: {framework_file}")
                markdown_checklist = self.load_framework_from_markdown(framework_file)
                if markdown_checklist:
                    self.compliance_checklist = markdown_checklist
                    log.info(
                        f"Successfully loaded {len(markdown_checklist)} categories from markdown"
                    )
                    return
        except Exception as e:
            log.warning(f"Failed to load from markdown: {e}, using hardcoded defaults")

        # Fallback to hardcoded defaults (last resort)
        self.compliance_checklist = self._default_checklist
        log.info(f"Using hardcoded {self.valves.REGULATORY_FRAMEWORK} framework")

    def load_framework_from_markdown(
        self, framework_file: Path
    ) -> Optional[Dict[str, Any]]:
        """Load compliance framework from markdown file"""

        try:
            with open(framework_file, "r", encoding="utf-8") as f:
                content = f.read()

            checklist = {}
            current_category = None
            current_item = {}

            for line in content.split("\n"):
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
        return documents[: self.valves.MAX_DOCUMENTS_TO_ANALYZE]

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

        for page in pages:
            # Skip document URLs (already handled separately)
            url = page["url"]
            if any(
                url.lower().endswith(ext)
                for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]
            ):
                continue

            # Note: crawler stores text in "content" field, not "text"
            text = page.get("content", "").lower()
            title = page.get("title", "").lower()

            if not text:
                continue  # Skip pages without text content

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
            ]
            for pattern in important_patterns:
                if pattern in url.lower():
                    score += 5
                    break

            if score > 0:
                scored_pages.append(
                    {
                        "url": url,
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

    async def upload_documents_to_kb(
        self,
        documents: List[Dict[str, str]],
        target_url: str,
        user_id: str,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """Upload documents to knowledge base with incremental updates"""

        try:
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
                doc["url"] for doc in documents[: self.valves.MAX_DOCUMENTS_TO_ANALYZE]
            }
            to_update = []  # Changed files (different hash)
            to_add = []  # New files
            to_keep = []  # Unchanged files
            to_delete = []  # Removed files

            # First pass: identify what needs to be done
            for doc in documents[: self.valves.MAX_DOCUMENTS_TO_ANALYZE]:
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

                                class MockUser:
                                    def __init__(self, user_id):
                                        self.id = user_id
                                        self.role = "user"

                                log.info(
                                    f"[KB] Adding existing file to KB collection: {filename}"
                                )
                                result = process_file(
                                    mock_request,
                                    ProcessFileForm(
                                        file_id=file_id, collection_name=knowledge.id
                                    ),
                                    user=MockUser(user_id),
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

                            # Create user object for process_file
                            class MockUser:
                                def __init__(self, user_id):
                                    self.id = user_id
                                    self.role = "user"

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
                                    user=MockUser(user_id),
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
                                    user=MockUser(user_id),
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
            if file_record and file_record.data:
                return file_record.data.get("source_url")
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
        for meta in metadatas[:3]:  # Top 3 sources
            # Look up URL from file record using file_id
            file_id = meta.get("file_id")
            url = self._get_file_url(file_id) if file_id else None
            # Fallback to metadata fields if lookup fails
            if not url:
                url = meta.get("url") or meta.get("source")

            source_ref = {
                "name": meta.get("name", "Unknown"),
                "page": meta.get("page_label", meta.get("page", "?")),
                "url": url,
            }
            sources.append(source_ref)

        # Extract first URL for backward compatibility
        first_url = None
        for meta in metadatas:
            file_id = meta.get("file_id")
            url = self._get_file_url(file_id) if file_id else None
            if not url:
                url = meta.get("url") or meta.get("source")
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
                if source_url and source_url not in source_urls:
                    source_urls.append(source_url)

                context_parts.append(
                    f"Document {idx}: {doc_name} (Page {page})\n{doc[:400]}"  # 400 chars per doc
                )

            context = "\n\n---\n\n".join(context_parts)

            # Build compliance assessment prompt
            school_type_display = (
                self.detected_school_type.replace("-", " ").title()
                if self.detected_school_type
                else "Unknown"
            )
            prompt = f"""You are a compliance auditor for UK maintained schools.

REGULATORY AUTHORITY: UK Department for Education (DfE)
SOURCE: https://www.gov.uk/guidance/what-maintained-schools-must-publish-online
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
                    temperature=0.1,  # Low for consistency
                    max_tokens=500,
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
                    log.error(f"[LLM] No content in response")
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
                    for meta in metadatas[:3]:  # Top 3 sources
                        # Look up URL from file record using file_id
                        file_id = meta.get("file_id")
                        url = self._get_file_url(file_id) if file_id else None
                        # Fallback to metadata fields if lookup fails
                        if not url:
                            url = meta.get("url") or meta.get("source")

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

    async def validate_document_with_llm(
        self,
        document_text: str,
        requirement_name: str,
        event_call: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Dict[str, Any]:
        """Deprecated: Use assess_compliance_with_llm instead"""
        return {"validated": False}

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
            async for update in self.crawl_website(target_url, cache_key):
                pages = update.get("pages", [])
                crawl_stats = update.get("stats", crawl_stats)
                if self.valves.DEBUG_MODE and len(pages) % 20 == 0:
                    log.info(
                        f"[CRAWL] Progress: {len(pages)} pages, Stats: {crawl_stats}"
                    )

            log.info(f"[CRAWL] Complete: {len(pages)} pages discovered")
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

            # Collect relevant webpages (top 50 by compliance relevance)
            webpages = self.collect_relevant_webpages(pages, max_pages=50)
            log.info(f"[WEBPAGES] Collected {len(webpages)} relevant webpages")

            # Combine documents and webpages for KB upload
            all_content = documents + webpages
            log.info(
                f"[KB] Total content to process: {len(documents)} documents + {len(webpages)} webpages = {len(all_content)} items"
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
                user_id = user.get("id") if user else "unknown"
                collection_id = await self.upload_documents_to_kb(
                    all_content, target_url, user_id, event_emitter
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

        return {
            "target_url": target_url,
            "framework": self.valves.REGULATORY_FRAMEWORK,
            "school_type": self.detected_school_type,
            "started": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": elapsed,
            "pages": pages,
            "page_count": len(pages),
            "crawl_stats": crawl_stats,
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
        yield f"**Source:** [GOV.UK - What maintained schools must publish online](https://www.gov.uk/guidance/what-maintained-schools-must-publish-online)\n"

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
