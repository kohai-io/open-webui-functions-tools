"""
title: Jira Agent for Open-WebUI
description: A comprehensive tool for interacting with Jira - search, view, create, and comment on issues with ease.
repository: https://github.com/taylorwilsdon/open-webui-tools
original author: @taylorwilsdon
original author_url: https://github.com/taylorwilsdon
version: 1.8.2
requirements: requests, pydantic, cryptography, fastapi
changelog:
  - 1.8.2: Fixed draft_release_notes - LLM no longer continues generating after tool completes (return completion message instead of None)
  - 1.8.1: Improved release notes descriptions - increased truncation limit to 500 chars and added "Description:" label for clarity
  - 1.8.0: Added draft_release_notes feature - generate formatted release notes from tickets with smart grouping (markdown/HTML output)
  - 1.7.6: Fixed ORDER BY support in JQL queries - now properly handles sorting (e.g., ORDER BY priority DESC)
  - 1.7.5: Added debug_mode toggle in settings - reduced default logging to WARNING level, moved HTML error responses to debug-only
  - 1.7.4: Fixed update_issue and create_issue to use Atlassian Document Format (ADF) for descriptions (required by Jira API v3)
  - 1.7.3: Removed automatic theme detection - now uses consistent light theme (OS dark mode was overriding app settings)
  - 1.7.2: Fixed theme detection - now uses CSS @media queries instead of JavaScript for reliable theme matching
  - 1.7.1: Fixed theme detection - now uses prefers-color-scheme and color-scheme meta tag for better compatibility
  - 1.7.0: Added theme awareness - automatically detects and matches Open WebUI's light/dark/OLED theme
  - 1.6.1: Extended Rich UI to search results - beautiful HTML table with color-coded badges
  - 1.6.0: Added Rich UI Element Embedding - issue details now display in beautiful Jira-styled HTML iframe
  - 1.5.5: Fixed HTML formatting - convert HTML to clean markdown text, use markdown separators
  - 1.5.4: Added description display to issue details view
  - 1.5.3: Changed Project column to show full project name instead of project key
  - 1.5.2: Added Project column to search results table
  - 1.5.1: Streamlined help message - removed redundant sections for cleaner output
  - 1.5.0: Added update_issue function - update description, summary, priority, and labels
  - 1.4.0: Changed default behavior - searches now exclude Done/Closed/Resolved unless explicitly requested
  - 1.3.1: Added smart status translation - "Open" automatically converts to "NOT IN (Done, Closed, Resolved)"
  - 1.3.0: Fixed API v3 /search/jql pagination - handles token-based pagination without total count
  - 1.2.2: Added prominent clickable links to issue details (header + "View in Jira" button)
  - 1.2.1: Enhanced total count extraction with fallback for API v3 response variations
  - 1.2.0: Added pagination support (50 results default), follow-up suggestions, and fixed total count display
  - 1.1.1: Fixed search endpoint - changed from /search to /search/jql for API v3 compatibility
  - 1.1.0: Fixed API v2 deprecation - now uses v3 for all auth methods (search endpoint migration)
  - 1.0.9: Added comprehensive help command to document all features
  - 1.0.8: Enhanced debugging for issue retrieval and response parsing
  - 1.0.7: Fixed PAT authentication to use Basic Auth (email:PAT) instead of Bearer token
  - 1.0.6: Fixed PAT authentication for Jira Cloud by using API v3 instead of 'latest'
  - 1.0.5: Enhanced authentication debugging, added detailed logging for 403/401 errors
  - 1.0.4: Added encryption for API credentials using WEBUI_SECRET_KEY
  - 1.0.3: Improved date formatting, enhanced HTML content handling, better comment display
  - 1.0.2: Extensive refactor - simplified logging, fixed duplicate messages, improved formatting
  - 1.0.1: Update with PAT support
  - 1.0.0: Initial release with comprehensive Jira integration capabilities
"""

import json
import logging
import os
import re
import html
import hashlib
import base64
from typing import Any, Awaitable, Callable, Dict, Optional, List, Union
from datetime import datetime
import requests
from pydantic import BaseModel, Field, validator, GetCoreSchemaHandler
from pydantic_core import core_schema
from cryptography.fernet import Fernet, InvalidToken
from fastapi.responses import HTMLResponse

# Get logger for this module
logger = logging.getLogger("jira_tool")

# Default to WARNING level - will be set to DEBUG if debug_mode is enabled
logger.setLevel(logging.WARNING)

# Warn if encryption key is not available
if not os.getenv("WEBUI_SECRET_KEY"):
    logger.warning(
        "WEBUI_SECRET_KEY environment variable not set. "
        "Credentials will be stored without encryption. "
        "Please set WEBUI_SECRET_KEY for secure credential storage."
    )


# Encryption implementation from nano_banana.py
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """
        Generate encryption key from WEBUI_SECRET_KEY if available
        Returns None if no key is configured
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None

        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """
        Encrypt a string value if a key is available
        Returns the original value if no key is available
        """
        if not value or value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No encryption if no key
            return value

        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """
        Decrypt an encrypted string value if a key is available
        Returns the original value if no key is available or decryption fails
        """
        if not value or not value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No decryption if no key
            return value[len("encrypted:") :]  # Return without prefix

        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    # Pydantic integration
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


class IssueFormatter:
    """Helper class to format Jira issues consistently as markdown tables"""

    @staticmethod
    def format_date(date_str: str) -> str:
        """Format a date string from Jira API to a more readable format"""
        if not date_str or date_str == "Unknown":
            return "Unknown"

        try:
            # Clean up the timezone part if it has an extra offset
            if "+00:00" in date_str and (
                "+" in date_str.split("+00:00")[1] or "-" in date_str.split("+00:00")[1]
            ):
                date_str = date_str.split("+00:00")[0] + date_str.split("+00:00")[1]

            # Parse ISO 8601 format
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

            # Format as "Mar 10, 2025 12:34 PM"
            formatted_date = dt.strftime("%b %d, %Y %I:%M %p")

            return formatted_date

        except (ValueError, TypeError) as e:
            # Log the error but don't crash
            logger.debug(f"Date parsing error: {e} for string: {date_str}")
            # If parsing fails, return a clean error message
            return "Invalid date"

    @staticmethod
    def format_issue_details(issue: Dict[str, Any]) -> str:
        """Format a single issue in Jira-style markdown"""
        # Define status and priority icons
        status_icon = (
            "✅"
            if issue["status"].lower() in ["done", "closed", "resolved"]
            else (
                "🔄"
                if issue["status"].lower() in ["in progress", "in review"]
                else "🆕"
            )
        )
        priority_icon = (
            "🔥"
            if issue["priority"].lower() in ["highest", "high"]
            else "⚡" if issue["priority"].lower() == "medium" else "🔽"
        )

        # Format metadata badges
        metadata_badges = (
            f"`{status_icon} {issue['status']}`  "
            f"`{priority_icon} {issue['priority']}`  "
            f"`📋 {issue['type']}`  "
            f"`🕒 {IssueFormatter.format_date(issue['created'])}`  "
            f"`🔄 {IssueFormatter.format_date(issue['updated'])}`  "
            f"`🙋 {issue['reporter']}`  "
            f"`🕵️‍♂️ {issue['assignee']}`  "
        )

        # Format the main content with clickable link
        output = (
            f"## [{issue['key']}]({issue['link']}) {issue['title']}\n\n"
            f"🔗 **[View in Jira]({issue['link']})**\n\n"
            f"{metadata_badges}\n\n"
        )

        # Add description if available
        if issue.get("description"):
            # Strip HTML tags from description for clean markdown display
            desc = issue["description"]
            # Remove HTML tags but preserve text and structure
            desc = re.sub(r"<br\s*/?>", "\n", desc)  # Convert line breaks
            desc = re.sub(r"</p>\s*<p>", "\n\n", desc)  # Paragraph breaks
            desc = re.sub(r"<li>", "• ", desc)  # List items
            desc = re.sub(r"<[^>]+>", "", desc)  # Remove all other tags
            desc = html.unescape(desc)  # Decode HTML entities
            desc = re.sub(r"\n\s*\n\s*\n+", "\n\n", desc)  # Clean multiple newlines
            desc = desc.strip()

            output += f"### 📝 Description\n\n{desc}\n\n"

        return output

    @staticmethod
    def format_issue_list(
        issues: List[Dict[str, Any]], total: Optional[int], displayed: int
    ) -> str:
        """Format a list of issues as a markdown table"""
        if not issues:
            return "No issues found."

        # Handle cases where total is unknown (API v3 pagination)
        if total is None:
            table = f"### Found {displayed}+ issues (showing {displayed})\n\n"
        else:
            table = f"### Found {total} issues (showing {displayed})\n\n"
        table += "| Key | Project | Summary | Status | Type | Priority | Updated |\n"
        table += "|-----|---------|---------|--------|------|----------|--------|\n"

        for issue in issues:
            table += (
                f"| [{issue['key']}]({issue['link']}) "
                f"| {issue['project']} "
                f"| {issue['summary']} "
                f"| {issue['status']} "
                f"| {issue['type']} "
                f"| {issue['priority']} "
                f"| {IssueFormatter.format_date(issue['updated'])} |\n"
            )

        return table

    @staticmethod
    def format_issue_details_html(
        issue: Dict[str, Any], comments: List[Dict[str, Any]] = None
    ) -> str:
        """Generate rich HTML for issue details with Jira-style formatting"""

        # Status color mapping
        status_colors = {
            "done": "#00875A",
            "closed": "#00875A",
            "resolved": "#00875A",
            "in progress": "#0052CC",
            "in review": "#0052CC",
            "backlog": "#6554C0",
            "to do": "#6554C0",
        }
        status_color = status_colors.get(issue["status"].lower(), "#5E6C84")

        # Priority color mapping
        priority_colors = {
            "highest": "#DE350B",
            "high": "#FF5630",
            "medium": "#FF991F",
            "low": "#36B37E",
            "lowest": "#00875A",
        }
        priority_color = priority_colors.get(issue["priority"].lower(), "#5E6C84")

        # Build comments HTML
        comments_html = ""
        if comments:
            comments_html = (
                '<h3 style="margin-top: 30px;">💬 Comments ('
                + str(len(comments))
                + ")</h3>"
            )
            for comment in comments:
                text = comment["text"]
                # Clean HTML from comment text
                if text.startswith("<"):
                    text = re.sub(r"<[^>]+>", "", text)
                    text = html.unescape(text).strip()

                comments_html += (
                    """
                <div class="comment">
                    <div class="comment-meta">
                        <strong>"""
                    + html.escape(comment["author"])
                    + """</strong> • 
                        """
                    + IssueFormatter.format_date(comment["created"])
                    + """
                    </div>
                    <div class="comment-text">"""
                    + html.escape(text)
                    + """</div>
                </div>
                """
                )

        # Clean description
        desc_html = issue.get("description", "<em>No description provided</em>")
        if desc_html:
            # Convert HTML to readable text
            desc_html = re.sub(r"<br\s*/?>", "\n", desc_html)
            desc_html = re.sub(r"</p>\s*<p>", "\n\n", desc_html)
            desc_html = re.sub(r"<li>", "• ", desc_html)
            desc_html = re.sub(r"<[^>]+>", "", desc_html)
            desc_html = html.unescape(desc_html).strip()
            # Convert newlines to <br> for HTML display
            desc_html = desc_html.replace("\n", "<br>")

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            /* Light theme only (for consistent display) */
            --bg-primary: #FFFFFF;
            --bg-secondary: #F4F5F7;
            --bg-tertiary: #FAFBFC;
            --text-primary: #172B4D;
            --text-secondary: #5E6C84;
            --border-color: #DFE1E6;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        .issue-header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        .issue-title {{
            font-size: 24px;
            font-weight: 500;
            color: var(--text-primary);
            margin: 10px 0;
        }}
        .issue-key {{
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
        }}
        .metadata {{
            margin: 20px 0;
            padding: 15px;
            background: var(--bg-secondary);
            border-radius: 5px;
        }}
        .metadata-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
            font-size: 13px;
            color: var(--text-secondary);
        }}
        .metadata-label {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        .description {{
            margin: 20px 0;
            padding: 15px;
            background: var(--bg-tertiary);
            border-left: 3px solid #0052CC;
            border-radius: 3px;
        }}
        h3 {{
            color: var(--text-primary);
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        a {{
            color: #0052CC;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .comment {{
            border-left: 3px solid var(--border-color);
            padding-left: 15px;
            margin: 15px 0;
        }}
        .comment-meta {{
            color: var(--text-secondary);
            font-size: 13px;
            margin-bottom: 5px;
        }}
        .comment-meta strong {{
            color: var(--text-primary);
        }}
        .comment-text {{
            color: var(--text-primary);
        }}
    </style>
</head>
<body>
    <div class="issue-header">
        <div class="issue-key">
            <a href="{issue['link']}" target="_blank">{issue['key']}</a>
        </div>
        <div class="issue-title">{html.escape(issue['title'])}</div>
        <div style="margin-top: 10px;">
            <span class="badge" style="background-color: {status_color}; color: white;">
                {html.escape(issue['status'])}
            </span>
            <span class="badge" style="background-color: {priority_color}; color: white;">
                {html.escape(issue['priority'])}
            </span>
            <span class="badge" style="background-color: #6554C0; color: white;">
                {html.escape(issue['type'])}
            </span>
        </div>
    </div>
    
    <div class="metadata">
        <div class="metadata-item">
            <span class="metadata-label">Project:</span> {html.escape(issue.get('project', 'Unknown'))}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Assignee:</span> {html.escape(issue['assignee'])}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Reporter:</span> {html.escape(issue['reporter'])}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Created:</span> {html.escape(issue['created'])}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Updated:</span> {html.escape(issue['updated'])}
        </div>
    </div>
    
    <div class="description">
        <h3>📝 Description</h3>
        <div>{desc_html}</div>
    </div>
    
    {comments_html}
    
    <div style="margin-top: 30px; padding-top: 15px; border-top: 1px solid var(--border-color); text-align: center;">
        <a href="{issue['link']}" target="_blank" style="font-weight: 600;">
            🔗 View in Jira
        </a>
    </div>
</body>
</html>
"""
        return html_content

    @staticmethod
    def format_issue_list_html(
        issues: List[Dict[str, Any]],
        total: Optional[int],
        displayed: int,
        start_at: int = 0,
    ) -> str:
        """Generate rich HTML table for search results with Jira-style formatting"""

        # Status color mapping
        status_colors = {
            "done": "#00875A",
            "closed": "#00875A",
            "resolved": "#00875A",
            "in progress": "#0052CC",
            "in review": "#0052CC",
            "backlog": "#6554C0",
            "to do": "#6554C0",
            "ready": "#0052CC",
        }

        # Priority color mapping
        priority_colors = {
            "highest": "#DE350B",
            "high": "#FF5630",
            "medium": "#FF991F",
            "low": "#36B37E",
            "lowest": "#00875A",
            "none": "#5E6C84",
        }

        # Build table rows
        rows_html = ""
        for issue in issues:
            status_color = status_colors.get(issue["status"].lower(), "#5E6C84")
            priority_color = priority_colors.get(issue["priority"].lower(), "#5E6C84")

            rows_html += f"""
            <tr>
                <td><a href="{issue['link']}" target="_blank" style="font-weight: 600;">{html.escape(issue['key'])}</a></td>
                <td class="secondary-text">{html.escape(issue['project'])}</td>
                <td>{html.escape(issue['summary'])}</td>
                <td>
                    <span class="badge" style="background-color: {status_color}; color: white;">
                        {html.escape(issue['status'])}
                    </span>
                </td>
                <td class="secondary-text small-text">{html.escape(issue['type'])}</td>
                <td>
                    <span class="badge" style="background-color: {priority_color}; color: white; font-size: 11px;">
                        {html.escape(issue['priority'])}
                    </span>
                </td>
                <td class="secondary-text small-text">{html.escape(IssueFormatter.format_date(issue['updated']))}</td>
            </tr>
            """

        # Build pagination info
        current_end = start_at + displayed
        pagination_html = ""
        if total is None:
            pagination_html = (
                f"Showing results {start_at + 1}-{current_end}. More may be available."
            )
        elif total > displayed:
            pagination_html = (
                f"Showing results {start_at + 1}-{current_end} of {total} total issues."
            )
        else:
            pagination_html = f"Found {total} issues."

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            /* Light theme only (for consistent display) */
            --bg-primary: #FFFFFF;
            --bg-secondary: #F4F5F7;
            --bg-hover: #F4F5F7;
            --text-primary: #172B4D;
            --text-secondary: #5E6C84;
            --border-color: #DFE1E6;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
        }}
        .header {{
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}
        .header h2 {{
            margin: 0 0 10px 0;
            color: var(--text-primary);
            font-size: 20px;
            font-weight: 500;
        }}
        .pagination-info {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-primary);
        }}
        th {{
            background: var(--bg-secondary);
            padding: 10px 8px;
            text-align: left;
            font-weight: 600;
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            border-bottom: 2px solid var(--border-color);
        }}
        td {{
            padding: 12px 8px;
            border-bottom: 1px solid var(--border-color);
            font-size: 14px;
            color: var(--text-primary);
        }}
        tr:hover {{
            background: var(--bg-hover);
        }}
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
        }}
        .secondary-text {{
            color: var(--text-secondary);
        }}
        .small-text {{
            font-size: 12px;
        }}
        a {{
            color: #0052CC;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🔍 Search Results</h2>
        <div class="pagination-info">{pagination_html}</div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th style="width: 110px;">Key</th>
                <th style="width: 140px;">Project</th>
                <th>Summary</th>
                <th style="width: 120px;">Status</th>
                <th style="width: 80px;">Type</th>
                <th style="width: 90px;">Priority</th>
                <th style="width: 100px;">Updated</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>
"""
        return html_content

    @staticmethod
    def format_comments(issue_id: str, comments: List[Dict[str, Any]]) -> str:
        """Format issue comments in Jira-style markdown"""
        if not comments:
            return ""

        comment_text = f"### 💬 Comments ({len(comments)})\n\n"
        for comment in comments:
            # Handle HTML content in comments
            text = comment["text"]
            if text.startswith("<") and ">" in text:
                # Strip HTML tags for clean markdown display
                # Remove HTML tags but preserve text
                text = re.sub(r"<[^>]+>", "", text)
                # Decode HTML entities
                text = html.unescape(text)
                # Clean up excessive whitespace
                text = re.sub(r"\n\s*\n", "\n\n", text)
                text = text.strip()

            # Format each comment in a more visually appealing style
            comment_text += (
                f"**{comment['author']}** • {IssueFormatter.format_date(comment['created'])}\n\n"
                f"{text}\n\n"
                "---\n\n"
            )
        return comment_text


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Awaitable[None]]):
        self.event_emitter = event_emitter
        self.logger = logging.getLogger("jira_tool.emitter")

    async def emit_status(
        self, description: str, done: bool, error: bool = False
    ) -> None:
        """Emit a status event with a description and completion status."""
        if error and not done:
            raise ValueError("Error status must also be marked as done")

        icon = "✅" if done and not error else "🚫 " if error else "💬"

        try:
            await self.event_emitter(
                {
                    "data": {
                        "description": f"{icon} {description}",
                        "status": "complete" if done else "in_progress",
                        "done": done,
                    },
                    "type": "status",
                }
            )
        except Exception as e:
            logger.error(f"Failed to emit status event: {str(e)}")
            raise RuntimeError(f"Failed to emit status event: {str(e)}") from e

    async def emit_message(self, content: str) -> None:
        """Emit a simple message event."""
        if not content:
            raise ValueError("Message content cannot be empty")

        try:
            await self.event_emitter({"data": {"content": content}, "type": "message"})
        except Exception as e:
            logger.error(f"Failed to emit message event: {str(e)}")
            raise RuntimeError(f"Failed to emit message event: {str(e)}") from e

    async def emit_source(
        self, name: str, url: str, content: str = "", html: bool = False
    ) -> None:
        """Emit a citation source event."""
        if not name or not url:
            raise ValueError("Source name and URL are required")

        try:
            await self.event_emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [content] if content else [],
                        "metadata": [{"source": url, "html": html}],
                        "source": {"name": name},
                    },
                }
            )
        except Exception as e:
            logger.error(f"Failed to emit source event: {str(e)}")
            raise RuntimeError(f"Failed to emit source event: {str(e)}") from e

    async def emit_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: Optional[str] = "Results",
    ) -> None:
        """Emit a formatted markdown table of data."""
        if not headers:
            raise ValueError("Table must have at least one header")

        if any(len(row) != len(headers) for row in rows):
            raise ValueError("All rows must have the same number of columns as headers")

        # Create markdown table
        table = (
            f"### {title}\n\n|"
            + "|".join(headers)
            + "|\n|"
            + "|".join(["---"] * len(headers))
            + "|\n"
        )

        for row in rows:
            # Convert all cells to strings and escape pipe characters
            formatted_row = [str(cell).replace("|", "\\|") for cell in row]
            table += "|" + "|".join(formatted_row) + "|\n"

        await self.emit_message(table)


class JiraApiError(Exception):
    """Exception raised for Jira API errors"""

    pass


class Jira:
    def __init__(self, username: str, password: str, base_url: str, pat: str = ""):
        self.logger = logging.getLogger("jira_tool.api")
        self.base_url = base_url.rstrip("/")
        self.username = username
        # Store credentials in a closure to avoid keeping them in instance variables
        self._get_auth = self._create_auth_function(username, password, pat)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        # Jira Cloud PATs use Basic Auth (email:PAT), not Bearer token
        # Despite being called "Personal Access Token", they're used with Basic Auth
        if pat:
            self.logger.debug(f"Using PAT with Basic Auth (token length: {len(pat)})")
        else:
            self.logger.debug(f"Using Basic Auth with username: {username}")

        # Always use API v3 for Jira Cloud (v2 search endpoint deprecated as of 2025)
        # See: https://developer.atlassian.com/changelog/#CHANGE-2046
        self.api_version = "3"
        self.logger.debug(
            f"Initialized Jira client for {self.base_url} (API version: {self.api_version})"
        )

    def _create_auth_function(self, username: str, password: str, pat: str):
        """Create auth function with credentials stored in closure"""

        def get_auth():
            # For PAT: use email as username, PAT as password in Basic Auth
            if pat:
                return (username, pat)
            return (username, password)

        return get_auth

    def _handle_response(self, response: requests.Response, operation: str):
        """Handle API response and raise appropriate exceptions"""
        if response.status_code >= 200 and response.status_code < 300:
            if not response.content:
                return {}

            try:
                return response.json()
            except json.JSONDecodeError as e:
                raise JiraApiError(f"Invalid JSON response: {str(e)}") from e

        # Create appropriate error message based on status code
        if response.status_code == 401:
            error_msg = "Authentication failed. Please check your username and API key."
            self.logger.debug(f"401 Error - Response headers: {dict(response.headers)}")
        elif response.status_code == 403:
            error_msg = "You don't have permission to perform this operation."
            self.logger.debug(f"403 Error - Response headers: {dict(response.headers)}")
            self.logger.debug(f"403 Error - Response body: {response.text[:500]}")
        elif response.status_code == 404:
            error_msg = f"Resource not found while attempting to {operation}."
        elif response.status_code == 400:
            try:
                error_details = response.json()
                error_msg = f"Bad request: {error_details.get('errorMessages', ['Unknown error'])[0]}"
            except:
                # Extract just the error message, not the full HTML response
                error_msg = f"Bad request (check debug mode for details)"
        else:
            error_msg = f"Jira API error ({response.status_code})"
            self.logger.debug(f"Error response: {response.text[:500]}")

        raise JiraApiError(error_msg)

    def get(self, endpoint: str, params: Dict[str, Any] = None):
        url = f"{self.base_url}/rest/api/{self.api_version}/{endpoint}"
        self.logger.debug(f"GET request to {url}")
        self.logger.debug(f"Request params: {params}")

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                auth=self._get_auth(),
                timeout=30,
            )
            self.logger.debug(f"Response status: {response.status_code}")
            return self._handle_response(response, f"get {endpoint}")
        except requests.RequestException as e:
            self.logger.error(
                f"Request failed for GET {endpoint}: {str(e)}", exc_info=True
            )
            raise JiraApiError(f"Request failed: {str(e)}") from e

    def post(self, endpoint: str, data: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/{self.api_version}/{endpoint}"
        self.logger.debug(f"POST request to {url}")
        self.logger.debug(f"Request data: {json.dumps(data)[:1000]}")

        try:
            response = requests.post(
                url, json=data, headers=self.headers, auth=self._get_auth(), timeout=30
            )
            return self._handle_response(response, f"post to {endpoint}")
        except requests.RequestException as e:
            self.logger.error(
                f"Request failed for POST {endpoint}: {str(e)}", exc_info=True
            )
            raise JiraApiError(f"Request failed: {str(e)}") from e

    def put(self, endpoint: str, data: Dict[str, Any]):
        url = f"{self.base_url}/rest/api/{self.api_version}/{endpoint}"
        self.logger.debug(f"PUT request to {url}")
        self.logger.debug(f"Request data: {json.dumps(data)[:1000]}")

        try:
            response = requests.put(
                url, json=data, headers=self.headers, auth=self._get_auth(), timeout=30
            )
            return self._handle_response(response, f"update {endpoint}")
        except requests.RequestException as e:
            self.logger.error(
                f"Request failed for PUT {endpoint}: {str(e)}", exc_info=True
            )
            raise JiraApiError(f"Request failed: {str(e)}") from e

    def get_issue(
        self,
        issue_id: str,
        fields: str = "summary,description,status,assignee,reporter,created,updated,priority,issuetype,project",
    ):
        """Get detailed information about a specific Jira issue"""
        self.logger.debug(f"Getting issue details for {issue_id}")
        endpoint = f"issue/{issue_id}"

        try:
            result = self.get(
                endpoint, {"fields": fields, "expand": "renderedFields,names"}
            )

            if result is None:
                raise JiraApiError(f"Empty response received for issue {issue_id}")

            # Check if fields exists in result
            if "fields" not in result:
                self.logger.error(f"Missing 'fields' in response for {issue_id}")
                raise JiraApiError(
                    f"Invalid response structure: missing 'fields' for issue {issue_id}"
                )

            # Create a structured issue data object with proper fallbacks
            issue_data = {
                "key": issue_id,
                "title": result["fields"].get("summary", "No summary"),
                "status": (result["fields"].get("status", {}) or {}).get(
                    "name", "Unknown"
                ),
                "type": (result["fields"].get("issuetype", {}) or {}).get(
                    "name", "Unknown"
                ),
                "project": (result["fields"].get("project", {}) or {}).get(
                    "name", "Unknown"
                ),
                "priority": (result["fields"].get("priority", {}) or {}).get(
                    "name", "Not set"
                ),
                "created": IssueFormatter.format_date(
                    result["fields"].get("created", "Unknown")
                ),
                "updated": IssueFormatter.format_date(
                    result["fields"].get("updated", "Unknown")
                ),
                "reporter": (result["fields"].get("reporter", {}) or {}).get(
                    "displayName", self.username
                ),
                "assignee": (result["fields"].get("assignee", {}) or {}).get(
                    "displayName", "Unassigned"
                ),
                "link": f"{self.base_url}/browse/{issue_id}",
            }

            # Handle description with better error checking
            description_html = None

            # Try to get rendered description
            if result.get("renderedFields") and result["renderedFields"].get(
                "description"
            ):
                description_html = result["renderedFields"]["description"]
            # If no rendered description, try raw description
            elif result["fields"].get("description"):
                description_html = f"<p>{result['fields']['description']}</p>"
            else:
                description_html = "<p><em>No description provided</em></p>"

            issue_data["description"] = description_html

            self.logger.debug(f"Successfully retrieved issue {issue_id}")
            return issue_data

        except KeyError as e:
            self.logger.error(
                f"Missing field in issue response: {str(e)}", exc_info=True
            )
            if result:
                self.logger.debug(f"Response structure: {json.dumps(result)[:500]}")
            raise JiraApiError(f"Invalid response structure: missing {str(e)}") from e

    def search(self, query: str, max_results: int = 50, start_at: int = 0):
        """Search for Jira issues using JQL or free text"""
        self.logger.debug(
            f"Searching issues with query: {query} (max={max_results}, start={start_at})"
        )
        # API v3 requires 'search/jql' instead of 'search'
        endpoint = "search/jql"

        # Determine if the query is already JQL or needs conversion
        if any(
            operator in query
            for operator in ["=", "~", ">", "<", " AND ", " OR ", " ORDER BY "]
        ):
            jql = query
        else:
            # Convert free text to JQL
            terms = query.split()
            if terms:
                cql_terms = " OR ".join([f'text ~ "{term}"' for term in terms])
            else:
                cql_terms = f'text ~ "{query}"'
            jql = cql_terms
            self.logger.debug(f"Converted free text to JQL: {jql}")

        # Smart status translation: convert "Open" to "not done/closed/resolved"
        # This handles variations like "status = Open", "status = 'Open'", etc.
        open_pattern = r"status\s*=\s*['\"]?Open['\"]?"
        if re.search(open_pattern, jql, re.IGNORECASE):
            # Replace "status = Open" with proper NOT IN clause
            jql = re.sub(
                open_pattern,
                "status NOT IN ('Done', 'Closed', 'Resolved')",
                jql,
                flags=re.IGNORECASE,
            )
            self.logger.debug(f"Translated 'Open' status to: {jql}")

        # Default behavior: exclude completed tickets unless explicitly requested
        # Check if user explicitly mentions Done, Closed, or Resolved statuses
        completed_statuses = ["Done", "Closed", "Resolved"]
        explicitly_includes_completed = any(
            re.search(rf"\b{status}\b", jql, re.IGNORECASE)
            for status in completed_statuses
        )

        # Only add default filter if:
        # 1. User hasn't explicitly mentioned completed statuses
        # 2. Query doesn't already have a status filter that would conflict
        if not explicitly_includes_completed and "status NOT IN" not in jql:
            # Extract ORDER BY clause if present (must come at the end)
            order_by_match = re.search(r"\s+ORDER\s+BY\s+.+$", jql, re.IGNORECASE)
            order_by_clause = ""
            if order_by_match:
                order_by_clause = order_by_match.group(0)
                jql = jql[: order_by_match.start()]

            # Add default exclusion filter
            if jql.strip():
                jql = f"({jql}) AND status NOT IN ('Done', 'Closed', 'Resolved')"
            else:
                jql = "status NOT IN ('Done', 'Closed', 'Resolved')"

            # Re-append ORDER BY clause at the end
            if order_by_clause:
                jql = jql + order_by_clause

            self.logger.debug(f"Applied default filter to exclude completed issues")

        params = {
            "jql": jql,
            "maxResults": max_results,
            "startAt": start_at,
            "fields": "summary,status,issuetype,priority,updated,project",
        }
        raw_response = self.get(endpoint, params)

        issues = []
        for item in raw_response.get("issues", []):
            try:
                issues.append(
                    {
                        "key": item["key"],
                        "project": item["fields"]
                        .get("project", {})
                        .get("name", "Unknown"),
                        "summary": item["fields"].get("summary", "No summary"),
                        "status": item["fields"]
                        .get("status", {})
                        .get("name", "Unknown"),
                        "type": item["fields"]
                        .get("issuetype", {})
                        .get("name", "Unknown"),
                        "priority": item["fields"]
                        .get("priority", {})
                        .get("name", "Not set"),
                        "updated": item["fields"].get("updated", "Unknown"),
                        "link": f"{self.base_url}/browse/{item['key']}",
                    }
                )
            except KeyError as e:
                self.logger.warning(f"Missing field in search result item: {e}")
                # Continue processing other results rather than failing completely

        # API v3 /search/jql uses token-based pagination and doesn't return total count
        # It returns 'isLast' boolean and optional 'nextPageToken'
        # We need to estimate based on the response
        is_last = raw_response.get("isLast", True)
        has_next_token = "nextPageToken" in raw_response

        # Try to extract total count from various possible fields
        total = raw_response.get("total")

        if total is None:
            # API v3 /search/jql doesn't provide total, estimate based on pagination
            if len(issues) == 0:
                total = 0
            elif is_last and start_at == 0:
                # First and last page, total equals displayed
                total = len(issues)
            elif is_last:
                # Last page of multi-page results
                total = start_at + len(issues)
            else:
                # More pages available, show "X+" format
                total = None  # Will indicate "more available"

        self.logger.debug(
            f"Extracted total: {total}, displayed: {len(issues)}, isLast: {is_last}"
        )

        return {
            "issues": issues,
            "total": total,
            "displayed": len(issues),
            "start_at": start_at,
            "max_results": max_results,
        }

    def get_projects(self):
        """Get a list of available projects"""
        self.logger.debug("Getting list of projects")
        endpoint = "project"
        result = self.get(endpoint)

        projects = []
        for item in result:
            try:
                projects.append(
                    {"key": item["key"], "name": item["name"], "id": item["id"]}
                )
            except KeyError as e:
                self.logger.warning(f"Missing field in project: {e}")

        self.logger.debug(f"Retrieved {len(projects)} projects")
        return projects

    def get_issue_types(self, project_key: str = None):
        """Get available issue types, optionally filtered by project"""
        self.logger.debug(
            f"Getting issue types{' for project ' + project_key if project_key else ''}"
        )

        try:
            if project_key:
                endpoint = f"project/{project_key}"
                result = self.get(endpoint)
                issue_types = result.get("issueTypes", [])
            else:
                endpoint = "issuetype"
                issue_types = self.get(endpoint)

            return [{"id": it["id"], "name": it["name"]} for it in issue_types]
        except Exception as e:
            self.logger.error(f"Error getting issue types: {str(e)}", exc_info=True)
            raise JiraApiError(f"Failed to retrieve issue types: {str(e)}") from e

    def get_priorities(self):
        """Get available priorities"""
        self.logger.debug("Getting list of priorities")
        endpoint = "priority"
        priorities = self.get(endpoint)
        return [{"id": p["id"], "name": p["name"]} for p in priorities]

    def create_issue(
        self,
        project_key: str,
        summary: str,
        description: str,
        issue_type: str,
        priority: str = None,
    ):
        """Create a new Jira issue"""
        self.logger.debug(f"Creating new issue in project {project_key}")
        endpoint = "issue"
        default_issue_type = "Task"
        if not issue_type:
            issue_type = default_issue_type
            self.logger.debug(
                f"No issue type provided, using default: {default_issue_type}"
            )

        # Build the issue fields with ADF format for description
        issue_data = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": self._text_to_adf(description),
                "issuetype": {"name": issue_type},
            }
        }

        # Add priority if specified
        if priority:
            issue_data["fields"]["priority"] = {"name": priority}

        self.logger.debug(f"Creating issue with ADF description format")

        try:
            result = self.post(endpoint, issue_data)
        except JiraApiError as e:
            # If ADF format fails, try plain text (for Jira Data Center)
            if "400" in str(e):
                self.logger.debug(
                    "ADF format failed, retrying with plain text description"
                )
                issue_data["fields"]["description"] = description
                result = self.post(endpoint, issue_data)
            else:
                raise

        return {
            "key": result["key"],
            "id": result["id"],
            "link": f"{self.base_url}/browse/{result['key']}",
        }

    def add_comment(self, issue_id: str, comment: str):
        """Add a comment to an existing issue"""
        self.logger.debug(f"Adding comment to issue {issue_id}")
        endpoint = f"issue/{issue_id}/comment"

        # For Jira Data Center, try the simpler format first
        try:
            # Simple format for Jira Data Center
            comment_data = {"body": comment}
            self.logger.debug("Attempting comment with legacy format")
            result = self.post(endpoint, comment_data)
            return {
                "id": result["id"],
                "created": result["created"],
                "issue_link": f"{self.base_url}/browse/{issue_id}",
            }
        except JiraApiError as e:
            # If simple format fails, try ADF format for Jira Cloud
            if "400" in str(e):
                self.logger.debug(
                    "Legacy format failed, trying Atlassian Document Format"
                )
                comment_data = {
                    "body": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": comment}],
                            }
                        ],
                    }
                }
                result = self.post(endpoint, comment_data)
                return {
                    "id": result["id"],
                    "created": result["created"],
                    "issue_link": f"{self.base_url}/browse/{issue_id}",
                }
            else:
                raise

    def get_comments(self, issue_id: str):
        """Get comments for an issue"""
        self.logger.debug(f"Getting comments for issue {issue_id}")
        endpoint = f"issue/{issue_id}/comment"

        try:
            result = self.get(endpoint)
            self.logger.debug(f"Retrieved {len(result.get('comments', []))} comments")

            comments = []
            for comment in result.get("comments", []):
                # Handle different comment formats
                text = ""

                # Try to extract from ADF format
                if (
                    "body" in comment
                    and isinstance(comment["body"], dict)
                    and "content" in comment["body"]
                ):
                    try:
                        for content in comment["body"]["content"]:
                            if "content" in content:
                                for text_content in content["content"]:
                                    if "text" in text_content:
                                        text += text_content["text"]
                    except (KeyError, TypeError) as e:
                        self.logger.warning(f"Error parsing ADF comment: {e}")

                # Try legacy format if ADF extraction yields nothing
                if not text and isinstance(comment.get("body"), str):
                    text = comment["body"]

                # If still no text, use a placeholder
                if not text:
                    text = "[Comment format not supported]"

                comments.append(
                    {
                        "id": comment["id"],
                        "author": comment.get("author", {}).get(
                            "displayName", "Unknown"
                        ),
                        "created": comment.get("created", "Unknown"),
                        "updated": comment.get("updated", "Unknown"),
                        "text": text,
                    }
                )

            return comments
        except Exception as e:
            self.logger.error(f"Error getting comments: {str(e)}", exc_info=True)
            raise JiraApiError(f"Failed to retrieve comments: {str(e)}") from e

    def assign_issue(self, issue_id: str, assignee: str):
        """Assign an issue to a user"""
        self.logger.debug(f"Assigning issue {issue_id} to {assignee or 'Unassigned'}")
        endpoint = f"issue/{issue_id}/assignee"

        # Handle special case for unassigning
        if not assignee or assignee.lower() in ["unassigned", "none"]:
            data = {"assignee": None}
        else:
            data = {"assignee": {"name": assignee}}

        self.put(endpoint, data)

        return {
            "issue_key": issue_id,
            "assignee": assignee or "Unassigned",
            "link": f"{self.base_url}/browse/{issue_id}",
        }

    def _text_to_adf(self, text: str) -> dict:
        """
        Convert plain text to Atlassian Document Format (ADF)

        Args:
            text: Plain text string

        Returns:
            ADF formatted dictionary
        """
        return {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        }

    def update_issue(
        self,
        issue_id: str,
        summary: str = None,
        description: str = None,
        priority: str = None,
        labels: List[str] = None,
    ):
        """
        Update issue fields (description, summary, priority, labels)

        Args:
            issue_id: Issue key (e.g., "PROJ-123")
            summary: New issue title/summary
            description: New issue description
            priority: New priority (e.g., "High", "Medium", "Low")
            labels: List of labels to set (replaces existing labels)

        Returns:
            Dictionary with updated fields and issue link
        """
        self.logger.debug(f"Updating issue {issue_id}")
        endpoint = f"issue/{issue_id}"

        # Build the update payload
        fields = {}
        updated_fields = []

        if summary is not None:
            fields["summary"] = summary
            updated_fields.append("summary")
            self.logger.debug(f"Updating summary to: {summary[:50]}...")

        if description is not None:
            # Try ADF format first (for Jira Cloud/API v3)
            fields["description"] = self._text_to_adf(description)
            updated_fields.append("description")
            self.logger.debug(
                f"Updating description ({len(description)} chars) using ADF format"
            )

        if priority is not None:
            fields["priority"] = {"name": priority}
            updated_fields.append("priority")
            self.logger.debug(f"Updating priority to: {priority}")

        if labels is not None:
            fields["labels"] = labels
            updated_fields.append("labels")
            self.logger.debug(f"Updating labels to: {labels}")

        if not fields:
            raise ValueError("At least one field must be provided for update")

        # Make the API call
        data = {"fields": fields}
        try:
            self.put(endpoint, data)
        except JiraApiError as e:
            # If ADF format fails, try plain text (for Jira Data Center)
            if "400" in str(e) and description is not None:
                self.logger.debug(
                    "ADF format failed, retrying with plain text description"
                )
                fields["description"] = description
                data = {"fields": fields}
                self.put(endpoint, data)
            else:
                raise

        self.logger.debug(
            f"Successfully updated {', '.join(updated_fields)} for {issue_id}"
        )

        return {
            "issue_key": issue_id,
            "updated_fields": updated_fields,
            "link": f"{self.base_url}/browse/{issue_id}",
        }

    def update_issue_status(
        self, issue_id: str, transition_id=None, transition_name=None
    ):
        """
        Update the status of an issue using either transition ID or name
        """
        self.logger.debug(
            f"Updating status of issue {issue_id} using {'ID' if transition_id else 'name'} {transition_id or transition_name}"
        )

        if not (transition_id or transition_name):
            raise ValueError("Either transition_id or transition_name must be provided")

        # First, get available transitions
        transitions_endpoint = f"issue/{issue_id}/transitions"
        transitions = self.get(transitions_endpoint)
        self.logger.debug(
            f"Available transitions: {', '.join([t['name'] for t in transitions.get('transitions', [])])}"
        )

        transition_to_use = None

        # Find the transition by ID or name
        if transition_id:
            for t in transitions.get("transitions", []):
                if t["id"] == transition_id:
                    transition_to_use = t["id"]
                    break
        elif transition_name:
            for t in transitions.get("transitions", []):
                if t["name"].lower() == transition_name.lower():
                    transition_to_use = t["id"]
                    break

        if not transition_to_use:
            available_transitions = ", ".join(
                [
                    f"{t['name']} (ID: {t['id']})"
                    for t in transitions.get("transitions", [])
                ]
            )
            self.logger.error(
                f"Transition {transition_id or transition_name} not found. Available: {available_transitions}"
            )
            raise JiraApiError(
                f"Transition not found. Available transitions: {available_transitions}"
            )

        # Perform the transition
        transition_data = {"transition": {"id": transition_to_use}}
        self.post(f"issue/{issue_id}/transitions", transition_data)

        # Get updated issue to confirm new status
        updated_issue = self.get_issue(issue_id, "status")

        return {
            "issue_key": issue_id,
            "new_status": updated_issue["status"],
            "link": f"{self.base_url}/browse/{issue_id}",
        }

    def get_available_transitions(self, issue_id: str):
        """Get available status transitions for an issue"""
        self.logger.debug(f"Getting available transitions for issue {issue_id}")
        transitions_endpoint = f"issue/{issue_id}/transitions"
        transitions = self.get(transitions_endpoint)

        return [
            {"id": t["id"], "name": t["name"], "to_status": t["to"]["name"]}
            for t in transitions.get("transitions", [])
        ]


class Tools:
    def __init__(self):
        self.logger = logging.getLogger("jira_tool.tools")
        self.valves = self.Valves()

    class Valves(BaseModel):
        username: str = Field(
            "", description="Your Jira email address (required for authentication)"
        )
        password: EncryptedStr = Field(
            default="",
            description="Your Jira API Token (generate at id.atlassian.com/manage-profile/security/api-tokens) or leave empty to use PAT field",
        )
        pat: EncryptedStr = Field(
            default="",
            description="Alternative: Personal Access Token or API Token (leave password empty if using this field)",
        )
        base_url: str = Field(
            "",
            description="Your Jira base URL (e.g., https://your-company.atlassian.net)",
        )
        debug_mode: bool = Field(
            default=False,
            description="Enable detailed debug logging for troubleshooting API calls and responses",
        )

        @validator("base_url")
        def validate_url(cls, v):
            if not v:
                return v
            if not v.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            return v

        @validator("pat")
        def validate_credentials(cls, v, values):
            if not v and (not values.get("username") or not values.get("password")):
                raise ValueError("Either PAT or username/password must be provided")
            return v

    def _get_jira_client(self):
        """Initialize and return a Jira client using valve values"""
        # Set logging level based on debug_mode
        if self.valves.debug_mode:
            logger.setLevel(logging.DEBUG)
            # Also configure root logger for debug mode
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,
            )
        else:
            logger.setLevel(logging.WARNING)

        if not self.valves.base_url:
            raise ValueError(
                "Jira base URL not configured. Please provide your Jira base URL."
            )

        # Decrypt credentials
        decrypted_password = EncryptedStr.decrypt(self.valves.password)
        decrypted_pat = EncryptedStr.decrypt(self.valves.pat)

        if self.valves.debug_mode:
            # Debug logging (without exposing actual credentials)
            self.logger.debug(
                f"PAT provided: {bool(decrypted_pat)}, PAT length: {len(decrypted_pat) if decrypted_pat else 0}"
            )
            self.logger.debug(
                f"Password provided: {bool(decrypted_password)}, Password length: {len(decrypted_password) if decrypted_password else 0}"
            )

        if not decrypted_pat and (not self.valves.username or not decrypted_password):
            raise ValueError(
                "Jira credentials not configured. Please provide either username/password or a Personal Access Token."
            )
        return Jira(
            self.valves.username,
            decrypted_password,
            self.valves.base_url,
            decrypted_pat,
        )

    async def get_issue(
        self,
        issue_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ):
        """Get detailed information about a Jira issue by its ID."""
        event_emitter = EventEmitter(__event_emitter__)

        try:
            await event_emitter.emit_status(f"Retrieving Jira issue {issue_id}", False)
            jira = self._get_jira_client()

            try:
                # Get issue data
                issue = jira.get_issue(issue_id)

                # Get comments
                comments = jira.get_comments(issue_id)

                # Generate rich HTML UI
                html_content = IssueFormatter.format_issue_details_html(issue, comments)

                await event_emitter.emit_status(
                    f"Successfully retrieved Jira issue {issue_id}", True
                )

                # Return HTML response with inline disposition for embedding
                return HTMLResponse(
                    content=html_content, headers={"Content-Disposition": "inline"}
                )

            except JiraApiError as e:
                self.logger.error(f"JiraApiError in get_issue: {str(e)}", exc_info=True)
                await event_emitter.emit_status(
                    f"Failed to get issue {issue_id}: {str(e)}", True, True
                )
                return None

        except Exception as e:
            self.logger.error(f"Exception in get_issue: {str(e)}", exc_info=True)
            await event_emitter.emit_status(
                f"Failed to get issue {issue_id}: {str(e)}", True, True
            )
            return None

    async def search_issues(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        max_results: int = 50,
        start_at: int = 0,
        __user__: dict = {},
    ):
        """Search for Jira issues using JQL or free text."""
        event_emitter = EventEmitter(__event_emitter__)

        try:
            await event_emitter.emit_status(f"Searching Jira for: {query}", False)
            jira = self._get_jira_client()
            results = jira.search(query, max_results, start_at)

            if not results["issues"]:
                await event_emitter.emit_status(
                    f"No issues found matching: {query}", True
                )
                return None

            # Generate rich HTML table
            html_content = IssueFormatter.format_issue_list_html(
                results["issues"],
                results["total"],
                results["displayed"],
                results["start_at"],
            )

            await event_emitter.emit_status(
                f"Found {results['total'] if results['total'] else results['displayed']+'+'} issues",
                True,
            )

            # Return HTML response with inline disposition for embedding
            return HTMLResponse(
                content=html_content, headers={"Content-Disposition": "inline"}
            )

        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to search issues: {str(e)}", True, True
            )
            return None

    async def create_issue(
        self,
        project_key: str,
        summary: str,
        description: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        issue_type: str = "Task",
        priority: str = None,
        __user__: dict = {},
    ):
        """Create a new Jira issue."""
        event_emitter = EventEmitter(__event_emitter__)

        try:
            await event_emitter.emit_status(
                f"Creating new {issue_type} in project {project_key}", False
            )

            jira = self._get_jira_client()
            result = jira.create_issue(
                project_key, summary, description, issue_type, priority
            )

            creation_time = datetime.now().strftime("%b %d, %Y %I:%M %p")

            # Format success message as a table for consistency
            success_message = f"""
### ✅ Issue Created Successfully

| Attribute | Value |
|-----------|-------|
| Key | [{result['key']}]({result['link']}) |
| Summary | {summary} |
| Type | {issue_type} |
| Project | {project_key} |
| Created | {creation_time} |
"""
            await event_emitter.emit_message(success_message)
            await event_emitter.emit_status(
                f"Successfully created issue {result['key']}", True
            )

            # Return nothing to avoid duplicate message
            return f"Successfully created issue {result['key']}"

        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to create issue: {str(e)}", True, True
            )
            return None

    async def add_comment(
        self,
        issue_id: str,
        comment: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ):
        """
        Add a comment to an existing Jira issue.
        :param issue_id: The ID of the issue (e.g., PROJECT-123)
        :param comment: The comment text to add
        :return: Comment details
        """
        event_emitter = EventEmitter(__event_emitter__)

        try:
            await event_emitter.emit_status(f"Adding comment to {issue_id}", False)

            jira = self._get_jira_client()
            result = jira.add_comment(issue_id, comment)

            confirmation = f"""
### 💬 Comment Added
Successfully added a comment to [{issue_id}]({result['issue_link']}).  
**Added at:** {result['created']}
"""
            await event_emitter.emit_message(confirmation)
            await event_emitter.emit_status(f"Comment added to {issue_id}", True)

            return None

        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to add comment: {str(e)}", True, True
            )
            return f"Error: {str(e)}"

    async def update_issue(
        self,
        issue_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        summary: str = None,
        description: str = None,
        priority: str = None,
        labels: str = None,
        __user__: dict = {},
    ):
        """
        Update one or more fields of an existing Jira issue.

        :param issue_id: The ID of the issue (e.g., PROJECT-123)
        :param summary: New issue title/summary
        :param description: New issue description
        :param priority: New priority (e.g., "High", "Medium", "Low")
        :param labels: Comma-separated labels (e.g., "bug,urgent,frontend")
        :return: Update confirmation
        """
        event_emitter = EventEmitter(__event_emitter__)

        try:
            # Parse labels if provided
            labels_list = None
            if labels:
                labels_list = [label.strip() for label in labels.split(",")]

            # Check that at least one field is provided
            if not any([summary, description, priority, labels]):
                await event_emitter.emit_status(
                    "No fields provided for update. Specify at least one: summary, description, priority, or labels",
                    True,
                    True,
                )
                return None

            fields_to_update = []
            if summary:
                fields_to_update.append("summary")
            if description:
                fields_to_update.append("description")
            if priority:
                fields_to_update.append("priority")
            if labels:
                fields_to_update.append("labels")

            await event_emitter.emit_status(
                f"Updating {', '.join(fields_to_update)} for {issue_id}", False
            )

            jira = self._get_jira_client()
            result = jira.update_issue(
                issue_id,
                summary=summary,
                description=description,
                priority=priority,
                labels=labels_list,
            )

            # Format confirmation message
            confirmation = f"""
### ✏️ Issue Updated

Successfully updated **{issue_id}**:

"""
            if summary:
                confirmation += f"- **Summary**: {summary}\n"
            if description:
                confirmation += (
                    f"- **Description**: Updated ({len(description)} characters)\n"
                )
            if priority:
                confirmation += f"- **Priority**: {priority}\n"
            if labels_list:
                confirmation += f"- **Labels**: {', '.join(labels_list)}\n"

            confirmation += f"\n🔗 **[View Issue]({result['link']})**"

            await event_emitter.emit_message(confirmation)
            await event_emitter.emit_status(f"Successfully updated {issue_id}", True)

            return None

        except Exception as e:
            await event_emitter.emit_status(
                f"Failed to update issue: {str(e)}", True, True
            )
            return f"Error: {str(e)}"

    async def draft_release_notes(
        self,
        tickets: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        version: str = "Unreleased",
        format: str = "markdown",
        group_by: str = "type",
        include_description: bool = False,
        __user__: dict = {},
    ):
        """
        Generate formatted release notes from a set of Jira tickets.
        
        :param tickets: Either a JQL query (e.g., "fixVersion = '1.0.0'") or comma-separated ticket IDs (e.g., "PROJ-1, PROJ-2")
        :param version: Version or release name (default: "Unreleased")
        :param format: Output format - "markdown" or "html" (default: "markdown")
        :param group_by: Grouping strategy - "type", "priority", "assignee", or "none" (default: "type")
        :param include_description: Whether to include full ticket descriptions (default: False)
        :return: Formatted release notes
        """
        event_emitter = EventEmitter(__event_emitter__)
        
        try:
            await event_emitter.emit_status(
                f"Generating release notes for {version}", False
            )
            
            jira = self._get_jira_client()
            
            # Determine if input is JQL or comma-separated IDs
            is_jql = any(op in tickets for op in ["=", "~", ">", "<", " AND ", " OR ", " IN "])
            
            issues_data = []
            
            if is_jql:
                # Use JQL search
                self.logger.debug(f"Using JQL query: {tickets}")
                results = jira.search(tickets, max_results=1000)
                
                if not results["issues"]:
                    await event_emitter.emit_status(
                        f"No issues found matching query", True, True
                    )
                    return "No issues found matching the query."
                
                # Fetch detailed info for each issue
                for issue in results["issues"]:
                    try:
                        detailed = jira.get_issue(issue["key"])
                        issues_data.append(detailed)
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch details for {issue['key']}: {e}")
                        # Use basic info from search
                        issues_data.append(issue)
            else:
                # Parse comma-separated ticket IDs
                ticket_ids = [t.strip() for t in tickets.split(",")]
                self.logger.debug(f"Fetching {len(ticket_ids)} tickets: {ticket_ids}")
                
                for ticket_id in ticket_ids:
                    if not ticket_id:
                        continue
                    try:
                        issue = jira.get_issue(ticket_id)
                        issues_data.append(issue)
                    except JiraApiError as e:
                        self.logger.warning(f"Failed to fetch {ticket_id}: {e}")
                        await event_emitter.emit_status(
                            f"Warning: Could not fetch {ticket_id}", False
                        )
            
            if not issues_data:
                await event_emitter.emit_status(
                    "No valid issues found", True, True
                )
                return "No valid issues were found."
            
            await event_emitter.emit_status(
                f"Processing {len(issues_data)} issues", False
            )
            
            # Group issues based on strategy
            grouped_issues = self._group_issues(issues_data, group_by)
            
            # Generate output based on format
            if format.lower() == "html":
                output = self._format_release_notes_html(
                    grouped_issues, version, group_by, include_description
                )
                await event_emitter.emit_status(
                    f"Release notes generated for {version}", True
                )
                return HTMLResponse(
                    content=output, headers={"Content-Disposition": "inline"}
                )
            else:
                output = self._format_release_notes_markdown(
                    grouped_issues, version, group_by, include_description
                )
                await event_emitter.emit_message(output)
                await event_emitter.emit_status(
                    f"Release notes generated for {version}", True
                )
                return f"Release notes generated successfully for {version} with {len(issues_data)} issues."
                
        except Exception as e:
            self.logger.error(f"Error generating release notes: {str(e)}", exc_info=True)
            await event_emitter.emit_status(
                f"Failed to generate release notes: {str(e)}", True, True
            )
            return f"Error: {str(e)}"
    
    def _group_issues(self, issues: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group issues by specified strategy"""
        groups = {}
        
        if group_by == "type":
            # Group by issue type with semantic categories
            type_mapping = {
                "feature": ["Story", "Epic", "New Feature", "Enhancement"],
                "bug": ["Bug", "Defect"],
                "improvement": ["Improvement", "Task", "Sub-task"],
                "security": ["Security"],
            }
            
            for issue in issues:
                issue_type = issue.get("type", "Unknown")
                
                # Find which category this type belongs to
                category = "other"
                for cat, types in type_mapping.items():
                    if any(t.lower() in issue_type.lower() for t in types):
                        category = cat
                        break
                
                if category not in groups:
                    groups[category] = []
                groups[category].append(issue)
                
        elif group_by == "priority":
            for issue in issues:
                priority = issue.get("priority", "Not set")
                if priority not in groups:
                    groups[priority] = []
                groups[priority].append(issue)
                
        elif group_by == "assignee":
            for issue in issues:
                assignee = issue.get("assignee", "Unassigned")
                if assignee not in groups:
                    groups[assignee] = []
                groups[assignee].append(issue)
        else:
            # No grouping
            groups["all"] = issues
        
        return groups
    
    def _format_release_notes_markdown(
        self, 
        grouped_issues: Dict[str, List[Dict[str, Any]]], 
        version: str,
        group_by: str,
        include_description: bool
    ) -> str:
        """Format release notes as markdown"""
        
        # Category display names and emojis
        category_display = {
            "feature": "🎯 New Features",
            "bug": "🐛 Bug Fixes",
            "improvement": "⚡ Improvements",
            "security": "🔒 Security Updates",
            "other": "📝 Other Changes",
        }
        
        output = f"# Release Notes - {version}\n\n"
        output += f"**Generated:** {datetime.now().strftime('%B %d, %Y')}\n\n"
        output += f"**Total Changes:** {sum(len(issues) for issues in grouped_issues.values())} issues\n\n"
        output += "---\n\n"
        
        # Sort categories in a logical order
        category_order = ["feature", "improvement", "bug", "security", "other"]
        
        for category in category_order:
            if category not in grouped_issues:
                continue
                
            issues = grouped_issues[category]
            if not issues:
                continue
            
            if group_by == "type":
                header = category_display.get(category, category.title())
            else:
                header = category
            
            output += f"## {header}\n\n"
            
            for issue in issues:
                key = issue.get("key", "UNKNOWN")
                title = issue.get("title", issue.get("summary", "No title"))
                link = issue.get("link", "#")
                assignee = issue.get("assignee", "Unassigned")
                
                output += f"- **[{key}]({link})** - {title}"
                
                if group_by != "assignee":
                    output += f" _(by {assignee})_"
                
                output += "\n"
                
                if include_description:
                    desc = issue.get("description", "")
                    if desc and desc != "<p><em>No description provided</em></p>":
                        # Clean HTML from description
                        clean_desc = re.sub(r"<[^>]+>", "", desc)
                        clean_desc = html.unescape(clean_desc).strip()
                        if clean_desc:
                            output += f"  > **Description:** {clean_desc[:500]}{'...' if len(clean_desc) > 500 else ''}\n"
                    output += "\n"
            
            output += "\n"
        
        # Handle other grouping strategies
        if group_by in ["priority", "assignee"]:
            for group_name in sorted(grouped_issues.keys()):
                issues = grouped_issues[group_name]
                if not issues:
                    continue
                    
                output += f"## {group_name}\n\n"
                
                for issue in issues:
                    key = issue.get("key", "UNKNOWN")
                    title = issue.get("title", issue.get("summary", "No title"))
                    link = issue.get("link", "#")
                    issue_type = issue.get("type", "Task")
                    
                    output += f"- **[{key}]({link})** ({issue_type}) - {title}\n"
                    
                    if include_description:
                        desc = issue.get("description", "")
                        if desc and desc != "<p><em>No description provided</em></p>":
                            clean_desc = re.sub(r"<[^>]+>", "", desc)
                            clean_desc = html.unescape(clean_desc).strip()
                            if clean_desc:
                                output += f"  > **Description:** {clean_desc[:500]}{'...' if len(clean_desc) > 500 else ''}\n"
                        output += "\n"
                
                output += "\n"
        
        return output
    
    def _format_release_notes_html(
        self,
        grouped_issues: Dict[str, List[Dict[str, Any]]],
        version: str,
        group_by: str,
        include_description: bool
    ) -> str:
        """Format release notes as HTML"""
        
        category_display = {
            "feature": "🎯 New Features",
            "bug": "🐛 Bug Fixes",
            "improvement": "⚡ Improvements",
            "security": "🔒 Security Updates",
            "other": "📝 Other Changes",
        }
        
        total_issues = sum(len(issues) for issues in grouped_issues.values())
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {{
            --bg-primary: #FFFFFF;
            --bg-secondary: #F4F5F7;
            --bg-hover: #EBECF0;
            --text-primary: #172B4D;
            --text-secondary: #5E6C84;
            --border-color: #DFE1E6;
            --accent-color: #0052CC;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .header {{
            border-bottom: 3px solid var(--accent-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            color: var(--text-primary);
            font-size: 28px;
            font-weight: 600;
        }}
        
        .metadata {{
            display: flex;
            gap: 20px;
            color: var(--text-secondary);
            font-size: 14px;
        }}
        
        .metadata-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .section {{
            margin-bottom: 35px;
        }}
        
        .section-header {{
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .issue-item {{
            padding: 12px 15px;
            margin-bottom: 8px;
            background: var(--bg-secondary);
            border-radius: 5px;
            border-left: 3px solid var(--accent-color);
            transition: background 0.2s;
        }}
        
        .issue-item:hover {{
            background: var(--bg-hover);
        }}
        
        .issue-key {{
            font-weight: 600;
            color: var(--accent-color);
            text-decoration: none;
            margin-right: 8px;
        }}
        
        .issue-key:hover {{
            text-decoration: underline;
        }}
        
        .issue-title {{
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        .issue-meta {{
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 5px;
        }}
        
        .issue-description {{
            margin-top: 8px;
            padding-left: 15px;
            border-left: 2px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 13px;
            font-style: italic;
        }}
        
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 5px;
        }}
        
        .badge-feature {{
            background-color: #0052CC;
            color: white;
        }}
        
        .badge-bug {{
            background-color: #DE350B;
            color: white;
        }}
        
        .badge-improvement {{
            background-color: #FF991F;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 Release Notes - {html.escape(version)}</h1>
        <div class="metadata">
            <div class="metadata-item">
                <span>📅 Generated:</span>
                <strong>{datetime.now().strftime('%B %d, %Y')}</strong>
            </div>
            <div class="metadata-item">
                <span>📊 Total Changes:</span>
                <strong>{total_issues} issues</strong>
            </div>
        </div>
    </div>
"""
        
        category_order = ["feature", "improvement", "bug", "security", "other"]
        
        for category in category_order:
            if category not in grouped_issues:
                continue
                
            issues = grouped_issues[category]
            if not issues:
                continue
            
            if group_by == "type":
                header = category_display.get(category, category.title())
            else:
                header = category
            
            html_content += f"""
    <div class="section">
        <div class="section-header">{html.escape(header)}</div>
"""
            
            for issue in issues:
                key = issue.get("key", "UNKNOWN")
                title = issue.get("title", issue.get("summary", "No title"))
                link = issue.get("link", "#")
                assignee = issue.get("assignee", "Unassigned")
                issue_type = issue.get("type", "Task")
                
                html_content += f"""
        <div class="issue-item">
            <div>
                <a href="{html.escape(link)}" target="_blank" class="issue-key">{html.escape(key)}</a>
                <span class="issue-title">{html.escape(title)}</span>
"""
                
                if group_by == "type":
                    badge_class = f"badge-{category}"
                else:
                    badge_class = "badge-feature"
                
                html_content += f"""                <span class="badge {badge_class}">{html.escape(issue_type)}</span>
            </div>
"""
                
                if group_by != "assignee":
                    html_content += f"""            <div class="issue-meta">Assignee: {html.escape(assignee)}</div>
"""
                
                if include_description:
                    desc = issue.get("description", "")
                    if desc and desc != "<p><em>No description provided</em></p>":
                        clean_desc = re.sub(r"<[^>]+>", "", desc)
                        clean_desc = html.unescape(clean_desc).strip()
                        if clean_desc:
                            truncated = clean_desc[:500] + ("..." if len(clean_desc) > 500 else "")
                            html_content += f"""            <div class="issue-description"><strong>Description:</strong> {html.escape(truncated)}</div>
"""
                
                html_content += """        </div>
"""
            
            html_content += """    </div>
"""
        
        # Handle other grouping strategies
        if group_by in ["priority", "assignee"]:
            for group_name in sorted(grouped_issues.keys()):
                issues = grouped_issues[group_name]
                if not issues:
                    continue
                
                html_content += f"""
    <div class="section">
        <div class="section-header">{html.escape(group_name)}</div>
"""
                
                for issue in issues:
                    key = issue.get("key", "UNKNOWN")
                    title = issue.get("title", issue.get("summary", "No title"))
                    link = issue.get("link", "#")
                    issue_type = issue.get("type", "Task")
                    
                    html_content += f"""
        <div class="issue-item">
            <div>
                <a href="{html.escape(link)}" target="_blank" class="issue-key">{html.escape(key)}</a>
                <span class="issue-title">{html.escape(title)}</span>
                <span class="badge badge-feature">{html.escape(issue_type)}</span>
            </div>
"""
                    
                    if include_description:
                        desc = issue.get("description", "")
                        if desc and desc != "<p><em>No description provided</em></p>":
                            clean_desc = re.sub(r"<[^>]+>", "", desc)
                            clean_desc = html.unescape(clean_desc).strip()
                            if clean_desc:
                                truncated = clean_desc[:500] + ("..." if len(clean_desc) > 500 else "")
                                html_content += f"""            <div class="issue-description"><strong>Description:</strong> {html.escape(truncated)}</div>
"""
                    
                    html_content += """        </div>
"""
                
                html_content += """    </div>
"""
        
        html_content += """
</body>
</html>
"""
        return html_content

    async def help(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __user__: dict = {},
    ):
        """
        Display help information about available Jira commands and features.
        """
        help_message = """
# 🎯 Jira Agent - Available Features

## 📋 **Issue Operations**

### 1. **View Issue Details**
Get comprehensive information about a specific Jira issue with rich Jira-styled UI.
- **Command**: `get_issue`
- **Parameters**: `issue_id` (e.g., "PROJ-123")
- **Returns**: Interactive HTML view with color-coded badges, formatted description, and all comments
- **Features**: Jira colors, clickable links, professional layout, embedded in chat
- **Example**: "Show me details for AAHM-140"

### 2. **Search for Issues**
Search for Jira issues using JQL (Jira Query Language) or free text with rich table UI.
- **Command**: `search_issues`
- **Parameters**: 
  - `query` - JQL query or free text search
  - `max_results` - Number of results (default: 50, max: 100)
  - `start_at` - Starting position for pagination (default: 0)
- **Default Behavior**: Automatically excludes Done/Closed/Resolved issues (shows only active work)
- **Returns**: Interactive HTML table with color-coded status/priority badges, clickable links
- **Features**: Hover effects, sortable columns, pagination info, Jira-style colors
- **Examples**:
  - "Search for issues assigned to me" (excludes completed by default)
  - "Find all bugs in project DEMO" (only active bugs)
  - "Search for: status = 'In Progress' AND priority = High"
  - "Search for issues including Done status" (to include completed issues)

### 3. **Create New Issue**
Create a new Jira issue in any project.
- **Command**: `create_issue`
- **Parameters**:
  - `project_key` - Project code (e.g., "PROJ")
  - `summary` - Issue title
  - `description` - Detailed description
  - `issue_type` - Type (Task, Bug, Story, etc.) [default: Task]
  - `priority` - Priority level (optional)
- **Returns**: New issue key and link
- **Example**: "Create a bug in DEMO project about login issues"

### 4. **Add Comment**
Add a comment to an existing issue.
- **Command**: `add_comment`
- **Parameters**:
  - `issue_id` - Issue key (e.g., "PROJ-123")
  - `comment` - Comment text
- **Returns**: Comment confirmation with timestamp
- **Example**: "Add comment to AAHM-140: Investigation completed"

### 5. **Update Issue**
Update issue fields like description, summary, priority, or labels.
- **Command**: `update_issue`
- **Parameters**:
  - `issue_id` - Issue key (e.g., "PROJ-123")
  - `summary` - New title (optional)
  - `description` - New description (optional)
  - `priority` - New priority: High, Medium, Low (optional)
  - `labels` - Comma-separated labels (optional)
- **Returns**: Confirmation showing updated fields
- **Examples**:
  - "Update description for AAHM-140 to: New detailed description"
  - "Update priority for DEMO-5 to High"
  - "Update AAHM-140 summary to: Better ticket title"
  - "Update AAHM-140 labels to: bug,urgent,frontend"

### 6. **Draft Release Notes**
Generate formatted release notes from a set of Jira tickets with smart grouping and professional styling.
- **Command**: `draft_release_notes`
- **Parameters**:
  - `tickets` - JQL query or comma-separated ticket IDs (e.g., "PROJ-1, PROJ-2" or "fixVersion = '1.0.0'")
  - `version` - Version/release name (default: "Unreleased")
  - `format` - Output format: "markdown" or "html" (default: "markdown")
  - `group_by` - Grouping: "type", "priority", "assignee", or "none" (default: "type")
  - `include_description` - Include ticket descriptions (default: False)
- **Returns**: Beautiful release notes with automatic categorization
- **Smart Grouping by Type**:
  - 🎯 **New Features** - Stories, Epics, Enhancements
  - ⚡ **Improvements** - Tasks, Improvements
  - 🐛 **Bug Fixes** - Bugs, Defects
  - 🔒 **Security Updates** - Security issues
  - 📝 **Other Changes** - Everything else
- **Examples**:
  - "Draft release notes for PROJ-123, PROJ-124, PROJ-125 for version 2.0.0"
  - "Generate release notes for fixVersion = '1.5.0'"
  - "Create release notes for project = DEMO AND status = Done AND updated >= -30d"
  - "Draft HTML release notes for assignee = currentUser() for version Sprint-42"
  - "Generate release notes grouped by assignee for PROJ-1, PROJ-2, PROJ-3"

---

## 🔍 **Search Examples**

### Free Text Search:
- "Find issues about authentication"
- "Search for payment bugs"

### JQL Search:
- "project = DEMO AND status = 'In Progress'"
- "assignee = currentUser() AND status = Open" (auto-translates to exclude Done/Closed/Resolved)
- "assignee = currentUser() ORDER BY updated DESC" (active issues only, by default)
- "priority = High AND created >= -7d" (active high-priority issues)

### Include Completed Issues:
- "assignee = currentUser() AND status = Done" (explicitly requests completed)
- "project = DEMO" + mention "including closed" in natural language
- Any JQL that mentions Done/Closed/Resolved will override the default filter

### Pagination:
- Default: Returns 50 results
- "Show me the next 50 results" (continues from where you left off)
- Use `start_at` parameter to jump to specific position
- Automatic suggestions appear when more results are available

---

## 📊 **What You Get**

### Issue Details Include (Rich HTML UI):
- 🎨 **Jira-styled interface** with color-coded status and priority badges
- 🔗 Clickable issue key and direct Jira link
- ✅ Status badges with authentic Jira colors (green for done, blue for in progress, etc.)
- 🔥 Priority badges (red for high, orange for medium, green for low)
- 📋 Issue Type badge (purple)
- 👤 Assignee & Reporter in clean metadata panel
- 📅 Created & Updated dates
- 📝 **Full description** with preserved formatting
- 💬 All comments with authors & timestamps in bordered panels
- 📱 Responsive design that fits perfectly in chat

### Search Results Include (Rich HTML Table):
- 🎨 **Jira-styled table** with clean, professional design
- 🔗 **Clickable issue keys** that open in Jira
- 📊 **Project names** clearly displayed
- 📝 **Issue summaries** with full text
- ✅ **Color-coded status badges** (green/blue/purple)
- 🔥 **Color-coded priority badges** (red/orange/green)
- 📋 **Issue types** (Bug, Task, Story, etc.)
- 📅 **Last updated dates** formatted nicely
- 📄 **Pagination info** at the top (e.g., "Showing 1-50 of 176")
- 🖱️ **Hover effects** on table rows for better UX
- 📱 **Responsive** table that works on all screen sizes

---

## 💡 **Tips**

1. **Default Filtering**: Searches automatically exclude Done/Closed/Resolved issues to show only active work
2. **Include Completed**: To see completed issues, explicitly mention them: `"status = Done"` or `"including resolved"`
3. **Issue Keys**: Always use the full key format (e.g., "PROJECT-123")
4. **"Open" Status**: Use `status = Open` - automatically translates to exclude Done/Closed/Resolved
5. **JQL Power**: Learn JQL for advanced searches (see [Atlassian JQL docs](https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jira-query-language-jql/))
6. **Batch Operations**: Search first, then act on specific issues
7. **Links**: All responses include direct Jira links for quick access
"""

        if __event_emitter__:
            event_emitter = EventEmitter(__event_emitter__)
            await event_emitter.emit_message(help_message)
            await event_emitter.emit_status("Help information displayed", True)

        return help_message
