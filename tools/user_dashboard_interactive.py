"""
title: Interactive User Dashboard
author: open-webui
version: 2.2.0
description: Interactive dashboard with rich UI embedding. Shows user statistics with live charts and visualizations. Admins see system-wide metrics, users see personal stats. Enhanced model tracking focused on admin-relevant insights.
required_open_webui_version: 0.3.9
requirements: cryptography
"""

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Optional, Callable, Awaitable, Any
from fastapi.responses import HTMLResponse
import time
from datetime import datetime
import aiohttp
import asyncio
import logging
import hashlib
import base64
import os
import json
import math
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)


class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        if not value or value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len("encrypted:"):]
        try:
            encrypted_part = value[len("encrypted:"):]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    def get_decrypted(self) -> str:
        return self.decrypt(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Simplified schema for JSON schema generation - just treat as string
        # The encryption/decryption still works via the class methods
        return core_schema.str_schema()


class Tools:
    class Valves(BaseModel):
        ENABLE_DASHBOARD: bool = Field(
            default=True,
            description="Enable/disable the interactive dashboard"
        )
        SHOW_ACTIVE_USERS: bool = Field(
            default=True,
            description="Show currently active users"
        )
        TOP_USERS_COUNT: int = Field(
            default=10,
            description="Number of top users to show in admin dashboard"
        )
        TOP_MODELS_COUNT: int = Field(
            default=10,
            description="Number of top models to show"
        )
        ENABLE_LITELLM_COSTS: bool = Field(
            default=False,
            description="Enable LiteLLM cost tracking (requires LiteLLM proxy)"
        )
        LITELLM_BASE_URL: str = Field(
            default="http://localhost:4000",
            description="LiteLLM proxy base URL"
        )
        LITELLM_API_KEY: EncryptedStr = Field(
            default="",
            description="LiteLLM admin API key (encrypted)"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _create_donut_chart_svg(self, labels: list, data: list, colors: list, width: int = 220, height: int = 220) -> str:
        """Generate Chart.js doughnut chart"""
        import random
        chart_id = f"chart_{random.randint(10000, 99999)}"
        
        if sum(data) == 0:
            return f'<div style="height: {height}px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">No data</div>'
        
        # Use base64 encoding to avoid any HTML/markdown processing issues
        import base64
        labels_json = json.dumps([str(label) for label in labels])
        data_json = json.dumps(data)
        colors_json = json.dumps(colors)
        
        # Base64 encode to completely avoid any transformation
        labels_b64 = base64.b64encode(labels_json.encode()).decode()
        data_b64 = base64.b64encode(data_json.encode()).decode()
        colors_b64 = base64.b64encode(colors_json.encode()).decode()
        
        return f'''
            <div id="container_{chart_id}" style="height: {height}px; max-height: {height}px; display: flex; justify-content: center; align-items: center;"
                 data-labels="{labels_b64}"
                 data-values="{data_b64}"
                 data-colors="{colors_b64}"></div>
            <script>
                (function() {{
                    try {{
                        const container = document.getElementById('container_{chart_id}');
                        const labelsB64 = container.dataset.labels;
                        const valuesB64 = container.dataset.values;
                        const colorsB64 = container.dataset.colors;
                        const canvas = document.createElement('canvas');
                        canvas.id = '{chart_id}';
                        canvas.style.maxWidth = '200px';
                        canvas.style.maxHeight = '200px';
                        container.appendChild(canvas);
                        const chartLabels = JSON.parse(atob(labelsB64));
                        const chartData = JSON.parse(atob(valuesB64));
                        const chartColors = JSON.parse(atob(colorsB64));
                        new Chart(canvas, {{
                            type: 'doughnut',
                            data: {{
                                labels: chartLabels,
                                datasets: [{{ data: chartData, backgroundColor: chartColors, borderWidth: 2, borderColor: 'rgba(255, 255, 255, 0.2)' }}]
                            }},
                            options: {{
                                responsive: true,
                                plugins: {{
                                    legend: {{
                                        position: 'bottom',
                                        align: 'center'
                                    }},
                                    tooltip: {{ enabled: true, backgroundColor: 'rgba(0, 0, 0, 0.8)', titleColor: '#fff', bodyColor: '#fff', padding: 12, cornerRadius: 8 }}
                                }}
                            }}
                        }});
                    }} catch(e) {{ console.error('Donut chart error:', e); }}
                }})();
            </script>
        '''

    def _create_bar_chart_svg(self, labels: list, data: list, colors: list | str, width: int = 250, height: int = 220) -> str:
        """Generate Chart.js bar chart"""
        import random
        chart_id = f"chart_{random.randint(10000, 99999)}"
        
        if not data or max(data) == 0:
            return f'<div style="height: {height}px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">No data</div>'
        
        # Handle colors: either list of colors or single color
        if isinstance(colors, list):
            colors_list = colors
        else:
            colors_list = [colors] * len(data)
        
        # Use base64 encoding to avoid any HTML/markdown processing issues
        import base64
        labels_json = json.dumps([str(label) for label in labels])
        data_json = json.dumps(data)
        colors_json = json.dumps(colors_list)
        
        # Base64 encode to completely avoid any transformation
        labels_b64 = base64.b64encode(labels_json.encode()).decode()
        data_b64 = base64.b64encode(data_json.encode()).decode()
        colors_b64 = base64.b64encode(colors_json.encode()).decode()
        
        return f'''
            <div id="container_{chart_id}" style="min-height: {height}px; max-height: {height}px;"
                 data-labels="{labels_b64}"
                 data-values="{data_b64}"
                 data-colors="{colors_b64}"></div>
            <script>
                (function() {{
                    try {{
                        const container = document.getElementById('container_{chart_id}');
                        const canvas = document.createElement('canvas');
                        canvas.id = '{chart_id}';
                        container.appendChild(canvas);
                        
                        // Decode base64 and parse JSON
                        const chartLabels = JSON.parse(atob(container.dataset.labels));
                        const chartData = JSON.parse(atob(container.dataset.values));
                        const chartColors = JSON.parse(atob(container.dataset.colors));
                        
                        console.log('=== BAR CHART DEBUG ({chart_id}) ===');
                        console.log('chartData:', chartData);
                        
                        new Chart(canvas, {{
                            type: 'bar',
                            data: {{
                                labels: chartLabels,
                                datasets: [{{
                                    data: chartData,
                                    backgroundColor: chartColors,
                                    borderRadius: 6,
                                    borderSkipped: false
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: true,
                                plugins: {{
                                    legend: {{ display: false }},
                                    tooltip: {{
                                        enabled: true,
                                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                        titleColor: '#fff',
                                        bodyColor: '#fff',
                                        padding: 12,
                                        cornerRadius: 8,
                                        displayColors: true
                                    }}
                                }},
                                scales: {{
                                    y: {{
                                        beginAtZero: true,
                                        grid: {{ color: 'rgba(139, 148, 184, 0.1)', drawBorder: false }},
                                        ticks: {{ color: '#8b94b8', font: {{ size: 11 }} }}
                                    }},
                                    x: {{
                                        grid: {{ display: false }},
                                        ticks: {{ color: '#8b94b8', font: {{ size: 12 }} }}
                                    }}
                                }}
                            }}
                        }});
                    }} catch(e) {{
                        console.error('Bar chart error:', e);
                    }}
                }})();
            </script>
        '''

    def _create_horizontal_bar_chart_svg(self, labels: list, data: list, color: str, width: int = 600, height: int = 250) -> str:
        """Generate Chart.js horizontal bar chart"""
        import random
        chart_id = f"chart_{random.randint(10000, 99999)}"
        
        if not data or max(data) == 0:
            return f'<div style="height: {height}px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">No data</div>'
        
        labels_json = json.dumps([str(label) for label in labels])
        data_json = json.dumps(data)
        labels_b64 = base64.b64encode(labels_json.encode()).decode()
        data_b64 = base64.b64encode(data_json.encode()).decode()
        
        return f'''
            <div id="container_{chart_id}" style="min-height: {height}px; max-height: {height}px;"
                 data-labels="{labels_b64}" data-values="{data_b64}" data-color="{color}"></div>
            <script>
                (function() {{
                    try {{
                        const container = document.getElementById('container_{chart_id}');
                        const canvas = document.createElement('canvas');
                        canvas.id = '{chart_id}';
                        container.appendChild(canvas);
                        const chartLabels = JSON.parse(atob(container.dataset.labels));
                        const chartData = JSON.parse(atob(container.dataset.values));
                        const chartColor = container.dataset.color;
                        new Chart(canvas, {{
                            type: 'bar',
                            data: {{ labels: chartLabels, datasets: [{{ data: chartData, backgroundColor: chartColor, borderRadius: 6, borderSkipped: false }}] }},
                            options: {{
                                indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                                plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: true, backgroundColor: 'rgba(0,0,0,0.8)', padding: 12, cornerRadius: 8 }} }},
                                scales: {{ x: {{ beginAtZero: true, grid: {{ color: 'rgba(139,148,184,0.1)', drawBorder: false }}, ticks: {{ color: '#8b94b8', font: {{ size: 11 }} }} }}, y: {{ grid: {{ display: false }}, ticks: {{ color: '#8b94b8', font: {{ size: 12 }}, crossAlign: 'far' }} }} }}
                            }}
                        }});
                    }} catch(e) {{ console.error('Horizontal bar chart error:', e); }}
                }})();
            </script>
        '''

    def _create_distribution_chart(self, title: str, labels: list, data: list, colors: list, percentiles: dict, top_10_share: float, width: int = 600, height: int = 200) -> str:
        """Generate 100% stacked bar chart for percentile distributions"""
        import random
        chart_id = f"chart_{random.randint(10000, 99999)}"
        
        if not data or sum(data) == 0:
            return f'<div style="height: {height}px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">No data</div>'
        
        # Calculate percentages
        total = sum(data)
        percentages = [round((val / total * 100), 1) if total > 0 else 0 for val in data]
        
        # Clean labels (remove user counts from labels for cleaner display)
        clean_labels = [
            'Zero use',
            'Below 50th %ile',
            '50th-75th %ile', 
            '75th-90th %ile',
            'Above 90th %ile (Top 10%)'
        ]
        
        # Create datasets - one per category for stacked bar
        datasets = []
        for i, (label, value, color, pct) in enumerate(zip(clean_labels, data, colors, percentages)):
            if value > 0:  # Only include non-zero segments
                datasets.append({
                    'label': f'{label}: {value} users ({pct}%)',
                    'data': [pct],
                    'backgroundColor': color,
                    'borderWidth': 0
                })
        
        # Format datasets as JavaScript array manually
        datasets_parts = []
        for ds in datasets:
            label = ds['label'].replace("'", "\\'")  # Escape quotes
            data_val = ds['data'][0]
            bg_color = ds['backgroundColor']
            datasets_parts.append(f"{{'label':'{label}','data':[{data_val}],'backgroundColor':'{bg_color}','borderWidth':0}}")
        datasets_str = '[' + ','.join(datasets_parts) + ']'
        
        return f'''
            <div style="display: flex !important; flex-direction: column !important; flex: 1 1 100% !important; flex-basis: 100% !important; width: 100% !important; max-width: 100% !important; min-height: 300px !important; align-items: stretch !important; justify-content: flex-start !important; margin: 0 !important; padding: 0 !important;">
                <div style="flex-shrink: 0; margin-bottom: 15px; padding: 10px; background: rgba(139, 148, 184, 0.08); border-radius: 8px; text-align: center;">
                    <strong style="color: var(--text-primary); font-size: 14px;">Top 10% account for <span style="color: var(--accent); font-size: 16px;">{top_10_share}%</span> of all activity</strong><br>
                    <span style="color: var(--text-muted); font-size: 10px; font-family: monospace;">Percentiles: {percentiles.get('50th', 0):,} at 50th | {percentiles.get('75th', 0):,} at 75th | {percentiles.get('90th', 0):,} at 90th</span>
                </div>
                <div id="container_{chart_id}" style="flex: 1; min-height: {height + 120}px;"></div>
            </div>
            <script>
                (function() {{
                    const container = document.getElementById('container_{chart_id}');
                    const canvas = document.createElement('canvas');
                    canvas.id = '{chart_id}';
                    canvas.style.display = 'block';
                    canvas.style.width = '100%';
                    container.appendChild(canvas);
                    
                    const chartDatasets = {datasets_str};
                    
                    new Chart(canvas, {{
                        type: 'bar',
                        data: {{
                            labels: ['User Distribution'],
                            datasets: chartDatasets
                        }},
                        options: {{
                            indexAxis: 'y',
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    stacked: true,
                                    max: 100,
                                    grid: {{
                                        display: false
                                    }},
                                    ticks: {{
                                        color: '#8b94b8',
                                        callback: function(value) {{
                                            return value + '%';
                                        }},
                                        font: {{
                                            size: 11
                                        }}
                                    }}
                                }},
                                y: {{
                                    stacked: true,
                                    display: false
                                }}
                            }},
                            plugins: {{
                                legend: {{
                                    display: true,
                                    position: 'bottom',
                                    align: 'start',
                                    labels: {{
                                        color: function(context) {{
                                            // Detect theme from document
                                            const theme = document.documentElement.getAttribute('data-theme');
                                            return theme === 'dark' ? '#9fb2ffcc' : '#2d3748';
                                        }},
                                        padding: 15,
                                        font: {{
                                            size: 13,
                                            weight: '500',
                                            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                                        }},
                                        usePointStyle: false,
                                        boxWidth: 24,
                                        boxHeight: 16,
                                        generateLabels: function(chart) {{
                                            const datasets = chart.data.datasets;
                                            const theme = document.documentElement.getAttribute('data-theme');
                                            const textColor = theme === 'dark' ? '#9fb2ffcc' : '#2d3748';
                                            
                                            return datasets.map((dataset, i) => ({{
                                                text: dataset.label,
                                                fillStyle: dataset.backgroundColor,
                                                strokeStyle: dataset.backgroundColor,
                                                lineWidth: 0,
                                                hidden: false,
                                                index: i,
                                                fontColor: textColor
                                            }}));
                                        }}
                                    }}
                                }},
                                tooltip: {{
                                    enabled: true,
                                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                                    titleColor: '#fff',
                                    bodyColor: '#fff',
                                    padding: 16,
                                    cornerRadius: 10
                                }}
                            }}
                        }}
                    }});
                }})();
            </script>
        '''

    async def get_capabilities(
        self,
        __user__=None,
    ):
        """
        Explain what this dashboard tool can do.
        
        WHEN TO USE:
        - User asks: "what can you do?", "help", "what are your capabilities?"
        - User wants to know what features are available
        - User is unsure how to use the dashboard
        
        :return: Description of available features
        """
        user_role = __user__.get("role", "user") if __user__ else "user"
        
        if user_role == "admin":
            return """## 📊 Dashboard Tool Capabilities

I can show you an **interactive visual dashboard** with charts and statistics about your Open WebUI platform.

### What I Can Show You (Admin View):
- **👥 Adoption Analytics** - User registrations, growth trends, active users
- **📊 Usage Metrics** - Total chats, models used, token consumption
- **⭐ Quality Metrics** - User feedback and satisfaction data
- **📚 Content & Knowledge** - Files, storage, and knowledge base stats
- **🔥 Top Users** - Most active users by chats and tokens
- **🤖 Model Usage** - Which AI models are being used most

### Visual Dashboard:
- **"show dashboard"** - Display the full interactive dashboard with charts
- **"show stats"** / **"view analytics"** - Same as above

### Data Queries (No Charts):
For specific questions without visuals, just ask naturally:
- **"how many users?"** - Get user counts
- **"total chats today?"** - Get chat statistics
- **"which models are used most?"** - Get model rankings
- **"top users by tokens"** - Get user leaderboards
- **"what's the spend?"** - Get cost data (if LiteLLM enabled)

Available data categories: `users`, `chats`, `models`, `tokens`, `files`, `groups`, `feedback`, `spend`, `knowledge`

The dashboard includes interactive charts with tooltips, a dark/light theme toggle, and hover effects on all cards."""
        else:
            return """## 📊 Dashboard Tool Capabilities

I can show you an **interactive visual dashboard** with your personal statistics.

### What I Can Show You:
- **💬 Your Chat Activity** - Total chats, recent activity
- **📈 Usage Over Time** - Your engagement trends
- **🤖 Models Used** - Which AI models you've interacted with

### Visual Dashboard:
- **"show dashboard"** - Display your personal dashboard with charts
- **"show my stats"** / **"view my usage"** - Same as above

### Data Queries (No Charts):
For specific questions without visuals, just ask naturally:
- **"how many chats do I have?"** - Get your chat count
- **"my recent activity"** - Get your usage summary

The dashboard includes interactive charts with tooltips and a dark/light theme toggle."""

    async def get_dashboard(
        self,
        __user__=None,
        __event_emitter__=None,
    ):
        """
        Display interactive visual dashboard with live charts, graphs, and statistics.
        Shows beautiful HTML interface with data visualizations embedded in chat.
        
        WHEN TO USE:
        - User says: "show dashboard", "display stats", "view analytics", "see metrics", "open dashboard"
        - User wants to SEE or VIEW visual charts/graphs/visualizations
        - User requests: "show me the dashboard", "dashboard please", "show stats", "view usage"
        - Admin wants: system overview, platform usage, user activity visualization, full reports
        - Regular users want: personal statistics, chat history, usage summary, my stats
        
        OUTPUT: Interactive HTML with charts and graphs (embedded iframe in chat)
        For raw JSON data without visuals, use get_dashboard_data() instead.

        :param __user__: User information
        :param __event_emitter__: Event emitter for status updates
        :return: Interactive HTML dashboard with embedded charts
        """
        
        log.info(f"[DASHBOARD] get_dashboard called for user role: {__user__.get('role', 'unknown') if __user__ else 'unknown'}")
        
        if not self.valves.ENABLE_DASHBOARD:
            return "❌ Dashboard is currently disabled."
        
        user_id = __user__.get("id") if __user__ else None
        user_role = __user__.get("role", "user") if __user__ else "user"
        user_name = __user__.get("name", "User") if __user__ else "User"
        
        if not user_id:
            return "❌ Unable to retrieve user information."
        
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Generating interactive dashboard...",
                        "done": False
                    }
                })
            
            from open_webui.models.users import Users
            from open_webui.models.chats import Chats
            from open_webui.models.feedbacks import Feedbacks
            from open_webui.models.files import Files
            from open_webui.models.models import Models
            from open_webui.models.groups import Groups
            from open_webui.models.knowledge import Knowledges
            
            if user_role == "admin":
                html, data = await self._generate_admin_dashboard_html(
                    Users, Chats, Feedbacks, Files, Models, Groups, Knowledges, user_name,
                    categories=None,
                    event_emitter=__event_emitter__
                )
            else:
                html, data = self._generate_user_dashboard_html(
                    Users, Chats, Feedbacks, user_id, user_name,
                    event_emitter=__event_emitter__
                )
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Dashboard ready!",
                        "done": True
                    }
                })
            
            headers = {"Content-Disposition": "inline"}
            return HTMLResponse(content=html, headers=headers)
            
        except Exception as e:
            log.exception(f"Dashboard generation error: {e}")
            return f"❌ **Error generating dashboard:** {str(e)}"

    async def get_dashboard_data(
        self,
        categories=None,
        __user__=None,
    ):
        """
        Get raw dashboard statistics as structured JSON for programmatic analysis.
        Returns numbers, metrics, and data WITHOUT any charts or visual display.
        
        WHEN TO USE:
        - User asks: "how many users", "what's the total", "count of", "number of", "who are"
        - User wants to ANALYZE, COMPARE, or CALCULATE with specific metrics
        - Questions like: "who are top users?", "which model is most used?", "token usage?", "what's the spend?"
        - User needs: specific stats, numerical data, rankings, totals, comparisons, lists
        - Follow-up queries: "show me just the user count", "how many chats today?", "top spenders?"
        
        OUTPUT: Pure JSON dictionary with structured metrics (no charts)
        For visual dashboards with charts, use get_dashboard() instead.
        
        IMPORTANT: Use the 'categories' parameter to fetch ONLY the data needed!
        This dramatically improves performance by skipping unnecessary processing.
        
        EXAMPLES:
        - "How many users?" → categories=["users"]
        - "Total chats today?" → categories=["chats"]
        - "Which models are used most?" → categories=["models"]
        - "What's the spend?" → categories=["spend"]
        - "How many knowledge bases?" → categories=["knowledge"]
        - "Show me user and token stats" → categories=["users", "tokens"]
        - "Platform overview" → categories=None (all data)
        
        :param categories: Optional list of stat categories to fetch. Available:
            - "users" (user counts, roles, activity, registrations)
            - "chats" (chat totals, activity, archives, pinned)
            - "models" (model usage, top models by messages)
            - "tokens" (token usage totals, top users by tokens in/out)
            - "files" (file storage, types, uploads)
            - "groups" (group stats, top groups by members/chats/spend)
            - "feedback" (feedback counts and activity)
            - "spend" (LiteLLM costs, requires ENABLE_LITELLM_COSTS)
            - "knowledge" (knowledge bases, documents, top users)
            If None or empty, returns ALL categories (default behavior)
        :param __user__: User information
        :return: Dictionary with structured dashboard data (filtered or complete)
        """
        
        if not self.valves.ENABLE_DASHBOARD:
            return {"error": "Dashboard is currently disabled"}
        
        user_id = __user__.get("id") if __user__ else None
        user_role = __user__.get("role", "user") if __user__ else "user"
        user_name = __user__.get("name", "User") if __user__ else "User"
        
        if not user_id:
            return {"error": "Unable to retrieve user information"}
        
        # Normalize and validate categories
        valid_categories = {"users", "chats", "models", "tokens", "files", "groups", "feedback", "spend", "knowledge"}
        if categories:
            categories = [c.lower() for c in categories if c]
            invalid = set(categories) - valid_categories
            if invalid:
                return {"error": f"Invalid categories: {', '.join(invalid)}. Valid: {', '.join(sorted(valid_categories))}"}
            categories_set = set(categories)
        else:
            categories_set = None  # None means fetch all
        
        try:
            from open_webui.models.users import Users
            from open_webui.models.chats import Chats
            from open_webui.models.feedbacks import Feedbacks
            from open_webui.models.files import Files
            from open_webui.models.models import Models
            from open_webui.models.groups import Groups
            from open_webui.models.knowledge import Knowledges
            
            if user_role == "admin":
                _, data = await self._generate_admin_dashboard_html(
                    Users, Chats, Feedbacks, Files, Models, Groups, Knowledges, user_name,
                    categories=categories_set,
                    event_emitter=None  # No status updates for data-only queries
                )
            else:
                _, data = self._generate_user_dashboard_html(
                    Users, Chats, Feedbacks, user_id, user_name
                )
            
            return data
            
        except Exception as e:
            log.exception(f"Dashboard data generation error: {e}")
            return {"error": str(e)}

    async def _fetch_litellm_costs(self, user_ids: list[str]) -> dict:
        """Fetch cost data from LiteLLM for given user IDs.
        
        Tries two endpoints:
        1. /customer/info?end_user_id=<id> - for end-user tracking
        2. /user/info?user_id=<id> - for key-based user tracking
        
        Returns: {user_id: {"spend": float, "error": str|None}}
        """
        if not self.valves.ENABLE_LITELLM_COSTS:
            return {}
        
        if not self.valves.LITELLM_API_KEY:
            log.warning("[DASHBOARD] LiteLLM costs enabled but no API key configured")
            return {}
        
        # Decrypt API key for use
        decrypted_key = EncryptedStr.decrypt(self.valves.LITELLM_API_KEY)
        
        costs = {}
        headers = {"Authorization": f"Bearer {decrypted_key}"}
        
        async with aiohttp.ClientSession() as session:
            for user_id in user_ids:
                try:
                    # Try /customer/info first (for end_user_id tracking)
                    url = f"{self.valves.LITELLM_BASE_URL}/customer/info?end_user_id={user_id}"
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            spend_value = data.get("spend", 0.0)
                            log.info(f"[DASHBOARD] LiteLLM /customer/info for {user_id[:8]}...: spend={spend_value}, full_response={data}")
                            costs[user_id] = {
                                "spend": spend_value,
                                "error": None
                            }
                            continue
                        elif response.status == 500:
                            # 500 might mean user not found in customer tracking, try /user/info
                            try:
                                error_text = await response.text()
                                log.debug(f"[DASHBOARD] /customer/info failed (500): {error_text[:200]}")
                            except:
                                pass
                        else:
                            error_text = await response.text()
                            log.warning(f"[DASHBOARD] /customer/info returned {response.status} for {user_id}: {error_text[:200]}")
                    
                    # Fallback: Try /user/info (for user_id on keys)
                    url = f"{self.valves.LITELLM_BASE_URL}/user/info?user_id={user_id}"
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            user_info = data.get("user_info", {})
                            spend_value = user_info.get("spend", 0.0) if isinstance(user_info, dict) else 0.0
                            log.info(f"[DASHBOARD] LiteLLM /user/info for {user_id[:8]}...: spend={spend_value}, full_response={data}")
                            costs[user_id] = {
                                "spend": spend_value,
                                "error": None
                            }
                        else:
                            error_text = await response.text()
                            log.warning(f"[DASHBOARD] Both LiteLLM endpoints failed for {user_id}. /user/info: {response.status}")
                            costs[user_id] = {"spend": 0.0, "error": f"Both endpoints failed (HTTP {response.status})"}
                            
                except asyncio.TimeoutError:
                    log.warning(f"[DASHBOARD] LiteLLM API timeout for user {user_id}")
                    costs[user_id] = {"spend": 0.0, "error": "timeout"}
                except Exception as e:
                    log.warning(f"[DASHBOARD] Failed to fetch LiteLLM cost for user {user_id}: {str(e)}")
                    costs[user_id] = {"spend": 0.0, "error": str(e)}
        
        return costs

    async def _generate_admin_dashboard_html(
        self, Users, Chats, Feedbacks, Files, Models, Groups, Knowledges, admin_name: str,
        categories=None,
        event_emitter=None
    ) -> tuple[str, dict]:
        """
        Generate interactive HTML dashboard for administrators. Returns (html, data) tuple.
        
        :param categories: Optional set of categories to include. If None, includes all.
                          Valid: users, chats, models, tokens, files, groups, feedback, spend, knowledge
        """
        
        log.info(f"[DASHBOARD] Generating admin dashboard, ENABLE_LITELLM_COSTS={self.valves.ENABLE_LITELLM_COSTS}, categories={categories}")
        
        # Detect data-only mode (no event emitter = called from get_dashboard_data)
        data_only_mode = event_emitter is None
        
        # Helper to emit status updates
        async def emit_status(description: str):
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {"description": description, "done": False}
                })
        
        # Helper to check if a category should be included
        def should_include(category: str) -> bool:
            return categories is None or category in categories
        
        # Always fetch users for basic stats (needed for other calculations)
        await emit_status("Fetching user data...")
        all_users_data = Users.get_users()
        all_users = all_users_data.get("users", [])
        total_users = all_users_data.get("total", 0)
        
        now = int(time.time())
        day_ago = now - 86400
        week_ago = now - 604800
        month_ago = now - 2592000
        
        # User-specific metrics (only if "users" category requested)
        if should_include("users"):
            # User registration metrics
            users_today = sum(1 for u in all_users if u.created_at >= day_ago)
            users_this_week = sum(1 for u in all_users if u.created_at >= week_ago)
            users_this_month = sum(1 for u in all_users if u.created_at >= month_ago)
            
            active_last_24h = sum(1 for u in all_users if u.last_active_at >= day_ago)
            active_last_week = sum(1 for u in all_users if u.last_active_at >= week_ago)
            active_last_month = sum(1 for u in all_users if u.last_active_at >= month_ago)
            active_all_time = sum(1 for u in all_users if u.last_active_at > 0)
            
            # Active now count
            active_now_count = 0
            if self.valves.SHOW_ACTIVE_USERS:
                try:
                    from open_webui.socket.main import get_active_user_ids
                    active_user_ids = get_active_user_ids()
                    active_now_count = len(active_user_ids)
                except:
                    pass
            
            # User roles
            admin_count = sum(1 for u in all_users if u.role == "admin")
            user_count = sum(1 for u in all_users if u.role == "user")
            pending_count = sum(1 for u in all_users if u.role == "pending")
            
            # Inactive users (zero chats)
            inactive_users = sum(1 for u in all_users if len(Chats.get_chats_by_user_id(u.id)) == 0)
        else:
            # Provide minimal defaults for HTML generation
            users_today = users_this_week = users_this_month = 0
            active_last_24h = active_last_week = active_last_month = active_all_time = 0
            active_now_count = admin_count = user_count = pending_count = 0
            inactive_users = 0
        
        # Chat statistics (fetch if chats, models, or tokens needed)
        if should_include("chats") or should_include("models") or should_include("tokens"):
            await emit_status("Analyzing chat history...")
            all_chats = Chats.get_chats()
            total_chats = len(all_chats)
            
            if should_include("chats"):
                total_chats_archived = sum(1 for c in all_chats if c.archived)
                total_chats_pinned = sum(1 for c in all_chats if c.pinned)
                chat_activity_today = sum(1 for c in all_chats if c.updated_at >= day_ago)
                chat_activity_week = sum(1 for c in all_chats if c.updated_at >= week_ago)
                chat_activity_month = sum(1 for c in all_chats if c.updated_at >= month_ago)
            else:
                total_chats_archived = total_chats_pinned = 0
                chat_activity_today = chat_activity_week = chat_activity_month = 0
        else:
            all_chats = []
            total_chats = total_chats_archived = total_chats_pinned = 0
            chat_activity_today = chat_activity_week = chat_activity_month = 0
        
        # Model usage and token tracking (only if models or tokens requested)
        model_usage = {}
        model_prompt_tokens = {}
        model_completion_tokens = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        if should_include("models") or should_include("tokens"):
            await emit_status("Processing model usage and tokens...")
            for chat in all_chats:
                chat_data = chat.chat if hasattr(chat, 'chat') else {}
                if isinstance(chat_data, dict):
                    messages = chat_data.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, dict):
                            model_id = msg.get("model") or msg.get("modelId")
                            if model_id:
                                model_usage[model_id] = model_usage.get(model_id, 0) + 1
                            
                            # Extract token usage from message
                            if "usage" in msg and isinstance(msg["usage"], dict):
                                usage = msg["usage"]
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                
                                total_prompt_tokens += prompt_tokens
                                total_completion_tokens += completion_tokens
                                
                                # Track per model
                                if model_id:
                                    model_prompt_tokens[model_id] = model_prompt_tokens.get(model_id, 0) + prompt_tokens
                                    model_completion_tokens[model_id] = model_completion_tokens.get(model_id, 0) + completion_tokens
        
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        # Enhanced model metadata retrieval (only if models requested)
        model_metadata = {}
        model_order_list = []
        model_config_stats = {}
        
        if should_include("models"):
            await emit_status("Fetching model configuration...")
            
            # Get workspace models with metadata (includes custom models and base models)
            try:
                # Get ALL models from database (both custom and base models)
                all_db_models = Models.get_all_models()  # Returns all models regardless of base_model_id
                model_metadata = {m.id: m for m in all_db_models}
                
                # Build model config stats - use direct attribute access (Pydantic models, not dicts)
                hidden_models = [m for m in all_db_models if m.meta and getattr(m.meta, 'hidden', False)]
                visible_models = [m for m in all_db_models if not (m.meta and getattr(m.meta, 'hidden', False))]
                models_with_base = [m for m in all_db_models if m.base_model_id is not None]
                base_models = [m for m in all_db_models if m.base_model_id is None]
                
                model_config_stats = {
                    'total_workspace_models': len(all_db_models),
                    'hidden_models': len(hidden_models),
                    'visible_models': len(visible_models),
                    'custom_modelfiles': len(models_with_base),
                    'base_models': len(base_models),
                }
            except Exception as e:
                log.warning(f"[DASHBOARD] Failed to fetch workspace models: {e}")
            
            # Get model order from config (for reference only, not needed for admin analysis)
            try:
                from open_webui.config import get_config
                
                config_data = get_config()
                
                if config_data:
                    ui_config = config_data.get('ui', {})
                    model_order = ui_config.get('model_order_list')
                    if model_order:
                        model_order_list = model_order if isinstance(model_order, list) else []
            except Exception as e:
                log.warning(f"[DASHBOARD] Failed to fetch model config: {e}")
        
        # Build enhanced model lists with metadata
        all_models_enhanced = []
        for model_id, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
            model_meta = model_metadata.get(model_id)
            display_name = model_meta.name if model_meta else model_id
            # Use getattr for Pydantic model attributes
            is_hidden = getattr(model_meta.meta, 'hidden', False) if (model_meta and model_meta.meta) else False
            # is_workspace: model exists in database (user-created OR imported)
            is_workspace = model_id in model_metadata
            # is_modelfile: model is a custom modelfile wrapping another model (has base_model_id)
            is_modelfile = model_meta.base_model_id is not None if model_meta else False
            
            all_models_enhanced.append({
                'id': model_id,
                'name': display_name,
                'messages': count,
                'prompt_tokens': model_prompt_tokens.get(model_id, 0),
                'completion_tokens': model_completion_tokens.get(model_id, 0),
                'total_tokens': model_prompt_tokens.get(model_id, 0) + model_completion_tokens.get(model_id, 0),
                'is_workspace': is_workspace,
                'is_modelfile': is_modelfile,
                'is_hidden': is_hidden,
            })
        
        # Legacy format for backward compatibility (simple tuples)
        all_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)
        all_models_by_prompt_tokens = sorted(model_prompt_tokens.items(), key=lambda x: x[1], reverse=True)
        all_models_by_completion_tokens = sorted(model_completion_tokens.items(), key=lambda x: x[1], reverse=True)
        
        # Top N for charts (legacy format)
        top_models = all_models[:self.valves.TOP_MODELS_COUNT]
        top_models_by_prompt_tokens = all_models_by_prompt_tokens[:self.valves.TOP_MODELS_COUNT]
        top_models_by_completion_tokens = all_models_by_completion_tokens[:self.valves.TOP_MODELS_COUNT]
        
        # Top N enhanced with full metadata
        top_models_enhanced = all_models_enhanced[:self.valves.TOP_MODELS_COUNT]
        
        # Identify hidden models still being used (should not be accessible)
        hidden_models_in_use = []
        if should_include("models") and model_metadata:
            for model_data in all_models_enhanced:
                if model_data['is_hidden']:
                    hidden_models_in_use.append({
                        'id': model_data['id'],
                        'name': model_data['name'],
                        'messages': model_data['messages'],
                        'note': 'Hidden model still being used (from old chats or direct access)'
                    })
        
        # Count workspace vs external models in use
        if should_include("models") and model_metadata:
            model_config_stats['workspace_models_used'] = sum(1 for m in all_models_enhanced if m['is_workspace'])
            model_config_stats['external_models_used'] = sum(1 for m in all_models_enhanced if not m['is_workspace'])
            model_config_stats['modelfiles_used'] = sum(1 for m in all_models_enhanced if m['is_modelfile'])
            model_config_stats['hidden_models_in_use'] = len(hidden_models_in_use)
        
        # User activity with token tracking
        user_chat_data = []
        for user in all_users:
            chats = Chats.get_chats_by_user_id(user.id)
            if len(chats) > 0:
                # Calculate token usage for this user
                user_prompt_tokens = 0
                user_completion_tokens = 0
                
                for chat in chats:
                    chat_data = chat.chat if hasattr(chat, 'chat') else {}
                    if isinstance(chat_data, dict):
                        messages = chat_data.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, dict) and "usage" in msg:
                                usage = msg.get("usage", {})
                                if isinstance(usage, dict):
                                    user_prompt_tokens += usage.get("prompt_tokens", 0)
                                    user_completion_tokens += usage.get("completion_tokens", 0)
                
                user_chat_data.append({
                    'name': user.name,
                    'chats': len(chats),
                    'user_id': user.id,
                    'prompt_tokens': user_prompt_tokens,
                    'completion_tokens': user_completion_tokens
                })
        
        user_chat_data.sort(key=lambda x: x['chats'], reverse=True)
        
        # Full lists for data output
        all_users_by_chats = user_chat_data
        all_users_by_prompt_tokens = sorted(
            [u for u in user_chat_data if u['prompt_tokens'] > 0],
            key=lambda x: x['prompt_tokens'],
            reverse=True
        )
        all_users_by_completion_tokens = sorted(
            [u for u in user_chat_data if u['completion_tokens'] > 0],
            key=lambda x: x['completion_tokens'],
            reverse=True
        )
        all_users_by_total_tokens = sorted(
            [u for u in user_chat_data if (u['prompt_tokens'] + u['completion_tokens']) > 0],
            key=lambda x: x['prompt_tokens'] + x['completion_tokens'],
            reverse=True
        )
        
        # Top N for charts
        top_users = all_users_by_chats[:self.valves.TOP_USERS_COUNT]
        top_users_by_prompt_tokens = all_users_by_prompt_tokens[:self.valves.TOP_USERS_COUNT]
        top_users_by_completion_tokens = all_users_by_completion_tokens[:self.valves.TOP_USERS_COUNT]
        top_users_by_total_tokens = all_users_by_total_tokens[:self.valves.TOP_USERS_COUNT]
        
        # User token usage percentile breakdown
        # Build complete user token map including users with zero usage
        user_tokens_map = {}
        for user_data in user_chat_data:
            user_tokens_map[user_data['user_id']] = user_data['prompt_tokens'] + user_data['completion_tokens']
        
        # Add all users who don't have chats (zero usage)
        for user in all_users:
            if user.id not in user_tokens_map:
                user_tokens_map[user.id] = 0
        
        users_with_tokens = list(user_tokens_map.values())
        zero_use_users = sum(1 for tokens in users_with_tokens if tokens == 0)
        
        # Calculate percentiles for users with non-zero token usage
        active_users_tokens = [t for t in users_with_tokens if t > 0]
        
        if active_users_tokens:
            active_users_tokens.sort()
            n = len(active_users_tokens)
            
            # Calculate percentile values
            p50_idx = int(n * 0.50)
            p75_idx = int(n * 0.75)
            p90_idx = int(n * 0.90)
            
            p50_value = active_users_tokens[p50_idx] if p50_idx < n else 0
            p75_value = active_users_tokens[p75_idx] if p75_idx < n else 0
            p90_value = active_users_tokens[p90_idx] if p90_idx < n else 0
            
            # Count users in each bucket
            below_p50 = sum(1 for t in active_users_tokens if 0 < t < p50_value)
            p50_to_p75 = sum(1 for t in active_users_tokens if p50_value <= t < p75_value)
            p75_to_p90 = sum(1 for t in active_users_tokens if p75_value <= t < p90_value)
            above_p90 = sum(1 for t in active_users_tokens if t >= p90_value)
        else:
            p50_value = p75_value = p90_value = 0
            below_p50 = p50_to_p75 = p75_to_p90 = above_p90 = 0
        
        # Calculate top 10% concentration for tokens
        top_10_percent_token_share = 0.0
        if total_tokens > 0 and users_with_tokens:
            sorted_user_tokens = sorted(users_with_tokens, reverse=True)
            top_10_count = max(1, int(len(sorted_user_tokens) * 0.10))
            top_10_tokens = sum(sorted_user_tokens[:top_10_count])
            top_10_percent_token_share = (top_10_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
        
        # Calculate temporal context for zero-use users
        zero_use_user_ids = [uid for uid, tokens in user_tokens_map.items() if tokens == 0]
        zero_use_new_week = sum(1 for uid in zero_use_user_ids 
                                for u in all_users if u.id == uid and u.created_at >= week_ago)
        zero_use_new_today = sum(1 for uid in zero_use_user_ids 
                                 for u in all_users if u.id == uid and u.created_at >= day_ago)
        zero_use_established = zero_use_users - zero_use_new_week
        
        user_token_distribution = {
            'total_users': total_users,
            'zero_use': zero_use_users,
            'zero_use_new_week': zero_use_new_week,
            'zero_use_new_today': zero_use_new_today,
            'zero_use_established': zero_use_established,
            'below_50th': below_p50,
            'p50_to_p75': p50_to_p75,
            'p75_to_p90': p75_to_p90,
            'above_p90': above_p90,
            'percentiles': {
                '50th': p50_value,
                '75th': p75_value,
                '90th': p90_value
            },
            'top_10_percent_share': round(top_10_percent_token_share, 1)
        }
        
        # User chat count distribution
        user_chats_map = {}
        for user_data in user_chat_data:
            user_chats_map[user_data['user_id']] = user_data['chats']
        
        # Add all users who don't have chats (zero chats)
        for user in all_users:
            if user.id not in user_chats_map:
                user_chats_map[user.id] = 0
        
        users_chat_counts = list(user_chats_map.values())
        zero_chats_users = sum(1 for chats in users_chat_counts if chats == 0)
        
        # Calculate percentiles for users with non-zero chat counts
        active_users_chats = [c for c in users_chat_counts if c > 0]
        
        if active_users_chats:
            active_users_chats.sort()
            n_chat = len(active_users_chats)
            
            # Calculate percentile values
            p50_chat_idx = int(n_chat * 0.50)
            p75_chat_idx = int(n_chat * 0.75)
            p90_chat_idx = int(n_chat * 0.90)
            
            p50_chat_value = active_users_chats[p50_chat_idx] if p50_chat_idx < n_chat else 0
            p75_chat_value = active_users_chats[p75_chat_idx] if p75_chat_idx < n_chat else 0
            p90_chat_value = active_users_chats[p90_chat_idx] if p90_chat_idx < n_chat else 0
            
            # Count users in each bucket
            below_p50_chat = sum(1 for c in active_users_chats if 0 < c < p50_chat_value)
            p50_to_p75_chat = sum(1 for c in active_users_chats if p50_chat_value <= c < p75_chat_value)
            p75_to_p90_chat = sum(1 for c in active_users_chats if p75_chat_value <= c < p90_chat_value)
            above_p90_chat = sum(1 for c in active_users_chats if c >= p90_chat_value)
        else:
            p50_chat_value = p75_chat_value = p90_chat_value = 0
            below_p50_chat = p50_to_p75_chat = p75_to_p90_chat = above_p90_chat = 0
        
        # Calculate top 10% concentration for chats
        top_10_percent_chat_share = 0.0
        if total_chats > 0 and users_chat_counts:
            sorted_user_chats = sorted(users_chat_counts, reverse=True)
            top_10_count_chat = max(1, int(len(sorted_user_chats) * 0.10))
            top_10_chats = sum(sorted_user_chats[:top_10_count_chat])
            top_10_percent_chat_share = (top_10_chats / total_chats * 100) if total_chats > 0 else 0.0
        
        # Calculate temporal context for zero-chat users
        zero_chat_user_ids = [uid for uid, chats in user_chats_map.items() if chats == 0]
        zero_chat_new_week = sum(1 for uid in zero_chat_user_ids 
                                 for u in all_users if u.id == uid and u.created_at >= week_ago)
        zero_chat_new_today = sum(1 for uid in zero_chat_user_ids 
                                  for u in all_users if u.id == uid and u.created_at >= day_ago)
        zero_chat_established = zero_chats_users - zero_chat_new_week
        
        user_chat_distribution = {
            'total_users': total_users,
            'zero_chats': zero_chats_users,
            'zero_chats_new_week': zero_chat_new_week,
            'zero_chats_new_today': zero_chat_new_today,
            'zero_chats_established': zero_chat_established,
            'below_50th': below_p50_chat,
            'p50_to_p75': p50_to_p75_chat,
            'p75_to_p90': p75_to_p90_chat,
            'above_p90': above_p90_chat,
            'percentiles': {
                '50th': p50_chat_value,
                '75th': p75_chat_value,
                '90th': p90_chat_value
            },
            'top_10_percent_share': round(top_10_percent_chat_share, 1)
        }
        
        # File stats (only if files requested)
        if should_include("files"):
            await emit_status("Analyzing file storage...")
            all_files = Files.get_files()
            total_files = len(all_files)
            total_size_bytes = sum(f.meta.get("size", 0) if f.meta else 0 for f in all_files)
            total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
            files_today = sum(1 for f in all_files if f.created_at >= day_ago)
            files_this_week = sum(1 for f in all_files if f.created_at >= week_ago)
            
            # File type breakdown
            file_types = {}
            for f in all_files:
                content_type = f.meta.get("content_type", "unknown") if f.meta else "unknown"
                file_types[content_type] = file_types.get(content_type, 0) + 1
            top_file_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            total_files = total_size_bytes = files_today = files_this_week = 0
            total_size_gb = 0.0
            file_types = {}
            top_file_types = []
        
        # Group statistics (only if groups requested)
        if should_include("groups"):
            await emit_status("Processing group statistics...")
            all_groups = Groups.get_groups()
            total_groups = len(all_groups)
            total_group_members = sum(len(g.user_ids) if g.user_ids else 0 for g in all_groups)
        else:
            all_groups = []
            total_groups = total_group_members = 0
        
        # All groups by member count
        groups_with_members = [
            {"name": g.name, "members": len(g.user_ids) if g.user_ids else 0}
            for g in all_groups
        ]
        groups_with_members.sort(key=lambda x: x['members'], reverse=True)
        
        # All groups by total chats (sum of member chats)
        groups_with_chats = []
        for g in all_groups:
            if g.user_ids:
                group_chats = sum(
                    len(Chats.get_chats_by_user_id(user_id))
                    for user_id in g.user_ids
                )
                if group_chats > 0:
                    groups_with_chats.append({"name": g.name, "chats": group_chats})
        groups_with_chats.sort(key=lambda x: x['chats'], reverse=True)
        
        # Top N for charts
        top_groups = groups_with_members[:self.valves.TOP_USERS_COUNT]
        top_groups_by_chats = groups_with_chats[:self.valves.TOP_USERS_COUNT]
        
        # Feedback statistics (only if feedback requested)
        if should_include("feedback"):
            all_feedbacks = Feedbacks.get_all_feedbacks()
            total_feedbacks = len(all_feedbacks)
            feedbacks_today = sum(1 for f in all_feedbacks if f.created_at >= day_ago)
            feedbacks_this_week = sum(1 for f in all_feedbacks if f.created_at >= week_ago)
            
            # Rating breakdown (1-10)
            # Note: data['rating'] is for arena/leaderboard (win/loss indicator)
            # The actual 1-10 rating is in data['details']['rating']
            rating_counts = {str(i): 0 for i in range(1, 11)}
            for f in all_feedbacks:
                if f.data and isinstance(f.data, dict):
                    # Check for rating in details first (1-10 scale)
                    details = f.data.get('details')
                    if details and isinstance(details, dict):
                        rating = details.get('rating')
                        if rating is not None:
                            try:
                                rating_int = int(rating)
                                if 1 <= rating_int <= 10:
                                    rating_counts[str(rating_int)] += 1
                            except (ValueError, TypeError):
                                pass
            
            # Reason breakdown
            # Database stores reasons in snake_case, map to readable labels
            reason_map = {
                'accurate_information': 'Accurate information',
                'followed_instructions_perfectly': 'Followed instructions perfectly',
                'showcased_creativity': 'Showcased creativity',
                'positive_attitude': 'Positive attitude',
                'attention_to_detail': 'Attention to detail',
                'thorough_explanation': 'Thorough explanation',
                'other': 'Other'
            }
            reason_counts = {label: 0 for label in reason_map.values()}
            
            for f in all_feedbacks:
                if f.data and isinstance(f.data, dict):
                    reason = f.data.get('reason')
                    if reason:
                        # Map snake_case to readable label
                        readable_reason = reason_map.get(reason)
                        if readable_reason:
                            reason_counts[readable_reason] += 1
        else:
            total_feedbacks = feedbacks_today = feedbacks_this_week = 0
            rating_counts = {str(i): 0 for i in range(1, 11)}
            reason_counts = {
                'Accurate information': 0,
                'Followed instructions perfectly': 0,
                'Showcased creativity': 0,
                'Positive attitude': 0,
                'Attention to detail': 0,
                'Thorough explanation': 0,
                'Other': 0
            }
        
        # Knowledge base statistics (only if knowledge requested)
        if should_include("knowledge"):
            await emit_status("Analyzing knowledge bases...")
            all_knowledge_bases = Knowledges.get_knowledge_bases()
            total_knowledge_bases = len(all_knowledge_bases)
            knowledge_today = sum(1 for kb in all_knowledge_bases if kb.created_at >= day_ago)
            knowledge_this_week = sum(1 for kb in all_knowledge_bases if kb.created_at >= week_ago)
            
            # Count documents per user (from data.file_ids)
            user_kb_stats = {}
            for kb in all_knowledge_bases:
                user_id = kb.user_id
                if user_id not in user_kb_stats:
                    user_kb_stats[user_id] = {'knowledge_bases': 0, 'documents': 0, 'name': kb.user.name if kb.user else 'Unknown'}
                user_kb_stats[user_id]['knowledge_bases'] += 1
                
                # Count documents from file_ids in data
                if kb.data and isinstance(kb.data, dict):
                    file_ids = kb.data.get('file_ids', [])
                    if isinstance(file_ids, list):
                        user_kb_stats[user_id]['documents'] += len(file_ids)
            
            # Calculate totals
            total_kb_documents = sum(stats['documents'] for stats in user_kb_stats.values())
            
            # Top users by knowledge bases
            top_users_by_kb = sorted(
                [{'name': stats['name'], 'knowledge_bases': stats['knowledge_bases'], 'documents': stats['documents']}
                 for stats in user_kb_stats.values()],
                key=lambda x: x['knowledge_bases'],
                reverse=True
            )[:self.valves.TOP_USERS_COUNT]
            
            # Top users by documents
            top_users_by_kb_docs = sorted(
                [{'name': stats['name'], 'knowledge_bases': stats['knowledge_bases'], 'documents': stats['documents']}
                 for stats in user_kb_stats.values() if stats['documents'] > 0],
                key=lambda x: x['documents'],
                reverse=True
            )[:self.valves.TOP_USERS_COUNT]
        else:
            total_knowledge_bases = knowledge_today = knowledge_this_week = 0
            total_kb_documents = 0
            top_users_by_kb = []
            top_users_by_kb_docs = []
        
        # LiteLLM costs (only if spend requested)
        cost_data = {}
        total_platform_spend = 0.0
        users_with_spend = 0
        if self.valves.ENABLE_LITELLM_COSTS and should_include("spend"):
            # Fetch costs for all users to get accurate total spend
            all_user_ids = [u.id for u in all_users]
            await emit_status(f"Fetching LiteLLM spend data for {len(all_user_ids)} users...")
            log.info(f"[DASHBOARD] Fetching LiteLLM costs for {len(all_user_ids)} users")
            cost_data = await self._fetch_litellm_costs(all_user_ids)
            log.info(f"[DASHBOARD] Retrieved cost data for {len(cost_data)} users")
            
            # Calculate total platform spend
            for user_id, cost_info in cost_data.items():
                spend = cost_info.get('spend', 0.0)
                total_platform_spend += spend
                if spend > 0:
                    users_with_spend += 1
            
            log.info(f"[DASHBOARD] Platform spend: ${total_platform_spend:.2f} across {users_with_spend} users")
            
            # Attach spend to top users for chart (rounded to 2 decimal places)
            for user_data in top_users:
                user_data['spend'] = round(cost_data.get(user_data['user_id'], {}).get('spend', 0.0), 2)
        
        # Top groups by total spend (sum of member spend) - AFTER cost_data is fetched
        groups_with_spend = []
        top_groups_by_spend = []
        if self.valves.ENABLE_LITELLM_COSTS and cost_data:
            for g in all_groups:
                if g.user_ids:
                    group_spend = sum(
                        cost_data.get(user_id, {}).get('spend', 0.0)
                        for user_id in g.user_ids
                    )
                    if group_spend > 0:
                        groups_with_spend.append({"name": g.name, "spend": round(group_spend, 2)})
            groups_with_spend.sort(key=lambda x: x['spend'], reverse=True)
            top_groups_by_spend = groups_with_spend[:self.valves.TOP_USERS_COUNT]
        
        # Generate SVG charts
        await emit_status("Generating charts and visualizations...")
        user_roles_chart = self._create_donut_chart_svg(
            ['Admins', 'Users', 'Pending'],
            [admin_count, user_count, pending_count],
            ['#667eea', '#764ba2', '#f093fb']
        )
        
        user_activity_chart = self._create_bar_chart_svg(
            ['24h', 'Week', 'Month', 'All Time'],
            [active_last_24h, active_last_week, active_last_month, active_all_time],
            ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        )
        
        chat_activity_chart = self._create_bar_chart_svg(
            ['Today', 'Week', 'Month', 'All Time'],
            [chat_activity_today, chat_activity_week, chat_activity_month, total_chats],
            ['#764ba2', '#667eea', '#f093fb', '#4facfe']
        )
        
        # Use enhanced model data for charts (with display names)
        top_models_chart = self._create_horizontal_bar_chart_svg(
            [m['name'][:30] for m in top_models_enhanced] if top_models_enhanced else [m[0][:30] for m in top_models],
            [m['messages'] for m in top_models_enhanced] if top_models_enhanced else [m[1] for m in top_models],
            '#667eea'
        )
        
        # Top models by prompt tokens chart
        top_models_prompt_chart = ''
        if top_models_by_prompt_tokens:
            if top_models_enhanced:
                # Use enhanced data sorted by prompt tokens
                models_by_prompt = sorted(top_models_enhanced, key=lambda x: x['prompt_tokens'], reverse=True)[:self.valves.TOP_MODELS_COUNT]
                top_models_prompt_chart = self._create_horizontal_bar_chart_svg(
                    [m['name'][:30] for m in models_by_prompt],
                    [m['prompt_tokens'] for m in models_by_prompt],
                    '#f093fb'
                )
            else:
                top_models_prompt_chart = self._create_horizontal_bar_chart_svg(
                    [m[0][:30] for m in top_models_by_prompt_tokens],
                    [m[1] for m in top_models_by_prompt_tokens],
                    '#f093fb'
                )
        
        # Top models by completion tokens chart
        top_models_completion_chart = ''
        if top_models_by_completion_tokens:
            if top_models_enhanced:
                # Use enhanced data sorted by completion tokens
                models_by_completion = sorted(top_models_enhanced, key=lambda x: x['completion_tokens'], reverse=True)[:self.valves.TOP_MODELS_COUNT]
                top_models_completion_chart = self._create_horizontal_bar_chart_svg(
                    [m['name'][:30] for m in models_by_completion],
                    [m['completion_tokens'] for m in models_by_completion],
                    '#4facfe'
                )
            else:
                top_models_completion_chart = self._create_horizontal_bar_chart_svg(
                    [m[0][:30] for m in top_models_by_completion_tokens],
                    [m[1] for m in top_models_by_completion_tokens],
                    '#4facfe'
                )
        
        top_users_chart = self._create_horizontal_bar_chart_svg(
            [u['name'][:20] for u in top_users],
            [u['chats'] for u in top_users],
            '#764ba2'
        )
        
        top_spend_chart = ''
        if self.valves.ENABLE_LITELLM_COSTS and top_users:
            top_spend_chart = self._create_horizontal_bar_chart_svg(
                [u['name'][:20] for u in top_users],
                [round(u.get('spend', 0.0), 2) for u in top_users],
                '#f093fb'
            )
        
        # Top Users by Tokens charts
        top_users_by_tokens_in_chart = ''
        if top_users_by_prompt_tokens:
            top_users_by_tokens_in_chart = self._create_horizontal_bar_chart_svg(
                [u['name'][:20] for u in top_users_by_prompt_tokens],
                [u['prompt_tokens'] for u in top_users_by_prompt_tokens],
                '#4facfe'
            )
        
        top_users_by_tokens_out_chart = ''
        if top_users_by_completion_tokens:
            top_users_by_tokens_out_chart = self._create_horizontal_bar_chart_svg(
                [u['name'][:20] for u in top_users_by_completion_tokens],
                [u['completion_tokens'] for u in top_users_by_completion_tokens],
                '#f093fb'
            )
        
        top_users_by_total_tokens_chart = ''
        if top_users_by_total_tokens:
            top_users_by_total_tokens_chart = self._create_horizontal_bar_chart_svg(
                [u['name'][:20] for u in top_users_by_total_tokens],
                [u['prompt_tokens'] + u['completion_tokens'] for u in top_users_by_total_tokens],
                '#764ba2'
            )
        
        # Knowledge base charts
        top_users_by_kb_chart = ''
        if top_users_by_kb:
            top_users_by_kb_chart = self._create_horizontal_bar_chart_svg(
                [u['name'][:20] for u in top_users_by_kb],
                [u['knowledge_bases'] for u in top_users_by_kb],
                '#667eea'
            )
        
        top_users_by_kb_docs_chart = ''
        if top_users_by_kb_docs:
            top_users_by_kb_docs_chart = self._create_horizontal_bar_chart_svg(
                [u['name'][:20] for u in top_users_by_kb_docs],
                [u['documents'] for u in top_users_by_kb_docs],
                '#4facfe'
            )
        
        # Top Groups chart
        top_groups_chart = ''
        if top_groups:
            top_groups_chart = self._create_horizontal_bar_chart_svg(
                [g['name'][:25] for g in top_groups],
                [g['members'] for g in top_groups],
                '#667eea'
            )
        
        # Top Groups by Chats chart
        top_groups_by_chats_chart = ''
        if top_groups_by_chats:
            top_groups_by_chats_chart = self._create_horizontal_bar_chart_svg(
                [g['name'][:25] for g in top_groups_by_chats],
                [g['chats'] for g in top_groups_by_chats],
                '#4facfe'
            )
        
        # Top Groups by Spend chart
        top_groups_by_spend_chart = ''
        if self.valves.ENABLE_LITELLM_COSTS and top_groups_by_spend:
            top_groups_by_spend_chart = self._create_horizontal_bar_chart_svg(
                [g['name'][:25] for g in top_groups_by_spend],
                [g['spend'] for g in top_groups_by_spend],
                '#f093fb'
            )
        
        # Top File Types chart
        top_file_types_chart = ''
        if top_file_types:
            top_file_types_chart = self._create_horizontal_bar_chart_svg(
                [ft[0][:30] for ft in top_file_types],
                [ft[1] for ft in top_file_types],
                '#764ba2'
            )
        
        # Feedback rating breakdown chart (1-10)
        feedback_rating_chart = ''
        if total_feedbacks > 0:
            # Shorter, cleaner labels
            rating_labels = [str(i) for i in range(1, 11)]
            rating_values = [rating_counts[str(i)] for i in range(1, 11)]
            if sum(rating_values) > 0:
                feedback_rating_chart = self._create_bar_chart_svg(
                    rating_labels,
                    rating_values,
                    ['#ff6b6b', '#ff6b6b', '#ff8c42', '#ffa600', '#ffd93d', 
                     '#6bcf7f', '#6bcf7f', '#4caf50', '#2e7d32', '#1b5e20']
                )
        
        # Feedback reason breakdown chart
        feedback_reason_chart = ''
        if total_feedbacks > 0:
            # Filter out reasons with 0 count for cleaner chart
            reasons_with_counts = [(reason, count) for reason, count in reason_counts.items() if count > 0]
            if reasons_with_counts:
                reasons_with_counts.sort(key=lambda x: x[1], reverse=True)
                feedback_reason_chart = self._create_horizontal_bar_chart_svg(
                    [r[0] for r in reasons_with_counts],
                    [r[1] for r in reasons_with_counts],
                    '#667eea'
                )
        
        # User token distribution chart
        user_distribution_chart = ''
        if total_users > 0:
            zero_label = f"Zero use ({user_token_distribution['zero_use']}"
            if user_token_distribution['zero_use_new_today'] > 0:
                zero_label += f", {user_token_distribution['zero_use_new_today']} new today"
            elif user_token_distribution['zero_use_new_week'] > 0:
                zero_label += f", {user_token_distribution['zero_use_new_week']} new this week"
            zero_label += ")"
            
            dist_labels = [
                zero_label,
                f"Below 50th %ile",
                f"50th-75th %ile",
                f"75th-90th %ile",
                f"Above 90th %ile"
            ]
            dist_values = [
                user_token_distribution['zero_use'],
                user_token_distribution['below_50th'],
                user_token_distribution['p50_to_p75'],
                user_token_distribution['p75_to_p90'],
                user_token_distribution['above_p90']
            ]
            dist_colors = ['#ff6b6b', '#ff8c42', '#ffd33d', '#6bcf7f', '#43a047']
            percentiles_data = {
                '50th': user_token_distribution.get('p50_value', 0),
                '75th': user_token_distribution.get('p75_value', 0),
                '90th': user_token_distribution.get('p90_value', 0)
            }
            if sum(dist_values) > 0:
                user_distribution_chart = self._create_distribution_chart(
                    'User Token Distribution',
                    dist_labels,
                    dist_values,
                    dist_colors,
                    percentiles_data,
                    user_token_distribution['top_10_percent_share']
                )
        
        # User chat distribution chart
        user_chat_dist_chart = ''
        if total_users > 0:
            zero_chat_label = f"Zero chats ({user_chat_distribution['zero_chats']}"
            if user_chat_distribution['zero_chats_new_today'] > 0:
                zero_chat_label += f", {user_chat_distribution['zero_chats_new_today']} new today"
            elif user_chat_distribution['zero_chats_new_week'] > 0:
                zero_chat_label += f", {user_chat_distribution['zero_chats_new_week']} new this week"
            zero_chat_label += ")"
            
            chat_dist_labels = [
                zero_chat_label,
                f"Below 50th %ile",
                f"50th-75th %ile",
                f"75th-90th %ile",
                f"Above 90th %ile"
            ]
            chat_dist_values = [
                user_chat_distribution['zero_chats'],
                user_chat_distribution['below_50th'],
                user_chat_distribution['p50_to_p75'],
                user_chat_distribution['p75_to_p90'],
                user_chat_distribution['above_p90']
            ]
            chat_dist_colors = ['#ff6b6b', '#ff8c42', '#ffd33d', '#6bcf7f', '#43a047']
            chat_percentiles_data = {
                '50th': user_chat_distribution.get('p50_value', 0),
                '75th': user_chat_distribution.get('p75_value', 0),
                '90th': user_chat_distribution.get('p90_value', 0)
            }
            if sum(chat_dist_values) > 0:
                user_chat_dist_chart = self._create_distribution_chart(
                    'User Chat Distribution',
                    chat_dist_labels,
                    chat_dist_values,
                    chat_dist_colors,
                    chat_percentiles_data,
                    user_chat_distribution['top_10_percent_share']
                )
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard</title>
    <style>
        /* Light Theme (Default) */
        :root {{
            --bg-primary: #667eea;
            --bg-secondary: #764ba2;
            --card-bg: rgba(255, 255, 255, 0.95);
            --card-gradient-start: rgba(255, 255, 255, 0.98);
            --card-gradient-end: rgba(255, 255, 255, 0.95);
            --card-border: rgba(102, 126, 234, 0.1);
            --text-primary: #2d3748;
            --text-secondary: #666;
            --text-muted: #999;
            --accent: #667eea;
            --accent-glow: rgba(102, 126, 234, 0.4);
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            --shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.2);
            --radius: 16px;
            --radius-sm: 12px;
            --radial-1: rgba(118, 75, 162, 0.3);
            --radial-2: rgba(102, 126, 234, 0.3);
        }}
        
        /* Dark Theme */
        [data-theme="dark"] {{
            --bg-primary: #0b1020;
            --bg-secondary: #14204a;
            --card-bg: rgba(17, 26, 51, 0.95);
            --card-gradient-start: rgba(17, 26, 51, 0.98);
            --card-gradient-end: rgba(17, 26, 51, 0.95);
            --card-border: rgba(35, 48, 88, 0.6);
            --text-primary: #e7ecff;
            --text-secondary: #9fb2ffcc;
            --text-muted: #9fb2ff99;
            --accent: #82a0ff;
            --accent-glow: rgba(130, 160, 255, 0.5);
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            --shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.5);
            --radial-1: rgba(20, 32, 74, 0.8);
            --radial-2: rgba(27, 42, 99, 0.7);
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif;
            background: radial-gradient(1200px 700px at 80% -10%, var(--radial-1) 0%, transparent 60%),
                        radial-gradient(1000px 600px at -10% 0%, var(--radial-2) 0%, transparent 55%),
                        linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            padding: 20px;
            min-height: 100vh;
            transition: background 0.3s ease;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        }}
        .header {{
            background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
            border: 1px solid var(--card-border);
            border-radius: var(--radius);
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }}
        .header h1 {{
            color: var(--accent);
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .header p {{
            color: var(--text-secondary);
            font-size: 1.1em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 25px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: var(--shadow-hover);
        }}
        .stat-card h3 {{
            color: var(--text-secondary);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .stat-card .value {{
            color: var(--accent);
            font-size: 2.5em;
            font-weight: 700;
        }}
        .stat-card .label {{
            color: var(--text-muted);
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 25px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        .chart-card:hover {{
            transform: translateY(-5px);
            box-shadow: var(--shadow-hover);
        }}
        .chart-card h2 {{
            color: var(--text-primary);
            font-size: 1.3em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.15);
            font-weight: 600;
        }}
        .chart-container {{
            position: relative;
            min-height: 220px;
            display: flex;
            flex-direction: column;
            align-items: stretch;
            justify-content: flex-start;
            overflow: visible;
        }}
        .section-header {{
            background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
            border: 1px solid var(--card-border);
            border-radius: var(--radius-sm);
            padding: 20px 30px;
            margin: 30px 0 20px 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border-left: 5px solid var(--accent);
        }}
        .section-header h2 {{
            color: var(--accent);
            font-size: 1.8em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
        }}
        .section-header .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 12px var(--accent-glow);
            animation: pulse 2s ease-in-out infinite;
        }}
        .section-header p {{
            color: var(--text-secondary);
            margin: 8px 0 0 0;
            font-size: 0.95em;
        }}
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 50px;
            padding: 10px 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }}
        .theme-toggle:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }}
        .theme-toggle .icon {{
            font-size: 18px;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            .theme-toggle {{
                top: 10px;
                right: 10px;
                padding: 8px 16px;
                font-size: 12px;
            }}
        }}
    </style>
    <script>
        // Theme toggle functionality with localStorage persistence
        function initTheme() {{
            const savedTheme = localStorage.getItem('dashboard-theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateToggleButton(savedTheme);
        }}
        
        function toggleTheme() {{
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('dashboard-theme', newTheme);
            updateToggleButton(newTheme);
        }}
        
        function updateToggleButton(theme) {{
            const btn = document.getElementById('theme-toggle-btn');
            if (btn) {{
                if (theme === 'dark') {{
                    btn.innerHTML = '<span class="icon">☀️</span><span>Light Mode</span>';
                }} else {{
                    btn.innerHTML = '<span class="icon">🌙</span><span>Dark Mode</span>';
                }}
            }}
        }}
        
        // Initialize theme on page load
        document.addEventListener('DOMContentLoaded', initTheme);
    </script>
</head>
<body>
    <!-- Theme Toggle Button -->
    <button id="theme-toggle-btn" class="theme-toggle" onclick="toggleTheme()">
        <span class="icon">🌙</span>
        <span>Dark Mode</span>
    </button>
    
    <div class="container">
        <div class="header">
            <h1>📊 Admin Dashboard <span style="font-size: 14px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 12px; font-weight: 500; margin-left: 12px;">v2.2.0</span></h1>
            <p><strong>Administrator:</strong> {admin_name} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <!-- ADOPTION SECTION -->
        <div class="section-header">
            <h2><span class="dot"></span>👥 Adoption</h2>
            <p>User registration, growth, and platform reach</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Users</h3>
                <div class="value mono">{total_users}</div>
                <div class="label"><span class="mono">{active_last_24h}</span> active today{f', {active_now_count} online now' if active_now_count > 0 else ''}</div>
            </div>
            <div class="stat-card">
                <h3>New Users (Week)</h3>
                <div class="value mono">{users_this_week}</div>
                <div class="label"><span class="mono">{users_today}</span> registered today</div>
            </div>
            <div class="stat-card">
                <h3>Active Users (Week)</h3>
                <div class="value mono">{active_last_week}</div>
                <div class="label"><span class="mono">{active_last_month}</span> active this month</div>
            </div>
            <div class="stat-card">
                <h3>Inactive Users</h3>
                <div class="value mono">{inactive_users}</div>
                <div class="label">Zero chats created</div>
            </div>
        </div>

        <!-- USAGE SECTION -->
        <div class="section-header">
            <h2><span class="dot"></span>📊 Usage</h2>
            <p>Platform activity, chats, models, and tokens</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Chats</h3>
                <div class="value mono">{total_chats:,}</div>
                <div class="label"><span class="mono">{chat_activity_today}</span> updated today</div>
            </div>
            <div class="stat-card">
                <h3>Models Used</h3>
                <div class="value mono">{len(model_usage)}</div>
                <div class="label">Unique models in use</div>
            </div>
            <div class="stat-card">
                <h3>Total Tokens</h3>
                <div class="value mono">{total_tokens:,}</div>
                <div class="label">Input + Output combined</div>
            </div>
            <div class="stat-card">
                <h3>Input Tokens</h3>
                <div class="value mono">{total_prompt_tokens:,}</div>
                <div class="label">Prompts sent</div>
            </div>
            <div class="stat-card">
                <h3>Output Tokens</h3>
                <div class="value mono">{total_completion_tokens:,}</div>
                <div class="label">Completions received</div>
            </div>
        </div>
        
        {'<!-- COST SECTION --><div class="section-header"><h2><span class="dot"></span>💰 Cost</h2><p>LiteLLM spend tracking and financial metrics</p></div><div class="stats-grid"><div class="stat-card"><h3>Total Spend</h3><div class="value mono">${:.2f}</div><div class="label"><span class="mono">{}</span> users with spend</div></div><div class="stat-card"><h3>Avg Spend/User</h3><div class="value mono">${:.2f}</div><div class="label">Per active user</div></div></div>'.format(total_platform_spend, users_with_spend, total_platform_spend / users_with_spend if users_with_spend > 0 else 0.0) if self.valves.ENABLE_LITELLM_COSTS else ''}

        <!-- QUALITY SECTION -->
        <div class="section-header">
            <h2><span class="dot"></span>⭐ Quality</h2>
            <p>User feedback and satisfaction metrics</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Feedback</h3>
                <div class="value mono">{total_feedbacks}</div>
                <div class="label"><span class="mono">{feedbacks_today}</span> submitted today</div>
            </div>
            <div class="stat-card">
                <h3>This Week</h3>
                <div class="value mono">{feedbacks_this_week}</div>
                <div class="label">Recent feedback activity</div>
            </div>
        </div>

        <!-- CONTENT/KB SECTION -->
        <div class="section-header">
            <h2><span class="dot"></span>📚 Content & Knowledge</h2>
            <p>Files, storage, and knowledge base metrics</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>File Storage</h3>
                <div class="value mono">{total_size_gb:.2f} GB</div>
                <div class="label"><span class="mono">{total_files:,}</span> files total</div>
            </div>
            <div class="stat-card">
                <h3>Files Uploaded</h3>
                <div class="value mono">{files_this_week}</div>
                <div class="label">This week (<span class="mono">{files_today}</span> today)</div>
            </div>
            <div class="stat-card">
                <h3>File Types</h3>
                <div class="value mono">{len(file_types)}</div>
                <div class="label">Unique content types</div>
            </div>
            <div class="stat-card">
                <h3>Knowledge Bases</h3>
                <div class="value mono">{total_knowledge_bases}</div>
                <div class="label"><span class="mono">{knowledge_this_week}</span> created this week</div>
            </div>
            <div class="stat-card">
                <h3>KB Documents</h3>
                <div class="value mono">{total_kb_documents:,}</div>
                <div class="label">Total indexed documents</div>
            </div>
        </div>

        <!-- GOVERNANCE SECTION -->
        <div class="section-header">
            <h2><span class="dot"></span>🔐 Governance</h2>
            <p>Groups, user roles, and access control</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Groups</h3>
                <div class="value mono">{total_groups}</div>
                <div class="label"><span class="mono">{total_group_members}</span> total members</div>
            </div>
            <div class="stat-card">
                <h3>Admins</h3>
                <div class="value mono">{admin_count}</div>
                <div class="label">Administrator accounts</div>
            </div>
            <div class="stat-card">
                <h3>Regular Users</h3>
                <div class="value mono">{user_count}</div>
                <div class="label">Standard user accounts</div>
            </div>
            <div class="stat-card">
                <h3>Pending</h3>
                <div class="value mono">{pending_count}</div>
                <div class="label">Awaiting approval</div>
            </div>
        </div>

        <!-- ADOPTION CHARTS -->
        <div class="section-header">
            <h2><span class="dot"></span>👥 Adoption Analytics</h2>
        </div>
        <div class="charts-grid">
            <div class="chart-card">
                <h2>👥 User Roles Distribution</h2>
                <div class="chart-container">
                    {user_roles_chart}
                </div>
            </div>
            
            <div class="chart-card">
                <h2>📊 Active Users Timeline</h2>
                <div class="chart-container">
                    {user_activity_chart}
                </div>
            </div>
            
            {'<div class="chart-card"><h2>📊 User Token Distribution</h2><div class="chart-container">' + user_distribution_chart + '</div></div>' if user_distribution_chart else ''}
            
            {'<div class="chart-card"><h2>💬 User Chat Distribution</h2><div class="chart-container">' + user_chat_dist_chart + '</div></div>' if user_chat_dist_chart else ''}
        </div>

        <!-- USAGE CHARTS -->
        <div class="section-header">
            <h2><span class="dot"></span>📊 Usage Analytics</h2>
        </div>
        <div class="charts-grid">
            <div class="chart-card">
                <h2>💬 Chat Activity</h2>
                <div class="chart-container">
                    {chat_activity_chart}
                </div>
            </div>
            
            <div class="chart-card">
                <h2>🤖 Top Models by Messages</h2>
                <div class="chart-container">
                    {top_models_chart}
                </div>
            </div>
            
            {'<div class="chart-card"><h2>📥 Top Models by Input Tokens</h2><div class="chart-container">' + top_models_prompt_chart + '</div></div>' if top_models_prompt_chart else ''}
            
            {'<div class="chart-card"><h2>📤 Top Models by Output Tokens</h2><div class="chart-container">' + top_models_completion_chart + '</div></div>' if top_models_completion_chart else ''}
            
            <div class="chart-card">
                <h2>🏆 Top Users by Chats</h2>
                <div class="chart-container">
                    {top_users_chart}
                </div>
            </div>
            
            {'<div class="chart-card"><h2>� Top Users by Tokens In</h2><div class="chart-container">' + top_users_by_tokens_in_chart + '</div></div>' if top_users_by_tokens_in_chart else ''}
            
            {'<div class="chart-card"><h2>📤 Top Users by Tokens Out</h2><div class="chart-container">' + top_users_by_tokens_out_chart + '</div></div>' if top_users_by_tokens_out_chart else ''}
            
            {'<div class="chart-card"><h2>🔄 Top Users by Total Tokens</h2><div class="chart-container">' + top_users_by_total_tokens_chart + '</div></div>' if top_users_by_total_tokens_chart else ''}
            
        </div>

        {'<!-- COST CHARTS --><div class="section-header"><h2><span class="dot"></span>💰 Cost Analytics</h2></div><div class="charts-grid"><div class="chart-card"><h2>💰 Top Users by Spend</h2><div class="chart-container">' + top_spend_chart + '</div></div>' + ('<div class="chart-card"><h2>💰 Top Groups by Spend</h2><div class="chart-container">' + top_groups_by_spend_chart + '</div></div>' if top_groups_by_spend_chart else '') + '</div>' if self.valves.ENABLE_LITELLM_COSTS and top_spend_chart else ''}

        <!-- QUALITY CHARTS -->
        {'<div class="section-header"><h2><span class="dot"></span>⭐ Quality Analytics</h2></div><div class="charts-grid">' + ('<div class="chart-card"><h2>⭐ Feedback Ratings (1-10)</h2><div class="chart-container">' + feedback_rating_chart + '</div></div>' if feedback_rating_chart else '') + ('<div class="chart-card"><h2>� Feedback Reasons</h2><div class="chart-container">' + feedback_reason_chart + '</div></div>' if feedback_reason_chart else '') + '</div>' if feedback_rating_chart or feedback_reason_chart else ''}

        <!-- CONTENT/KB CHARTS -->
        <div class="section-header">
            <h2><span class="dot"></span>📚 Content & Knowledge Analytics</h2>
        </div>
        <div class="charts-grid">
            {'<div class="chart-card"><h2>📁 Top File Types</h2><div class="chart-container">' + top_file_types_chart + '</div></div>' if top_file_types_chart else ''}
            
            {'<div class="chart-card"><h2>📚 Top Users by Knowledge Bases</h2><div class="chart-container">' + top_users_by_kb_chart + '</div></div>' if top_users_by_kb_chart else ''}
            
            {'<div class="chart-card"><h2>📄 Top Users by KB Documents</h2><div class="chart-container">' + top_users_by_kb_docs_chart + '</div></div>' if top_users_by_kb_docs_chart else ''}
        </div>

        <!-- GOVERNANCE CHARTS -->
        <div class="section-header">
            <h2><span class="dot"></span>🔐 Governance Analytics</h2>
        </div>
        <div class="charts-grid">
            {'<div class="chart-card"><h2>� Top Groups by Members</h2><div class="chart-container">' + top_groups_chart + '</div></div>' if top_groups_chart else ''}
            
            {'<div class="chart-card"><h2>💬 Top Groups by Chats</h2><div class="chart-container">' + top_groups_by_chats_chart + '</div></div>' if top_groups_by_chats_chart else ''}
        </div>
    </div>
</body>
</html>
"""
        
        # Prepare structured data for LLM reasoning (filtered by categories)
        data = {
            "timestamp": datetime.now().isoformat(),
            "admin_name": admin_name,
        }
        
        if should_include("users"):
            data["user_stats"] = {
                "total_users": total_users,
                "inactive_users": inactive_users,
                "new_users_today": users_today,
                "new_users_this_week": users_this_week,
                "new_users_this_month": users_this_month,
                "active_last_24h": active_last_24h,
                "active_last_week": active_last_week,
                "active_last_month": active_last_month,
                "active_all_time": active_all_time,
                "active_now": active_now_count,
                "roles": {
                    "admin": admin_count,
                    "user": user_count,
                    "pending": pending_count
                }
            }
        
        if should_include("chats"):
            data["chat_stats"] = {
                "total_chats": total_chats,
                "archived": total_chats_archived,
                "pinned": total_chats_pinned,
                "activity_today": chat_activity_today,
                "activity_week": chat_activity_week,
                "activity_month": chat_activity_month,
                "activity_all_time": total_chats
            }
        
        if should_include("files"):
            data["file_stats"] = {
                "total_files": total_files,
                "files_today": files_today,
                "files_this_week": files_this_week,
                "total_size_bytes": total_size_bytes,
                "total_size_gb": round(total_size_gb, 2),
                "file_types": len(file_types),
                "top_file_types": [{"type": ft[0], "count": ft[1]} for ft in top_file_types]
            }
        
        if should_include("groups"):
            groups_members_list = groups_with_members if data_only_mode else top_groups
            groups_chats_list = groups_with_chats if data_only_mode else top_groups_by_chats
            groups_spend_list = groups_with_spend if data_only_mode else top_groups_by_spend
            
            data["group_stats"] = {
                "total_groups": total_groups,
                "total_members": total_group_members,
                "top_groups_by_members": [{"name": g['name'], "members": g['members']} for g in groups_members_list],
                "top_groups_by_chats": [{"name": g['name'], "chats": g['chats']} for g in groups_chats_list],
                "top_groups_by_spend": [{"name": g['name'], "spend": g['spend']} for g in groups_spend_list] if self.valves.ENABLE_LITELLM_COSTS else []
            }
        
        if should_include("feedback"):
            data["feedback_stats"] = {
                "total_feedbacks": total_feedbacks,
                "feedbacks_today": feedbacks_today,
                "feedbacks_this_week": feedbacks_this_week,
                "rating_breakdown": rating_counts,
                "reason_breakdown": reason_counts
            }
        
        # Always include user token distribution as it's core user analytics
        data["user_token_distribution"] = user_token_distribution
        data["user_chat_distribution"] = user_chat_distribution
        
        if should_include("knowledge"):
            users_kb_list = all_users_by_kb if data_only_mode else top_users_by_kb
            users_kb_docs_list = all_users_by_kb_docs if data_only_mode else top_users_by_kb_docs
            
            data["knowledge_stats"] = {
                "total_knowledge_bases": total_knowledge_bases,
                "knowledge_today": knowledge_today,
                "knowledge_this_week": knowledge_this_week,
                "total_documents": total_kb_documents,
                "top_users_by_kb": [
                    {"name": u['name'], "knowledge_bases": u['knowledge_bases'], "documents": u['documents']}
                    for u in users_kb_list
                ],
                "top_users_by_documents": [
                    {"name": u['name'], "knowledge_bases": u['knowledge_bases'], "documents": u['documents']}
                    for u in users_kb_docs_list
                ]
            }
        
        if should_include("models"):
            models_list = all_models if data_only_mode else top_models
            models_prompt_list = all_models_by_prompt_tokens if data_only_mode else top_models_by_prompt_tokens
            models_completion_list = all_models_by_completion_tokens if data_only_mode else top_models_by_completion_tokens
            models_enhanced_list = all_models_enhanced if data_only_mode else top_models_enhanced
            
            data["model_stats"] = {
                "unique_models": len(model_usage),
                # Legacy format for backward compatibility
                "top_models": [{"model": m[0], "messages": m[1]} for m in models_list],
                "top_models_by_prompt_tokens": [{"model": m[0], "prompt_tokens": m[1]} for m in models_prompt_list],
                "top_models_by_completion_tokens": [{"model": m[0], "completion_tokens": m[1]} for m in models_completion_list],
                # Enhanced format with full metadata
                "models_detailed": models_enhanced_list,
                "model_configuration": model_config_stats,
                "hidden_models_in_use": hidden_models_in_use if hidden_models_in_use else []
            }
        
        if should_include("tokens"):
            users_prompt_list = all_users_by_prompt_tokens if data_only_mode else top_users_by_prompt_tokens
            users_completion_list = all_users_by_completion_tokens if data_only_mode else top_users_by_completion_tokens
            users_total_list = all_users_by_total_tokens if data_only_mode else top_users_by_total_tokens
            
            data["token_stats"] = {
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "top_users_by_tokens_in": [
                    {"name": u['name'], "prompt_tokens": u['prompt_tokens']}
                    for u in users_prompt_list
                ],
                "top_users_by_tokens_out": [
                    {"name": u['name'], "completion_tokens": u['completion_tokens']}
                    for u in users_completion_list
                ],
                "top_users_by_total_tokens": [
                    {"name": u['name'], "total_tokens": u['prompt_tokens'] + u['completion_tokens']}
                    for u in users_total_list
                ]
            }
        
        # Top users list (included if any user-related category requested)
        if should_include("users") or should_include("chats") or should_include("tokens"):
            users_list = all_users_by_chats if data_only_mode else top_users
            data["top_users"] = [
                {
                    "name": u['name'],
                    "chats": u['chats'],
                    "spend": round(u.get('spend', 0.0), 2) if self.valves.ENABLE_LITELLM_COSTS else None
                }
                for u in users_list
            ]
        
        # Add spend stats if LiteLLM is enabled (only if spend requested)
        if self.valves.ENABLE_LITELLM_COSTS and should_include("spend"):
            data["spend_stats"] = {
                "total_platform_spend": round(total_platform_spend, 2),
                "users_with_spend": users_with_spend,
                "average_spend_per_user": round(total_platform_spend / users_with_spend, 2) if users_with_spend > 0 else 0.0,
                "top_spenders": [
                    {"name": u['name'], "spend": round(u.get('spend', 0.0), 2)}
                    for u in sorted(top_users, key=lambda x: x.get('spend', 0.0), reverse=True)
                    if u.get('spend', 0.0) > 0
                ]
            }
        
        return html, data

    async def _generate_user_dashboard_html(
        self, Users, Chats, Feedbacks, user_id: str, user_name: str,
        event_emitter=None
    ) -> tuple[str, dict]:
        """Generate interactive HTML dashboard for regular users. Returns (html, data) tuple."""
        
        # Helper to emit status updates
        async def emit_status(description: str):
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {"description": description, "done": False}
                })
        
        await emit_status("Loading your profile...")
        user = Users.get_user_by_id(user_id)
        if not user:
            return ("<html><body><h1>Unable to retrieve user information</h1></body></html>", 
                    {"error": "User not found"})
        
        account_age_days = (int(time.time()) - user.created_at) // 86400
        
        await emit_status("Analyzing your chat history...")
        chats = Chats.get_chats_by_user_id(user_id)
        total_chats = len(chats)
        archived_chats = len([c for c in chats if c.archived])
        pinned_chats = len([c for c in chats if c.pinned])
        
        now = int(time.time())
        day_ago = now - 86400
        week_ago = now - 604800
        
        chats_today = sum(1 for c in chats if c.updated_at >= day_ago)
        chats_this_week = sum(1 for c in chats if c.updated_at >= week_ago)
        
        feedbacks = Feedbacks.get_feedbacks_by_user_id(user_id)
        total_feedbacks = len(feedbacks)
        
        # Generate SVG charts
        await emit_status("Creating your personal dashboard...")
        chat_status_chart = self._create_donut_chart_svg(
            ['Active', 'Pinned', 'Archived'],
            [total_chats - archived_chats - pinned_chats, pinned_chats, archived_chats],
            ['#667eea', '#764ba2', '#f093fb']
        )
        
        chat_activity_chart = self._create_bar_chart_svg(
            ['Today', 'This Week'],
            [chats_today, chats_this_week - chats_today],
            ['#667eea', '#764ba2']
        )
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-card h3 {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            color: #667eea;
            font-size: 2.5em;
            font-weight: bold;
        }}
        .stat-card .label {{
            color: #999;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .chart-card h2 {{
            color: #333;
            font-size: 1.3em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .chart-container {{
            position: relative;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Welcome back, {user_name}! <span style="font-size: 14px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 12px; font-weight: 500; margin-left: 12px;">v2.1.16</span></h1>
            <p><strong>Member since:</strong> {datetime.fromtimestamp(user.created_at).strftime('%Y-%m-%d')} ({account_age_days} days) | <strong>Role:</strong> {user.role.capitalize()}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Chats</h3>
                <div class="value">{total_chats}</div>
                <div class="label">{chats_today} today</div>
            </div>
            <div class="stat-card">
                <h3>Pinned Chats</h3>
                <div class="value">{pinned_chats}</div>
                <div class="label">Quick access</div>
            </div>
            <div class="stat-card">
                <h3>Archived Chats</h3>
                <div class="value">{archived_chats}</div>
                <div class="label">Organized</div>
            </div>
            <div class="stat-card">
                <h3>Feedbacks</h3>
                <div class="value">{total_feedbacks}</div>
                <div class="label">Submitted</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <h2>💬 Chat Status</h2>
                <div class="chart-container">
                    {chat_status_chart}
                </div>
            </div>
            
            <div class="chart-card">
                <h2>📊 Chat Updates</h2>
                <div class="chart-container">
                    {chat_activity_chart}
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        # Prepare structured data for LLM reasoning
        data = {
            "timestamp": datetime.now().isoformat(),
            "user_name": user_name,
            "user_id": user_id,
            "account_info": {
                "member_since": datetime.fromtimestamp(user.created_at).isoformat(),
                "account_age_days": account_age_days,
                "role": user.role,
                "email": user.email if hasattr(user, 'email') and user.email else None
            },
            "chat_stats": {
                "total_chats": total_chats,
                "pinned_chats": pinned_chats,
                "archived_chats": archived_chats,
                "active_chats": total_chats - archived_chats - pinned_chats,
                "chats_today": chats_today,
                "chats_this_week": chats_this_week
            },
            "feedback_stats": {
                "total_feedbacks": total_feedbacks
            }
        }
        
        return html, data