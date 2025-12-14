"""
title: Chat & Feedback Analytics Tool
author: _00_ (I∀I)
author_url: https://github.com/rgaricano
funding_url: https://github.com/open-webui
version: 0.1.5
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi.responses import HTMLResponse
from open_webui.config import WEBUI_URL


class Tools:
    def __init__(self):
        pass

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp to readable date"""
        if not timestamp:
            return "N/A"
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return "Invalid date"


    async def get_chat_feedback_analytics(
        self,
        __user__: dict,
        __event_emitter__: callable = None,
    ) -> HTMLResponse:
        """
        Display comprehensive analytics with chat titles, message counts, feedback data,
        sortable columns, and a separate feedback ranking table.
        """

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "Loading analytics...", "done": False}})

        try:
            # Import models for direct database access
            from open_webui.models.chats import Chats
            try:
                from open_webui.models.evaluations import Evaluations
            except ImportError:
                Evaluations = None

            user_id = __user__.get("id")
            user_role = __user__.get("role", "user")
            base_url = WEBUI_URL.value if WEBUI_URL.value else "http://localhost:8080"

            # Get chats based on user role
            if user_role == "admin":
                chats = Chats.get_chats()
            else:
                chats = Chats.get_chats_by_user_id(user_id)

            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Processing {len(chats)} chats...", "done": False}})

            # Get feedback data (if available)
            feedbacks = []
            if Evaluations:
                try:
                    feedbacks = Evaluations.get_evaluations()
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": f"Found {len(feedbacks)} feedback entries...", "done": False}})
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": f"Feedback access limited: {str(e)}", "done": False}})

            # Process data inline (no separate method)
            feedback_map = {}
            for feedback in feedbacks:
                chat_id = feedback.get("chat_id") if hasattr(feedback, 'get') else getattr(feedback, 'chat_id', None)
                if chat_id:
                    if chat_id not in feedback_map:
                        feedback_map[chat_id] = []
                    feedback_map[chat_id].append(feedback)

            # Process chat data
            chat_data = []
            for chat in chats:
                # Count messages efficiently
                message_count = 0
                if hasattr(chat, 'chat') and chat.chat and isinstance(chat.chat, dict):
                    messages = chat.chat.get("messages", [])
                    message_count = len(messages) if isinstance(messages, list) else 0

                # Get feedback for this chat
                chat_feedbacks = feedback_map.get(chat.id, [])
                feedback_count = len(chat_feedbacks)
                avg_rating = 0

                if chat_feedbacks:
                    ratings = []
                    for f in chat_feedbacks:
                        if hasattr(f, 'get'):
                            rating = f.get('data', {}).get('rating', 0)
                        else:
                            rating = getattr(f, 'data', {}).get('rating', 0) if hasattr(getattr(f, 'data', {}), 'get') else 0
                        if rating:
                            ratings.append(rating)
                    avg_rating = sum(ratings) / len(ratings) if ratings else 0

                chat_data.append({
                    'id': chat.id,
                    'title': chat.title or 'Untitled Chat',
                    'user_id': chat.user_id,
                    'message_count': message_count,
                    'feedback_count': feedback_count,
                    'avg_rating': avg_rating,
                    'created_at': chat.created_at,
                    'updated_at': chat.updated_at,
                    'chat_url': f"{base_url}/c/{chat.id}"
                })

            # Process feedback data for ranking table
            feedback_data = []
            for feedback in feedbacks:
                if hasattr(feedback, 'get'):
                    feedback_data.append({
                        'id': feedback.get('id'),
                        'chat_id': feedback.get('chat_id'),
                        'rating': feedback.get('data', {}).get('rating', 0),
                        'comment': feedback.get('data', {}).get('comment', ''),
                        'model_id': feedback.get('data', {}).get('model_id', ''),
                        'user_id': feedback.get('user_id'),
                        'updated_at': feedback.get('updated_at', 0)
                    })
                else:
                    data = getattr(feedback, 'data', {})
                    feedback_data.append({
                        'id': getattr(feedback, 'id', ''),
                        'chat_id': getattr(feedback, 'chat_id', ''),
                        'rating': data.get('rating', 0) if hasattr(data, 'get') else 0,
                        'comment': data.get('comment', '') if hasattr(data, 'get') else '',
                        'model_id': data.get('model_id', '') if hasattr(data, 'get') else '',
                        'user_id': getattr(feedback, 'user_id', ''),
                        'updated_at': getattr(feedback, 'updated_at', 0)
                    })

            # Generate complete HTML visualization inline
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chat & Feedback Analytics</title>
                <meta charset="utf-8">
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: #f8f9fa;
                    }}
                    .dashboard {{
                        max-width: 100%;
                        background: white;
                        border-radius: 12px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                        overflow: hidden;
                        margin-bottom: 20px;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 24px;
                        text-align: center;
                    }}
                    .stats {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 16px;
                        padding: 24px;
                        background: #f8f9fa;
                    }}
                    .stat-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        text-align: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .stat-number {{
                        font-size: 2em;
                        font-weight: bold;
                        color: #667eea;
                        margin-bottom: 8px;
                    }}
                    .table-container {{
                        overflow-x: auto;
                        max-height: 600px;
                    }}
                    .analytics-table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    .analytics-table th {{
                        background: #f8f9fa;
                        padding: 16px 12px;
                        text-align: left;
                        font-weight: 600;
                        position: sticky;
                        top: 0;
                        cursor: pointer;
                        user-select: none;
                        border-bottom: 2px solid #dee2e6;
                    }}
                    .analytics-table th:hover {{
                        background: #e9ecef;
                    }}
                    .analytics-table td {{
                        padding: 12px;
                        border-bottom: 1px solid #dee2e6;
                        vertical-align: middle;
                    }}
                    .analytics-table tr:hover {{
                        background-color: #f8f9fa;
                    }}
                    .chat-link {{
                        color: #667eea;
                        text-decoration: none;
                        font-weight: 500;
                        max-width: 250px;
                        display: block;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    }}
                    .chat-link:hover {{
                        text-decoration: underline;
                    }}
                    .badge {{
                        display: inline-block;
                        padding: 4px 8px;
                        border-radius: 12px;
                        font-size: 0.8em;
                        font-weight: 500;
                    }}
                    .badge-messages {{
                        background: #e3f2fd;
                        color: #1976d2;
                    }}
                    .badge-feedback {{
                        background: #f3e5f5;
                        color: #7b1fa2;
                    }}
                    .rating-stars {{
                        color: #ffc107;
                        font-size: 1.1em;
                    }}
                    .timestamp {{
                        font-size: 0.85em;
                        color: #6c757d;
                    }}
                    .user-info {{
                        font-size: 0.9em;
                        color: #6c757d;
                        font-family: monospace;
                    }}
                    .section-title {{
                        font-size: 1.3em;
                        font-weight: 600;
                        margin: 20px 0 10px 0;
                        color: #495057;
                    }}
                    .sort-indicator {{
                        margin-left: 5px;
                        font-size: 0.8em;
                    }}
                    .feedback-comment {{
                        max-width: 200px;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                        font-size: 0.9em;
                    }}
                </style>
                <script>
                    function sortTable(tableId, columnIndex, dataType = 'string') {{
                        const table = document.getElementById(tableId);
                        const tbody = table.querySelector('tbody');
                        const rows = Array.from(tbody.querySelectorAll('tr'));

                        const header = table.querySelectorAll('th')[columnIndex];
                        const currentDirection = header.dataset.sortDirection || 'asc';
                        const newDirection = currentDirection === 'asc' ? 'desc' : 'asc';

                        table.querySelectorAll('th').forEach(th => {{
                            th.dataset.sortDirection = '';
                            const indicator = th.querySelector('.sort-indicator');
                            if (indicator) indicator.textContent = '';
                        }});

                        header.dataset.sortDirection = newDirection;
                        const indicator = header.querySelector('.sort-indicator') || document.createElement('span');
                        indicator.className = 'sort-indicator';
                        indicator.textContent = newDirection === 'asc' ? '↑' : '↓';
                        if (!header.querySelector('.sort-indicator')) {{
                            header.appendChild(indicator);
                        }}

                        rows.sort((a, b) => {{
                            let aVal = a.cells[columnIndex].textContent.trim();
                            let bVal = b.cells[columnIndex].textContent.trim();

                            if (dataType === 'number') {{
                                aVal = parseFloat(aVal) || 0;
                                bVal = parseFloat(bVal) || 0;
                            }} else if (dataType === 'date') {{
                                aVal = new Date(aVal).getTime() || 0;
                                bVal = new Date(bVal).getTime() || 0;
                            }}

                            if (newDirection === 'asc') {{
                                return aVal > bVal ? 1 : -1;
                            }} else {{
                                return aVal < bVal ? 1 : -1;
                            }}
                        }});

                        rows.forEach(row => tbody.appendChild(row));
                    }}
                </script>
            </head>
            <body>
                <div class="dashboard">
                    <div class="header">
                        <h1>📊 Chat & Feedback Analytics</h1>
                        <p>Access Level: {user_role.title()} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>

                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">{len(chat_data)}</div>
                            <div class="stat-label">Total Conversations</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{sum(c['message_count'] for c in chat_data)}</div>
                            <div class="stat-label">Total Messages</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{len(feedback_data)}</div>
                            <div class="stat-label">Total Feedback</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{len([c for c in chat_data if c['feedback_count'] > 0])}</div>
                            <div class="stat-label">Chats with Feedback</div>
                        </div>
                    </div>

                    <h2 class="section-title">📋 Chat Overview</h2>
                    <div class="table-container"> 
                        <table class="analytics-table" id="chatTable">
                            <thead>
                                <tr>
                                    <th onclick="sortTable('chatTable', 0)">Chat Title <span class="sort-indicator"></span></th>
                                    <th onclick="sortTable('chatTable', 1, 'number')">Messages <span class="sort-indicator"></span></th>
                                    <th onclick="sortTable('chatTable', 2, 'number')">Feedback Count <span class="sort-indicator"></span></th>
                                    <th onclick="sortTable('chatTable', 3, 'number')">Avg Rating <span class="sort-indicator"></span></th>
                                    <th onclick="sortTable('chatTable', 4)">User <span class="sort-indicator"></span></th>
                                    <th onclick="sortTable('chatTable', 5, 'date')">Last Updated <span class="sort-indicator"></span></th>
                                </tr>
                            </thead>
                            <tbody>
            """

            for chat in chat_data:
                rating_display = ""
                if chat['avg_rating'] > 0:
                    stars = "★" * int(chat['avg_rating']) + "☆" * (5 - int(chat['avg_rating']))
                    rating_display = f'<span class="rating-stars">{stars}</span> ({chat["avg_rating"]:.1f})'
                else:
                    rating_display = '<span style="color: #6c757d;">No ratings</span>'

                html_content += f"""
                                <tr>
                                    <td>
                                        <a href="{chat['chat_url']}" class="chat-link" target="_blank" title="{chat['title']}">
                                            {chat['title']}
                                        </a>
                                    </td>
                                    <td>
                                        <span class="badge badge-messages">{chat['message_count']}</span>
                                    </td>
                                    <td>
                                        <span class="badge badge-feedback">{chat['feedback_count']}</span>
                                    </td>
                                    <td>{rating_display}</td>
                                    <td><span class="user-info">{chat['user_id']}</span></td>
                                    <td><span class="timestamp">{self._format_timestamp(chat['updated_at'])}</span></td>
                                </tr>
                """

                html_content += """
                                </tbody>
                            </table>
                        </div>

                        <h2 class="section-title">⭐ Feedback Rankings</h2>
                        <div class="table-container">
                            <table class="analytics-table" id="feedbackTable">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('feedbackTable', 0, 'number')">Rating <span class="sort-indicator"></span></th>
                                        <th onclick="sortTable('feedbackTable', 1)">Comment <span class="sort-indicator"></span></th>
                                        <th onclick="sortTable('feedbackTable', 2)">Model <span class="sort-indicator"></span></th>
                                        <th onclick="sortTable('feedbackTable', 3)">User <span class="sort-indicator"></span></th>
                                        <th onclick="sortTable('feedbackTable', 4)">Chat Title <span class="sort-indicator"></span></th>
                                        <th onclick="sortTable('feedbackTable', 5, 'date')">Date <span class="sort-indicator"></span></th>
                                    </tr>
                                </thead>
                                <tbody>
                """

                for feedback in feedback_data:
                    # Find the corresponding chat title
                    chat_title = "Unknown Chat"
                    chat_url = "#"
                    for chat in chat_data:
                        if chat['id'] == feedback['chat_id']:
                            chat_title = chat['title']
                            chat_url = chat['chat_url']
                            break

                    rating_stars = "★" * feedback['rating'] + "☆" * (5 - feedback['rating']) if feedback['rating'] > 0 else "No rating"
                    comment_preview = feedback['comment'][:50] + "..." if len(feedback['comment']) > 50 else feedback['comment']

                    html_content += f"""
                                    <tr>
                                        <td>
                                            <span class="rating-stars">{rating_stars}</span>
                                            <span class="badge" style="margin-left: 8px;">{feedback['rating']}</span>
                                        </td>
                                        <td>
                                            <div class="feedback-comment" title="{feedback['comment']}">
                                                {comment_preview if comment_preview else "No comment"}
                                            </div>
                                        </td>
                                        <td><span class="user-info">{feedback['model_id']}</span></td>
                                        <td><span class="user-info">{feedback['user_id']}</span></td>
                                        <td>
                                            <a href="{chat_url}" class="chat-link" target="_blank" title="{chat_title}">
                                                {chat_title[:30]}{"..." if len(chat_title) > 30 else ""}
                                            </a>
                                        </td>
                                        <td><span class="timestamp">{self._format_timestamp(feedback['updated_at'])}</span></td>
                                    </tr>
                    """

                html_content += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </body>
                </html>
                """

            return HTMLResponse(content=html_content, headers={"Content-Disposition": "inline"})
 
        except Exception as e:
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Analytics Error</title></head>
            <body style="font-family: system-ui; padding: 20px;">
                <div style="background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 8px;">
                    <h3>🚨 Error Loading Analytics</h3>
                    <p><strong>Error:</strong> {str(e)}</p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(
                content=error_html, headers={"Content-Disposition": "inline"}
            )