"""
title: Note Manager
author: open-webui
version: 1.0.6
description: Allows models to read, update, and append to Open WebUI notes. Enables AI-driven note management during conversations.
required_open_webui_version: 0.3.9
"""

from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import time

log = logging.getLogger(__name__)


class Tools:
    class Valves(BaseModel):
        ALLOW_CREATE: bool = Field(
            default=False,
            description="Allow models to create new notes (disabled by default for safety)"
        )
        ALLOW_DELETE: bool = Field(
            default=False,
            description="Allow models to delete notes (disabled by default for safety)"
        )
        MAX_CONTENT_LENGTH: int = Field(
            default=100000,
            description="Maximum content length for note updates (characters)"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _markdown_to_tiptap_html(self, md_content: str) -> str:
        """
        Convert markdown to TipTap-compatible HTML.
        Handles task lists with proper data attributes for TipTap TaskList extension.
        """
        import re
        
        lines = md_content.split('\n')
        html_parts = []
        in_task_list = False
        in_bullet_list = False
        
        for line in lines:
            stripped = line.strip()
            
            # Task list item: - [ ] or - [x]
            task_match = re.match(r'^[-*]\s*\[([ xX])\]\s*(.*)$', stripped)
            if task_match:
                if not in_task_list:
                    if in_bullet_list:
                        html_parts.append('</ul>')
                        in_bullet_list = False
                    html_parts.append('<ul data-type="taskList">')
                    in_task_list = True
                
                checked = task_match.group(1).lower() == 'x'
                text = task_match.group(2)
                checked_attr = 'true' if checked else 'false'
                html_parts.append(f'<li data-type="taskItem" data-checked="{checked_attr}"><label><input type="checkbox" {"checked" if checked else ""}><span></span></label><div><p>{text}</p></div></li>')
                continue
            
            # Regular bullet list item: - or *
            bullet_match = re.match(r'^[-*]\s+(.*)$', stripped)
            if bullet_match and not task_match:
                if not in_bullet_list:
                    if in_task_list:
                        html_parts.append('</ul>')
                        in_task_list = False
                    html_parts.append('<ul>')
                    in_bullet_list = True
                
                text = bullet_match.group(1)
                html_parts.append(f'<li><p>{text}</p></li>')
                continue
            
            # Close any open lists if we hit a non-list line
            if in_task_list:
                html_parts.append('</ul>')
                in_task_list = False
            if in_bullet_list:
                html_parts.append('</ul>')
                in_bullet_list = False
            
            # Empty line
            if not stripped:
                continue
            
            # Headings
            heading_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2)
                html_parts.append(f'<h{level}>{text}</h{level}>')
                continue
            
            # Regular paragraph
            html_parts.append(f'<p>{stripped}</p>')
        
        # Close any remaining open lists
        if in_task_list:
            html_parts.append('</ul>')
        if in_bullet_list:
            html_parts.append('</ul>')
        
        return ''.join(html_parts)

    async def get_note(
        self,
        note_id: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Get the content of a note by its ID.
        
        Use this to read the current content of a note before updating it.
        
        :param note_id: The ID of the note to retrieve
        :return: The note content or an error message
        """
        try:
            from open_webui.models.notes import Notes
            from open_webui.utils.access_control import has_access
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Reading note...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            user_role = __user__.get("role", "user") if __user__ else "user"
            
            note = Notes.get_note_by_id(note_id)
            if not note:
                return f"❌ Note not found: {note_id}"
            
            # Check access
            if user_role != "admin" and user_id != note.user_id:
                if not has_access(user_id, type="read", access_control=note.access_control):
                    return "❌ You don't have permission to read this note."
            
            # Extract content from note data
            # Notes use structure: data.content = {json, html, md}
            content = ""
            if note.data and isinstance(note.data, dict):
                content_obj = note.data.get("content", {})
                if isinstance(content_obj, dict):
                    # Prefer markdown, fallback to html
                    content = content_obj.get("md", "") or content_obj.get("html", "")
                elif isinstance(content_obj, str):
                    content = content_obj
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Note retrieved", "done": True}
                })
            
            return f"""## 📝 Note: {note.title}

**ID:** `{note.id}`
**Created:** {time.strftime('%Y-%m-%d %H:%M', time.localtime(note.created_at / 1000000000))}
**Updated:** {time.strftime('%Y-%m-%d %H:%M', time.localtime(note.updated_at / 1000000000))}

---

{content}
"""
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error getting note: {e}")
            return f"❌ Error reading note: {str(e)}"

    async def list_my_notes(
        self,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        List all notes accessible to the current user.
        
        Use this to find note IDs before reading or updating them.
        
        :return: A list of notes with their IDs and titles
        """
        try:
            from open_webui.models.notes import Notes
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Loading notes...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            notes = Notes.get_notes_by_permission(user_id, "write")
            
            if not notes:
                return "📝 You don't have any notes yet."
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Found {len(notes)} notes", "done": True}
                })
            
            result = "## 📝 Your Notes\n\n"
            result += "| Title | ID | Updated |\n"
            result += "|-------|----|---------|\n"
            
            for note in notes[:50]:  # Limit to 50 notes
                updated = time.strftime('%Y-%m-%d', time.localtime(note.updated_at / 1000000000))
                title = note.title[:40] + "..." if len(note.title) > 40 else note.title
                result += f"| {title} | `{note.id}` | {updated} |\n"
            
            if len(notes) > 50:
                result += f"\n*...and {len(notes) - 50} more notes*"
            
            return result
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error listing notes: {e}")
            return f"❌ Error listing notes: {str(e)}"

    async def update_note(
        self,
        note_id: str,
        new_content: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Replace the entire content of a note with new content.
        
        WARNING: This replaces ALL existing content. Use append_to_note() to add content instead.
        
        :param note_id: The ID of the note to update
        :param new_content: The new content to replace the existing content
        :return: Success or error message
        """
        try:
            from open_webui.models.notes import Notes, NoteUpdateForm
            from open_webui.utils.access_control import has_access
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Updating note...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            user_role = __user__.get("role", "user") if __user__ else "user"
            
            # Validate content length
            if len(new_content) > self.valves.MAX_CONTENT_LENGTH:
                return f"❌ Content too long. Maximum {self.valves.MAX_CONTENT_LENGTH} characters allowed."
            
            note = Notes.get_note_by_id(note_id)
            if not note:
                return f"❌ Note not found: {note_id}"
            
            # Check write access
            if user_role != "admin" and user_id != note.user_id:
                if not has_access(user_id, type="write", access_control=note.access_control):
                    return "❌ You don't have permission to update this note."
            
            # Update the note using NoteUpdateForm
            # Notes use a complex structure: data.content = {json, html, md}
            # The editor uses TipTap with TaskList extension
            html_content = self._markdown_to_tiptap_html(new_content)
            
            form_data = NoteUpdateForm(
                data={
                    "content": {
                        "md": new_content,
                        "html": html_content,
                        "json": None
                    }
                },
            )
            
            updated_note = Notes.update_note_by_id(note_id, form_data)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Note updated successfully", "done": True}
                })
            
            log.info(f"[NOTE MANAGER] Note {note_id} updated by user {user_id}")
            
            return f"✅ Note **{note.title}** updated successfully.\n\nNew content length: {len(new_content)} characters"
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error updating note: {e}")
            return f"❌ Error updating note: {str(e)}"

    async def append_to_note(
        self,
        note_id: str,
        content_to_add: str,
        add_timestamp: bool = True,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Append content to the end of an existing note.
        
        This is the PREFERRED method for adding content to notes - it preserves existing content.
        
        :param note_id: The ID of the note to append to
        :param content_to_add: The content to add at the end of the note
        :param add_timestamp: Whether to add a timestamp before the new content (default: True)
        :return: Success or error message
        """
        try:
            from open_webui.models.notes import Notes, NoteUpdateForm
            from open_webui.utils.access_control import has_access
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Appending to note...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            user_role = __user__.get("role", "user") if __user__ else "user"
            
            note = Notes.get_note_by_id(note_id)
            if not note:
                return f"❌ Note not found: {note_id}"
            
            # Check write access
            if user_role != "admin" and user_id != note.user_id:
                if not has_access(user_id, type="write", access_control=note.access_control):
                    return "❌ You don't have permission to update this note."
            
            # Get existing content
            # Notes use structure: data.content = {json, html, md}
            existing_content = ""
            if note.data and isinstance(note.data, dict):
                content_obj = note.data.get("content", {})
                if isinstance(content_obj, dict):
                    existing_content = content_obj.get("md", "") or content_obj.get("html", "")
                elif isinstance(content_obj, str):
                    existing_content = content_obj
            
            # Build new content
            if add_timestamp:
                timestamp = time.strftime('%Y-%m-%d %H:%M')
                new_entry = f"\n\n---\n\n**[{timestamp}]**\n\n{content_to_add}"
            else:
                new_entry = f"\n\n{content_to_add}"
            
            new_content = existing_content + new_entry
            
            # Validate content length
            if len(new_content) > self.valves.MAX_CONTENT_LENGTH:
                return f"❌ Note would exceed maximum length ({self.valves.MAX_CONTENT_LENGTH} characters). Current: {len(existing_content)}, Adding: {len(content_to_add)}"
            
            # Update the note using NoteUpdateForm
            # Notes use a complex structure: data.content = {json, html, md}
            # The editor uses TipTap with TaskList extension
            html_content = self._markdown_to_tiptap_html(new_content)
            
            form_data = NoteUpdateForm(
                data={
                    "content": {
                        "md": new_content,
                        "html": html_content,
                        "json": None
                    }
                },
            )
            
            updated_note = Notes.update_note_by_id(note_id, form_data)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Content appended to note", "done": True}
                })
            
            log.info(f"[NOTE MANAGER] Appended {len(content_to_add)} chars to note {note_id} by user {user_id}")
            
            return f"✅ Content appended to **{note.title}**.\n\nAdded: {len(content_to_add)} characters\nTotal length: {len(new_content)} characters"
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error appending to note: {e}")
            return f"❌ Error appending to note: {str(e)}"

    async def update_note_title(
        self,
        note_id: str,
        new_title: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Update the title of a note.
        
        :param note_id: The ID of the note to update
        :param new_title: The new title for the note
        :return: Success or error message
        """
        try:
            from open_webui.models.notes import Notes, NoteUpdateForm
            from open_webui.utils.access_control import has_access
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Updating note title...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            user_role = __user__.get("role", "user") if __user__ else "user"
            
            if not new_title or len(new_title.strip()) == 0:
                return "❌ Title cannot be empty."
            
            if len(new_title) > 200:
                return "❌ Title too long. Maximum 200 characters."
            
            note = Notes.get_note_by_id(note_id)
            if not note:
                return f"❌ Note not found: {note_id}"
            
            # Check write access
            if user_role != "admin" and user_id != note.user_id:
                if not has_access(user_id, type="write", access_control=note.access_control):
                    return "❌ You don't have permission to update this note."
            
            old_title = note.title
            
            # Update the note using NoteUpdateForm (only updates specified fields)
            form_data = NoteUpdateForm(
                title=new_title.strip(),
            )
            
            updated_note = Notes.update_note_by_id(note_id, form_data)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Note title updated", "done": True}
                })
            
            log.info(f"[NOTE MANAGER] Note {note_id} title changed from '{old_title}' to '{new_title}' by user {user_id}")
            
            return f"✅ Note title updated.\n\n**Old:** {old_title}\n**New:** {new_title}"
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error updating note title: {e}")
            return f"❌ Error updating note title: {str(e)}"
