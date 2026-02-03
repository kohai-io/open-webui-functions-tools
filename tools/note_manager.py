"""
title: Note Manager
author: open-webui
version: 1.3.0
description: Allows models to read, create, update, and append to Open WebUI notes. Enables AI-driven note management during conversations.
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

    def _save_version(self, note) -> list:
        """
        Save current note content to version history before making changes.
        Returns the updated versions list.
        """
        versions = []
        if note.data and isinstance(note.data, dict):
            versions = note.data.get("versions", []) or []
            content_obj = note.data.get("content", {})
            if isinstance(content_obj, dict):
                current_version = {
                    "json": content_obj.get("json"),
                    "html": content_obj.get("html", ""),
                    "md": content_obj.get("md", "")
                }
                # Only add if there's actual content and it's different from last version
                if current_version.get("md") or current_version.get("html"):
                    last_version = versions[-1] if versions else None
                    if not last_version or last_version.get("md") != current_version.get("md"):
                        versions.append(current_version)
        return versions

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

    async def create_note(
        self,
        title: str,
        content: str = "",
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Create a new note with the given title and optional content.
        
        This function requires ALLOW_CREATE to be enabled in the tool's valves/settings.
        
        :param title: The title for the new note
        :param content: Optional initial content for the note (markdown format)
        :return: Success message with the new note ID, or error message
        """
        try:
            # Check if creation is allowed
            if not self.valves.ALLOW_CREATE:
                return "❌ Note creation is disabled. Enable ALLOW_CREATE in the tool settings to allow models to create notes."
            
            from open_webui.models.notes import Notes, NoteForm
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Creating note...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available. Cannot create note."
            
            if not title or len(title.strip()) == 0:
                return "❌ Title cannot be empty."
            
            if len(title) > 200:
                return "❌ Title too long. Maximum 200 characters."
            
            if len(content) > self.valves.MAX_CONTENT_LENGTH:
                return f"❌ Content too long. Maximum {self.valves.MAX_CONTENT_LENGTH} characters allowed."
            
            # Convert markdown to TipTap HTML
            html_content = self._markdown_to_tiptap_html(content) if content else ""
            
            # Create the note with proper data structure
            form_data = NoteForm(
                title=title.strip(),
                data={
                    "content": {
                        "md": content,
                        "html": html_content,
                        "json": None
                    },
                    "versions": [],
                    "files": None
                },
                meta={},
                access_control={},
            )
            
            new_note = Notes.insert_new_note(form_data, user_id)
            
            if not new_note:
                return "❌ Failed to create note."
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Note created successfully", "done": True}
                })
            
            log.info(f"[NOTE MANAGER] Created note {new_note.id} '{title}' for user {user_id}")
            
            return f"✅ Note **{title}** created successfully.\n\n**Note ID:** `{new_note.id}`\n\nYou can now use this ID to update or append to the note."
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error creating note: {e}")
            return f"❌ Error creating note: {str(e)}"

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
            
            # Save current content to version history before updating
            versions = self._save_version(note)
            
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
                    },
                    "versions": versions
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
            
            # Save current content to version history before updating
            versions = self._save_version(note)
            
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
                    },
                    "versions": versions
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

    async def delete_note(
        self,
        note_id: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Delete a note permanently.
        
        This function requires ALLOW_DELETE to be enabled in the tool's valves/settings.
        WARNING: This action cannot be undone!
        
        :param note_id: The ID of the note to delete
        :return: Success or error message
        """
        try:
            # Check if deletion is allowed
            if not self.valves.ALLOW_DELETE:
                return "❌ Note deletion is disabled. Enable ALLOW_DELETE in the tool settings to allow models to delete notes."
            
            from open_webui.models.notes import Notes
            from open_webui.utils.access_control import has_access
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Deleting note...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            user_role = __user__.get("role", "user") if __user__ else "user"
            
            note = Notes.get_note_by_id(note_id)
            if not note:
                return f"❌ Note not found: {note_id}"
            
            # Check write access (only owner or admin can delete)
            if user_role != "admin" and user_id != note.user_id:
                return "❌ You don't have permission to delete this note. Only the owner can delete it."
            
            title = note.title
            Notes.delete_note_by_id(note_id)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Note deleted", "done": True}
                })
            
            log.info(f"[NOTE MANAGER] Note {note_id} '{title}' deleted by user {user_id}")
            
            return f"✅ Note **{title}** deleted successfully."
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error deleting note: {e}")
            return f"❌ Error deleting note: {str(e)}"

    async def search_notes(
        self,
        query: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Search notes by title.
        
        :param query: Search query to match against note titles
        :return: List of matching notes with their IDs
        """
        try:
            from open_webui.models.notes import Notes
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Searching notes...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            # Get all accessible notes and filter by query
            notes = Notes.get_notes_by_permission(user_id, "read")
            
            query_lower = query.lower()
            matching_notes = [
                note for note in notes 
                if query_lower in note.title.lower()
            ]
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Found {len(matching_notes)} matching notes", "done": True}
                })
            
            if not matching_notes:
                return f"📝 No notes found matching '{query}'."
            
            result = f"## 🔍 Notes matching '{query}'\n\n"
            result += "| Title | ID | Updated |\n"
            result += "|-------|----|---------|\n"
            
            for note in matching_notes[:20]:  # Limit to 20 results
                updated = time.strftime('%Y-%m-%d', time.localtime(note.updated_at / 1000000000))
                title = note.title[:40] + "..." if len(note.title) > 40 else note.title
                result += f"| {title} | `{note.id}` | {updated} |\n"
            
            if len(matching_notes) > 20:
                result += f"\n*...and {len(matching_notes) - 20} more matches*"
            
            return result
            
        except Exception as e:
            log.error(f"[NOTE MANAGER] Error searching notes: {e}")
            return f"❌ Error searching notes: {str(e)}"

    async def help(
        self,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Show available Note Manager capabilities and usage examples.
        
        :return: Help text describing all available functions
        """
        create_status = "✅ Enabled" if self.valves.ALLOW_CREATE else "❌ Disabled"
        delete_status = "✅ Enabled" if self.valves.ALLOW_DELETE else "❌ Disabled"
        
        return f"""## 📝 Note Manager - Available Functions

### Reading Notes
| Function | Description |
|----------|-------------|
| `list_my_notes()` | List all notes you have access to |
| `get_note(note_id)` | Read the content of a specific note |
| `search_notes(query)` | Search notes by title |

### Creating Notes
| Function | Description | Status |
|----------|-------------|--------|
| `create_note(title, content)` | Create a new note | {create_status} |

### Updating Notes
| Function | Description |
|----------|-------------|
| `update_note(note_id, new_content)` | Replace entire note content |
| `append_to_note(note_id, content, add_timestamp)` | Add content to end of note |
| `update_note_title(note_id, new_title)` | Change a note's title |

### Deleting Notes
| Function | Description | Status |
|----------|-------------|--------|
| `delete_note(note_id)` | Permanently delete a note | {delete_status} |

---

### 💡 Usage Examples

**List your notes:**
> "Show me my notes" or "List my notes"

**Read a note:**
> "What's in my Todo note?" or "Read note abc-123"

**Create a note:**
> "Create a new note called 'Meeting Notes' with a task list"

**Update a note:**
> "Add 'Buy groceries' to my Todo note"
> "Mark the first item on my Todo as complete"

**Search notes:**
> "Find notes about meetings"

---

### ⚙️ Settings
- **ALLOW_CREATE:** {create_status} - Controls whether new notes can be created
- **ALLOW_DELETE:** {delete_status} - Controls whether notes can be deleted
- **MAX_CONTENT_LENGTH:** {self.valves.MAX_CONTENT_LENGTH:,} characters

*Settings can be changed in Workspace → Tools → Note Manager → Valves*
"""
