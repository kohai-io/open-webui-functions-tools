# Open WebUI Pipes & Functions: File Access Pattern

## The Problem: Files Disappearing from Body

When building pipes or functions that need to access uploaded files, you may encounter this issue:

```python
async def pipe(self, body: dict, __user__: dict = None):
    files = body.get("files")  # None
    files = body.get("metadata", {}).get("files")  # Also None
    # Where did the files go? 🤔
```

**Why This Happens:**
- Tool handlers with `"file_handler": true` metadata delete files from `body["metadata"]["files"]` after processing
- This prevents downstream pipes from seeing the uploaded files
- Pipes fall back to unreliable workarounds like timestamp-based file detection

## The Solution: Use `__files__` Parameter

Open WebUI passes files through `extra_params` which is NOT affected by tool handlers:

```python
async def pipe(
    self,
    body: dict,
    __user__: Optional[dict] = None,
    __event_emitter__: Optional[Callable] = None,
    __event_call__: Optional[Callable] = None,
    __files__: Optional[list] = None,  # ✅ Add this parameter
):
    # Now files are reliably available!
    if __files__:
        for file_item in __files__:
            file_id = file_item.get("id")
            # Process file...
```

### How It Works

1. **User uploads file** → File metadata stored in `form_data["metadata"]["files"]`
2. **Backend creates extra_params** → `extra_params["__files__"] = metadata.get("files", [])`
   - See: `backend/open_webui/functions.py` lines 255-268
3. **Tool handler processes** → May delete `body["metadata"]["files"]`
4. **Your pipe receives** → `__files__` parameter still has the file list!

## Implementation Guide

### Step 1: Add `__files__` Parameter

```python
async def pipe(
    self,
    body: dict,
    __user__: Optional[dict] = None,
    __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    __files__: Optional[list] = None,  # ADD THIS
) -> str:
```

### Step 2: Check Files in Priority Order

```python
# PRIORITY 1: Check __files__ parameter (always available)
files_to_process = None
if __files__:
    self.log.info(f"Found {len(__files__)} file(s) in __files__ parameter")
    files_to_process = __files__

# PRIORITY 2: Fallback to body["files"] (may be None after tool handlers)
elif "files" in body and body["files"]:
    self.log.info(f"Found {len(body['files'])} file(s) in body.files")
    files_to_process = body["files"]

# PRIORITY 3: Fallback to body["metadata"]["files"] (may be None after tool handlers)
elif "metadata" in body and body["metadata"].get("files"):
    files_to_process = body["metadata"]["files"]

if not files_to_process:
    return "No files found"
```

### Step 3: Process Files

```python
if files_to_process:
    for file_item in files_to_process:
        file_id = file_item.get("id")
        
        # Get full file record from database
        from open_webui.models.files import Files as FilesDB
        file_record = FilesDB.get_file_by_id(file_id)
        
        if file_record:
            content_type = file_record.meta.get("content_type", "")
            filename = file_record.meta.get("name", "")
            file_path = file_record.path
            
            # Process the file...
```

## File Item Structure

Each item in `__files__` contains:

```python
{
    "type": "file",
    "id": "76220979-7fe6-43ab-b6d1-70fd179cd33c",
    "url": "/api/v1/files/76220979-7fe6-43ab-b6d1-70fd179cd33c",
    "name": "example.mp4",
    "status": "uploaded",
    "size": 24379303,
    "file": {
        "id": "76220979-7fe6-43ab-b6d1-70fd179cd33c",
        "user_id": "25cad468-87a3-4320-aa15-407152154bdd",
        "filename": "example.mp4",
        "path": "/opt/open-webui/backend/data/uploads/76220979-7fe6-43ab-b6d1-70fd179cd33c_example.mp4",
        "meta": {
            "name": "example.mp4",
            "content_type": "video/mp4",
            "size": 24379303,
            "data": {}
        },
        "data": {"status": "completed"},
        "created_at": 1765120697,
        "updated_at": 1765120697,
        "status": True
    }
}
```

## Real-World Examples

### Example 1: Video Transcription (Speechmatics)

**File:** `video_transcription_speechmatics.py` (version 3.2+)

```python
async def pipe(
    self,
    body: dict,
    __user__: Optional[dict] = None,
    __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    __files__: Optional[list] = None,
) -> Generator[str, None, None] | str:
    
    # PRIORITY 1: Check __files__ parameter
    video_file_id = None
    if __files__:
        self.log.info(f"Checking __files__ parameter: {len(__files__)} files")
        for file_item in __files__:
            file_id = file_item.get("id")
            if file_id:
                file_record = FilesDB.get_file_by_id(file_id)
                if file_record and file_record.meta.get("content_type", "").startswith("video/"):
                    video_file_id = file_id
                    self.log.info(f"Found video in __files__: {file_id}")
                    break
    
    # PRIORITY 2-4: Fallbacks...
    if not video_file_id:
        video_file_id = await self._find_video_in_files(body)
    # etc...
```

**Result:**
- ✅ Reliable video detection even after tool handlers run
- ✅ No more timestamp-based fallback hacks
- ✅ Works correctly with multiple files uploaded

### Example 2: Image Processing

```python
async def pipe(
    self,
    body: dict,
    __files__: Optional[list] = None,
) -> str:
    
    if not __files__:
        return "Please upload an image file"
    
    for file_item in __files__:
        file_id = file_item.get("id")
        file_record = FilesDB.get_file_by_id(file_id)
        
        if file_record:
            content_type = file_record.meta.get("content_type", "")
            
            if content_type.startswith("image/"):
                # Process image at file_record.path
                result = await self.process_image(file_record.path)
                return result
    
    return "No image files found"
```

## Common Available Parameters

Open WebUI pipes/functions can receive these parameters from `extra_params`:

```python
async def pipe(
    self,
    body: dict,                                      # The request body
    __user__: Optional[dict] = None,                 # User info (id, name, email, role)
    __event_emitter__: Optional[Callable] = None,    # Send status updates to UI
    __event_call__: Optional[Callable] = None,       # Call backend events
    __files__: Optional[list] = None,                # ✅ Uploaded files
    __messages__: Optional[list] = None,             # Message history
    __tools__: Optional[dict] = None,                # Available tools
    __task__: Optional[str] = None,                  # Task identifier
    __task_body__: Optional[dict] = None,            # Task data
    __chat_id__: Optional[str] = None,               # Chat session ID
    __session_id__: Optional[str] = None,            # Session ID
    __message_id__: Optional[str] = None,            # Message ID
    __metadata__: Optional[dict] = None,             # Request metadata
    __oauth_token__: Optional[str] = None,           # OAuth token if available
    __request__: Optional[Any] = None,               # FastAPI Request object
) -> str:
```

**Source:** `backend/open_webui/functions.py` lines 255-268

## Best Practices

### ✅ DO:
- **Always add `__files__` parameter** to pipes that need file access
- **Check `__files__` first** before checking body["files"] or body["metadata"]["files"]
- **Use FilesDB.get_file_by_id()** to get full file metadata and path
- **Check content_type** to filter for specific file types
- **Log when files are found** to help debugging

### ❌ DON'T:
- **Don't rely only on body["files"]** - it may be deleted by tool handlers
- **Don't use timestamp-based file detection** - unreliable when multiple files are uploaded
- **Don't assume files are in message content** - multimodal format is not guaranteed
- **Don't skip the __files__ parameter** - it's the most reliable source

## Debugging Tips

### Enable Debug Logging

```python
self.log.info(f"DEBUG: __files__ parameter = {__files__}")
self.log.info(f"DEBUG: 'files' in body = {'files' in body}")
self.log.info(f"DEBUG: body.get('files') = {body.get('files')}")
self.log.info(f"DEBUG: 'metadata' in body = {'metadata' in body}")
if 'metadata' in body:
    self.log.info(f"DEBUG: metadata.get('files') = {body['metadata'].get('files')}")
```

### Common Issues

**Issue:** `__files__` is None
- **Cause:** User didn't upload any files
- **Solution:** Check if files exist before processing

**Issue:** File ID exists but `FilesDB.get_file_by_id()` returns None
- **Cause:** File was deleted or never completed upload
- **Solution:** Check file status in file_item["status"]

**Issue:** File path doesn't exist
- **Cause:** File was uploaded but processing failed
- **Solution:** Check file_record.data["status"] == "completed"

## Related Code

- **Functions.py:** `backend/open_webui/functions.py` (lines 220-268)
  - Extracts files from metadata
  - Creates `extra_params` dict
  - Passes `__files__` to pipes/functions

- **Middleware.py:** `backend/open_webui/utils/middleware.py` (lines 490-514)
  - Tool handlers that delete files from body
  - `skip_files = True` when `file_handler: true`

- **Filter.py:** `backend/open_webui/utils/filter.py` (lines 129-134)
  - Inlet filters that remove files after processing

## Version History

- **v3.2 (2024-12-07):** Added `__files__` parameter support to video transcription pipes
- **v3.1 (2024-12-06):** Used timestamp-based file detection (unreliable)
- **v3.0 (2024-12-05):** Initial implementation with body["files"] only

## Contributing

When creating new pipes or functions that handle files:

1. Add `__files__` parameter to your pipe signature
2. Check `__files__` as highest priority
3. Add fallbacks for backward compatibility
4. Document file handling in your pipe's docstring
5. Include debug logging for troubleshooting

## Summary

**The Golden Rule:** Always use `__files__` parameter for reliable file access in pipes/functions.

```python
# ❌ BAD: Files may be None
files = body.get("metadata", {}).get("files")

# ✅ GOOD: Files are always available
async def pipe(self, body: dict, __files__: Optional[list] = None):
    if __files__:
        # Process files reliably
```

This pattern ensures your pipes work correctly even when tool handlers or inlet filters modify the request body.
