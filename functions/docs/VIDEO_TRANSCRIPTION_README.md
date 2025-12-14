# Video Transcription Pipeline

A comprehensive video transcription pipeline for Open WebUI that extracts audio from video files using ffmpeg and transcribes to text using OpenAI Whisper API or local Whisper models.

## Features

### 🎥 Video Support
- **Multiple formats**: MP4, MOV, WebM, MKV, AVI, FLV, WMV
- **Flexible input**: Upload files, provide URLs, or reference Open WebUI file IDs
- **Automatic video info detection**: Duration and metadata extraction

### 🎤 Audio Extraction
- **High-quality extraction**: Uses ffmpeg for reliable audio extraction
- **Configurable sample rate**: Default 16kHz (optimal for Whisper)
- **Mono conversion**: Reduces file size while maintaining quality
- **Size validation**: Prevents oversized files (default 25MB limit)

### 🤖 Transcription Modes

#### 1. OpenAI API Mode (Default)
- Uses OpenAI's Whisper API (whisper-1 model)
- Fast and accurate
- Requires OpenAI API key
- 25MB file size limit

#### 2. Local Mode
- Runs Whisper models locally
- Models: tiny, base, small, medium, large
- No API costs or size limits
- Requires `openai-whisper` package

#### 3. OpenAI-Compatible Mode
- Works with custom endpoints (e.g., Azure, local servers)
- Same API format as OpenAI
- Flexible deployment options

## Configuration (Valves)

### Essential Settings

| Valve | Default | Description |
|-------|---------|-------------|
| `WHISPER_MODE` | `openai` | Mode: `openai`, `local`, or `openai-compatible` |
| `OPENAI_API_KEY` | - | OpenAI API key (encrypted, required for `openai` mode) |
| `WHISPER_MODEL` | `whisper-1` | Model: `whisper-1` (API) or `tiny/base/small/medium/large` (local) |
| `OUTPUT_FORMAT` | `text` | Format: `text`, `srt`, `vtt`, or `json` |
| `INCLUDE_TIMESTAMPS` | `true` | Include timestamps in output |

### Advanced Settings

| Valve | Default | Description |
|-------|---------|-------------|
| `WHISPER_LANGUAGE` | `None` | Language code (e.g., `en`, `es`, `fr`) or auto-detect |
| `WHISPER_TEMPERATURE` | `0.0` | Sampling temperature (0.0-1.0) |
| `MAX_AUDIO_SIZE_MB` | `25` | Maximum audio file size (MB) |
| `AUDIO_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `TIMEOUT` | `300` | Maximum transcription time (seconds) |
| `DEBUG` | `false` | Enable verbose debug logging |

## Output Formats

### 1. Text Format (Default)
```
[00:00 → 00:05] Hello and welcome to this video tutorial.
[00:05 → 00:12] Today we're going to learn about video transcription.
[00:12 → 00:18] Let's get started with the basics.
```

### 2. SRT Format (Subtitles)
```
1
00:00:00,000 --> 00:00:05,000
Hello and welcome to this video tutorial.

2
00:00:05,000 --> 00:00:12,000
Today we're going to learn about video transcription.
```

### 3. VTT Format (WebVTT)
```
WEBVTT

00:00:00.000 --> 00:00:05.000
Hello and welcome to this video tutorial.

00:00:05.000 --> 00:00:12.000
Today we're going to learn about video transcription.
```

### 4. JSON Format (Raw Data)
```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Hello and welcome to this video tutorial."
    }
  ],
  "language": "en"
}
```

## Usage Examples

### Basic Transcription
1. Upload a video file in the chat
2. Send a message (e.g., "Transcribe this video")
3. Wait for processing
4. Receive formatted transcription

### Custom Language
Set `WHISPER_LANGUAGE` to `es` for Spanish, `fr` for French, etc.

### Generate Subtitles
Set `OUTPUT_FORMAT` to `srt` or `vtt` for subtitle files

### Local Processing
1. Install: `pip install openai-whisper`
2. Set `WHISPER_MODE` to `local`
3. Set `WHISPER_MODEL` to `base` or `small`
4. Process videos locally (no API costs!)

## Technical Details

### Architecture
```
Video Input → Download → Extract Audio → Transcribe → Format → Save → Return
   ↓            ↓            ↓              ↓          ↓       ↓       ↓
 Message    Temp File    ffmpeg        Whisper    Text/SRT  Files DB Output
```

### Audio Processing Pipeline
1. **Download**: Fetch video from URL or Open WebUI storage
2. **Extract**: Use ffmpeg to extract audio track as WAV
3. **Convert**: Mono, 16kHz PCM format (optimal for Whisper)
4. **Validate**: Check file size limits
5. **Transcribe**: Send to Whisper API or local model
6. **Format**: Convert to requested output format
7. **Save**: Store transcription in Files DB

### Error Handling
- Video download failures
- Audio extraction errors
- File size validation
- API timeouts and rate limits
- Malformed responses
- Cleanup of temporary files

## Requirements

### Python Packages
```bash
# Core requirements
pip install aiohttp cryptography pydantic imageio-ffmpeg

# For local Whisper (optional)
pip install openai-whisper
```

### System Requirements
- ffmpeg (provided by imageio-ffmpeg)
- 500MB+ free disk space for temporary files
- For local mode: 1-5GB RAM depending on model size

## Performance

### OpenAI API Mode
- **Speed**: ~0.5x realtime (10 min video = ~5 min processing)
- **Cost**: ~$0.006 per minute of audio
- **Limit**: 25MB audio file size

### Local Mode
- **Speed**: Varies by model and hardware
  - Tiny: ~5x realtime (very fast)
  - Base: ~2x realtime
  - Small: ~1x realtime
  - Medium: ~0.5x realtime
  - Large: ~0.2x realtime
- **Cost**: Free (one-time model download)
- **Limit**: System RAM only

## Troubleshooting

### "Audio file too large"
- Reduce `MAX_AUDIO_SIZE_MB` if using local mode
- Use shorter video clips
- Use local mode for unlimited size

### "Transcription failed"
- Check API key is valid
- Verify network connectivity
- Check audio extraction succeeded
- Review debug logs (`DEBUG=true`)

### "Failed to extract audio"
- Ensure video file is valid
- Check video codec compatibility
- Verify ffmpeg is working

### Local mode not working
- Install: `pip install openai-whisper`
- First run downloads model (~100MB-3GB)
- Requires sufficient RAM for model

## Integration Examples

### Subtitle Generation Workflow
```python
# 1. Set valves
WHISPER_MODE = "openai"
OUTPUT_FORMAT = "srt"
INCLUDE_TIMESTAMPS = True

# 2. Upload video
# 3. Get SRT file
# 4. Use with video player
```

### Multi-Language Support
```python
# Auto-detect (default)
WHISPER_LANGUAGE = None

# Force Spanish
WHISPER_LANGUAGE = "es"

# Force French
WHISPER_LANGUAGE = "fr"
```

### Cost Optimization
```python
# Use local mode for free processing
WHISPER_MODE = "local"
WHISPER_MODEL = "base"  # Fast and accurate enough

# Or use tiny for very fast processing
WHISPER_MODEL = "tiny"
```

## Comparison with video_to_sfx.py

| Feature | video_to_sfx.py | video_transcription.py |
|---------|-----------------|------------------------|
| Purpose | Generate sound effects | Transcribe speech to text |
| Input | Video frames + instructions | Video with audio |
| Output | Audio file (MP3/WAV) | Text/SRT/VTT/JSON |
| AI Model | GPT-4o + ElevenLabs | OpenAI Whisper |
| Processing | Vision → Prompt → Audio | Video → Audio → Text |
| Use Case | Add SFX to videos | Create subtitles, transcripts |

## Future Enhancements

- [ ] Speaker diarization (who said what)
- [ ] Word-level timestamps
- [ ] Translation support
- [ ] Batch processing multiple videos
- [ ] Progress tracking for long videos
- [ ] Direct video player integration
- [ ] Custom vocabulary/prompt support

## License

MIT License - Same as video_to_sfx.py

## Author

Created for Open WebUI community
Based on video_to_sfx.py architecture
