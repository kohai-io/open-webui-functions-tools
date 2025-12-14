# Open WebUI Functions & Tools

A collection of custom functions and tools for [Open WebUI](https://github.com/open-webui/open-webui), extending its capabilities with integrations for video processing, AI services, productivity tools, and more.

## 📁 Repository Structure

```
functions_tools/
├── functions/          # Custom pipeline functions
│   ├── docs/          # Function documentation
│   └── *.py           # Function implementations
└── tools/             # Custom tool integrations
    └── *.py           # Tool implementations
```

## 🔧 Functions

Functions are pipeline components that process data and extend Open WebUI's functionality.

### Video & Media Processing

- **`gemini_video_understanding.py`** - Video analysis and understanding using Google Gemini
- **`veo_inline.py`** - Inline video generation with Google Veo
- **`video_to_sfx.py`** - Generate sound effects from video content
- **`video_transcription.py`** - Transcribe video audio to text
- **`video_transcription_speechmatics.py`** - Advanced transcription using Speechmatics API
- **`video_transcription_tool.py`** - Tool wrapper for video transcription
- **`video_transcription_tool_speechmatics.py`** - Tool wrapper for Speechmatics transcription
- **`video_transcription_subtitle_tool.py`** - Generate subtitles from video transcription

### Audio & Narration

- **`elevenlabs_narrator.py`** - Text-to-speech narration using ElevenLabs
- **`elevenlabs_podcast.py`** - Generate podcast-style audio content
- **`elevenlabs_srt_narrator.py`** - Narrate SRT subtitle files

### AI Model Integration

- **`nano_banana_pro.py`** - Integration with Nano Banana Pro AI model
- **`nano_banana_pro_chat.py`** - Chat interface for Nano Banana Pro
- **`nano_banana_otel.py`** - OpenTelemetry integration for Nano Banana

### Productivity & Workflow

- **`jira-embeded-ui.py`** - Embedded JIRA interface
- **`n8n-pipeline.py`** - Integration with n8n workflow automation

## 🛠️ Tools

Tools provide specific functionality that can be called by the LLM during conversations.

### Audio Generation

- **`ElevenLabsTTS.py`** - Text-to-speech conversion using ElevenLabs
- **`elevenlabs_music.py`** - Generate music with ElevenLabs
- **`elevenlabs_sfx.py`** - Generate sound effects with ElevenLabs

### Project Management

- **`open_webui_jira.py`** - JIRA integration for issue tracking and management
  - See [`JIRA_FEATURES.md`](tools/JIRA_FEATURES.md) for features
  - See [`JIRA_AUTH_TROUBLESHOOTING.md`](tools/JIRA_AUTH_TROUBLESHOOTING.md) for setup help

### AI Services

- **`nano_banana.py`** - Nano Banana AI model integration

### Utilities

- **`Hustory_Chats_and_feedbacks_tool.py`** - Chat history and feedback management

## 📚 Documentation

Additional documentation is available in the `functions/docs/` directory:

- **[C2PA Implementation Guide](functions/docs/C2PA_IMPLEMENTATION.md)** - Content authenticity and provenance
- **[C2PA Setup Guide](functions/docs/C2PA_SETUP_GUIDE.md)** - Setting up C2PA certificates
- **[C2PA SDK Test Certificates Guide](functions/docs/C2PA_SDK_TEST_CERTS_GUIDE.md)** - Testing with C2PA
- **[Video Transcription README](functions/docs/VIDEO_TRANSCRIPTION_README.md)** - Video transcription features
- **[Nano Banana Pro Prompt Guide](functions/docs/NANO_BANANA_PRO_PROMPT_GUIDE.md)** - Optimizing prompts
- **[Pipe Functions README](functions/docs/pipe-functions-readme.md)** - Understanding pipeline functions

## 🚀 Installation

### As a Submodule (Recommended for Development)

If you're working with the main Open WebUI repository:

```bash
cd open-webui
git submodule update --init --recursive
```

### Standalone Clone

```bash
git clone https://github.com/kohai-io/open-webui-functions-tools.git
```

## 💡 Usage

### Adding Functions to Open WebUI

1. Navigate to **Settings** → **Functions** in Open WebUI
2. Click **Import Function**
3. Paste the content of the desired `.py` file
4. Configure any required API keys or settings
5. Enable the function

### Adding Tools to Open WebUI

1. Navigate to **Settings** → **Tools** in Open WebUI
2. Click **Import Tool**
3. Paste the content of the desired `.py` file
4. Configure any required API keys or settings
5. Enable the tool

## 🔑 API Keys & Configuration

Many functions and tools require external API keys:

- **ElevenLabs**: Get API key from [elevenlabs.io](https://elevenlabs.io)
- **Google Gemini/Veo**: Configure through Google Cloud Console
- **Speechmatics**: Sign up at [speechmatics.com](https://www.speechmatics.com)
- **JIRA**: Configure with your Atlassian instance credentials
- **Nano Banana**: Contact provider for API access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This repository is part of the Open WebUI ecosystem. See individual files for specific licensing information.

## 🔗 Links

- [Open WebUI](https://github.com/open-webui/open-webui)
- [Open WebUI Documentation](https://docs.openwebui.com)
- [Kohai Organization](https://github.com/kohai-io)

## 📧 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/kohai-io/open-webui-functions-tools/issues)
- Contact: mail@kohai.io
