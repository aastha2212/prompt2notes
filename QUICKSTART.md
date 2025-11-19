# Quick Start Guide

Get Prompt2Notes running in 5 minutes!

## Prerequisites

- Python 3.8+
- FFmpeg installed
- ~2GB free disk space (for models)

## Installation

```bash
# 1. Clone or navigate to project
cd Prompt2Notes

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install OpenAI package for better summaries
pip install openai
```

## Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## First Test

1. **Create a test video** (see `examples/README.md`):
   ```bash
   ffmpeg -f lavfi -i testsrc=duration=30:size=640x480:rate=30 \
          -f lavfi -i sine=frequency=440:duration=30 \
          -c:v libx264 -c:a aac test_video.mp4
   ```

2. **Upload** the video in the app

3. **Process** - Click "Start Processing" (takes 1-2 minutes on CPU)

4. **Query** - Try: "Summarize this video"

5. **Export** - Download the PDF notes

## Troubleshooting

**Whisper not found?**
```bash
pip install --upgrade openai-whisper
```

**FFmpeg not found?**
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from https://ffmpeg.org

**Out of memory?**
- Use `whisper-tiny` model (default)
- Process shorter videos (< 2 minutes)

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/README.md](examples/README.md) for test video creation
- Run tests: `pytest tests/ -v`

