# Prompt2Notes — RAG-powered Video & Image Summariser (MVP)

A production-oriented, CPU-friendly application that processes video and image uploads to generate timestamped transcripts, creates semantic embeddings, and provides RAG-powered summarization with PDF export capabilities.

# Deployed on Streamlit
https://prompt2notes-jikjxmvmyr2u6vwr8dumnm.streamlit.app
## Features

- 📹 **Video & Image Upload**: Accepts MP4, WAV, and image files
- 🎤 **Automatic Transcription**: Uses faster-whisper (2-4x faster than standard Whisper)
- 📝 **Timestamped Chunking**: Intelligent chunking with overlap for semantic continuity
- 🔍 **Semantic Search**: Vector-based retrieval using all-MiniLM-L6-v2 embeddings
- 💾 **Persistent Storage**: ChromaDB for vector storage with metadata
- 🤖 **RAG Summarization**: Hierarchical summarization (micro → meta) with evidence retrieval
- 📄 **PDF Export**: Generate formatted notes with timestamps and references
- ⚡ **Caching**: Filesystem cache for transcripts and embeddings

## Project Structure

```
Prompt2Notes/
├── app.py                 # Streamlit web UI
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── backend/
│   ├── __init__.py
│   ├── asr.py            # Whisper transcription wrapper
│   ├── chunker.py        # Timestamped chunking logic
│   ├── embedder.py       # Sentence transformer embeddings
│   ├── vectorstore.py    # ChromaDB wrapper
│   ├── rag.py            # RAG orchestration & summarization
│   └── pdf_export.py     # PDF generation
├── utils/
│   ├── __init__.py
│   ├── file_hash.py      # File hashing for caching
│   └── cache.py          # Filesystem cache
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_vectorstore_mock.py
│   └── test_rag_prompt_format.py
├── examples/
│   └── README.md         # Instructions for creating test video
└── .github/
    └── workflows/
        └── test.yml      # CI workflow (optional)
```

## Setup

### Prerequisites

- Python 3.11+ (recommended for CI and hosting)
- pip
- FFmpeg (for video/audio processing)

Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Installation

1. Clone or navigate to the project directory:
```bash
cd Prompt2Notes
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For external LLM API support, create a `.env` file:
```bash
# .env
# Google Gemini (recommended - free tier available)
GEMINI_API_KEY=your_gemini_key_here
# Or OpenAI
OPENAI_API_KEY=your_openai_key_here
# Note: If both are set, Gemini will be used (priority: Gemini > OpenAI > Local)
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Workflow

1. **Upload**: Upload a video (MP4) or audio file (WAV/MP3)
2. **Process**: Click "Start Processing" to:
   - Extract audio from video
   - Transcribe using faster-whisper (2-4x faster)
   - Extract video frames for visual context
   - Chunk transcript with timestamps
   - Generate embeddings
   - Store in ChromaDB
3. **Query**: Enter a query for semantic search or summarization
4. **Export**: Download notes as PDF

### Example Prompts

**Summarization:**
```
Summarize this lecture into 5 concise bullet points, focusing on key concepts and action items.
```

**Detailed Notes:**
```
Produce structured notes with sections: Overview, Key Concepts, Examples, Action Items, References (with timestamps).
```

**Follow-up Question:**
```
Where was 'gradient descent' discussed? Provide timestamped evidence and short quote.
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=backend --cov=utils -v
```

## Creating Test Video

See `examples/README.md` for instructions on creating a short test video (< 2 minutes) to validate the pipeline.

Quick test using `ffmpeg`:
```bash
# Generate a 30-second test video with speech
ffmpeg -f lavfi -i testsrc=duration=30:size=640x480:rate=30 \
       -f lavfi -i sine=frequency=440:duration=30 \
       -c:v libx264 -c:a aac examples/test_video.mp4
```

## Configuration

### ASR Model
Default: `tiny` using faster-whisper (2-4x faster, CPU-optimized)
- Uses faster-whisper by default (recommended for speed)
- Falls back to openai-whisper if faster-whisper unavailable
- Change model size in `app.py`: `model_name = "tiny"` → `"base"` or `"small"` for better accuracy (slower)

### Chunking Parameters
Default: `chunk_secs=75`, `overlap_secs=15`
- Modify in `backend/chunker.py` or via UI

### Embedding Model
Default: `all-MiniLM-L6-v2` (384 dimensions, fast)
- Change in `backend/embedder.py` if needed

### RAG Parameters
- `top_k`: Number of chunks to retrieve (default: 5)
- `chunk_size`: Maximum tokens per chunk
- Configure in `backend/rag.py`

### LLM Configuration
The app defaults to a simple local summarizer for offline use. To use external APIs:

1. **Google Gemini** (Recommended): Set `GEMINI_API_KEY` in `.env`
   - Get free API key: https://makersuite.google.com/app/apikey
   - Free tier with generous limits
2. **OpenAI API**: Set `OPENAI_API_KEY` in `.env`
   - Get API key: https://platform.openai.com/api-keys
3. **Priority**: If both keys are set, Gemini will be used (Gemini > OpenAI > Local)

See `backend/rag.py` for configuration options.

## Deployment

### Hugging Face Spaces

1. Create a new Space with Streamlit SDK
2. Upload all project files
3. Add `requirements.txt`
4. Set environment variables in Space settings (if using external APIs)
5. Deploy

### Local Production

For production use, consider:
- Using `gunicorn` or `uvicorn` with Streamlit
- Setting up proper logging
- Using a production-grade vector database
- Implementing authentication

## Performance Notes

- **Transcription**: ~1-2x real-time with faster-whisper (e.g., 1 min video = 1-2 min processing)
  - With faster-whisper: 2-4x faster than standard Whisper
  - On M2 Mac: Optimized CPU operations provide excellent performance
- **Embeddings**: ~100-500 chunks/second on CPU
- **Memory**: ~2-4 GB RAM for typical usage
- **Storage**: ChromaDB persists in `./chroma_db/` directory

## Troubleshooting

### Whisper not found
If `openai-whisper` fails, the app will log instructions to use `whisper.cpp`. Alternatively:
```bash
pip install --upgrade openai-whisper
```

### ChromaDB errors
Clear the database:
```bash
rm -rf chroma_db/
```

### Memory issues
- Use `whisper-tiny` model
- Reduce chunk size
- Process shorter videos

## Example Run Transcript

```
[INFO] Starting Prompt2Notes processing...
[INFO] Uploaded file: test_video.mp4 (1.5 MB)
[INFO] Extracting audio... Done (2.3s)
[INFO] Transcribing with Whisper-tiny... Done (45.2s)
[INFO] Transcript: 12 segments, 1m 23s total
[INFO] Chunking transcript... 3 chunks created
[INFO] Generating embeddings... Done (1.1s)
[INFO] Storing in ChromaDB... Done (0.3s)
[INFO] Processing complete!

Query: "What are the main topics discussed?"
[INFO] Retrieved 5 relevant chunks
[INFO] Generating summary...
[INFO] Summary generated (2.1s)

Results:
- Chunk 1 (0:00-1:15): Introduction to machine learning...
- Chunk 2 (1:15-2:30): Discussion of neural networks...
...
```

## License

This project is open-source. See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8
- Tests pass
- Documentation is updated

## Acknowledgments

- OpenAI Whisper for ASR
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- Streamlit for UI framework

