# Test Video Creation Instructions

This directory contains instructions for creating test videos to validate the Prompt2Notes pipeline.

## Quick Test Video Generation

### Option 1: Using FFmpeg (Recommended)

Generate a simple test video with audio:

```bash
# Generate 30-second test video with test pattern and sine wave audio
ffmpeg -f lavfi -i testsrc=duration=30:size=640x480:rate=30 \
       -f lavfi -i sine=frequency=440:duration=30 \
       -c:v libx264 -c:a aac examples/test_video.mp4
```

### Option 2: Using FFmpeg with Text-to-Speech (if available)

If you have `espeak` or `say` (macOS) installed:

```bash
# macOS
say "This is a test video for Prompt2Notes. Machine learning is a subset of artificial intelligence. Neural networks are computational models inspired by biological neurons. Deep learning uses multiple layers to learn hierarchical representations." -o test_audio.aiff
ffmpeg -f lavfi -i testsrc=duration=30:size=640x480:rate=30 \
       -i test_audio.aiff \
       -c:v libx264 -c:a aac -shortest examples/test_video.mp4
```

### Option 3: Record Your Own

1. Record a short video (< 2 minutes) using your phone or webcam
2. Save it as MP4 format
3. Place it in the `examples/` directory

## Sample Test Content

For best results, create a video with clear speech covering topics like:

- **Introduction**: "Welcome to this test lecture on machine learning."
- **Key Concepts**: "Machine learning involves training algorithms on data to make predictions."
- **Examples**: "For example, image recognition uses convolutional neural networks."
- **Conclusion**: "In summary, machine learning is transforming many industries."

## Validation Checklist

After processing your test video, verify:

- [ ] Transcript is generated with timestamps
- [ ] Chunks are created (typically 1-3 chunks for a 1-2 minute video)
- [ ] Embeddings are generated and stored
- [ ] Semantic search returns relevant chunks
- [ ] RAG summarization produces coherent output
- [ ] PDF export generates a formatted document

## Expected Processing Times (CPU)

- **30-second video**: ~1-2 minutes transcription
- **1-minute video**: ~2-4 minutes transcription
- **2-minute video**: ~4-8 minutes transcription

*Note: Times vary based on CPU performance*

## Troubleshooting

### Video not processing
- Ensure FFmpeg is installed: `ffmpeg -version`
- Check file format is supported (MP4, AVI, MOV, MKV)
- Verify file is not corrupted

### No audio extracted
- Check video has audio track: `ffprobe -i video.mp4`
- Try converting audio first: `ffmpeg -i video.mp4 -vn -acodec copy audio.wav`

### Transcription fails
- Ensure Whisper model is downloaded (first run will download automatically)
- Check available disk space
- Verify audio quality is sufficient

