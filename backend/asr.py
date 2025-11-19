"""
Automatic Speech Recognition (ASR) wrapper for Whisper.
Supports faster-whisper (recommended) and openai-whisper (fallback).
faster-whisper is 2-4x faster with same accuracy.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import warnings

logger = logging.getLogger(__name__)

# Try to import faster-whisper (preferred - faster)
try:
    from faster_whisper import WhisperModel  # type: ignore
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.info("faster-whisper not available. Install with: pip install faster-whisper")

# Try to import standard whisper (fallback)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("openai-whisper not available. Install with: pip install openai-whisper")


class ASRTranscriber:
    """
    Wrapper for Whisper ASR with timestamped transcript generation.
    Uses faster-whisper (preferred) or openai-whisper (fallback).
    Automatically uses GPU on M2 Macs when available.
    """
    
    def __init__(self, model_name: str = "tiny", device: Optional[str] = None, use_faster_whisper: bool = True):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cpu", "cuda", "mps"). Auto-detects if None.
            use_faster_whisper: Whether to use faster-whisper (faster) or openai-whisper (fallback)
        """
        self.model_name = model_name
        self.model = None
        self.use_faster_whisper = use_faster_whisper and FASTER_WHISPER_AVAILABLE
        self.device = device
        
        # Auto-detect device if not specified
        if device is None:
            device = self._detect_device()
        
        self.device = device
        
        # Try faster-whisper first (preferred)
        if self.use_faster_whisper:
            try:
                logger.info(f"Loading faster-whisper model: {model_name}")
                # faster-whisper uses optimized CPU operations (CTranslate2)
                # On M2 Mac, this is still 2-4x faster than standard Whisper
                # Note: faster-whisper doesn't support MPS directly, but CPU is highly optimized
                device_param = "cpu"  # faster-whisper handles optimization internally
                
                # Use int8 for speed on Apple Silicon (good balance of speed/accuracy)
                # Can use "int8_float16" for better accuracy but slightly slower
                compute_type = "int8"
                
                self.model = WhisperModel(model_name, device=device_param, compute_type=compute_type)
                logger.info(f"faster-whisper model loaded successfully (device: {device_param}, compute_type: {compute_type})")
                logger.info("Note: faster-whisper uses optimized CPU operations (2-4x faster than standard Whisper)")
                return
            except Exception as e:
                logger.warning(f"Failed to load faster-whisper: {e}, falling back to openai-whisper")
                self.use_faster_whisper = False
        
        # Fallback to openai-whisper
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Neither faster-whisper nor openai-whisper is installed.\n"
                "Install with: pip install faster-whisper (recommended)\n"
                "Or: pip install openai-whisper"
            )
        
        try:
            logger.info(f"Loading openai-whisper model: {model_name} on {device}")
            self.model = whisper.load_model(model_name, device=device)
            if device == "mps":
                logger.info("openai-whisper model loaded successfully with MPS (Metal GPU acceleration)")
            else:
                logger.info("openai-whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _detect_device(self) -> str:
        """
        Auto-detect the best device to use.
        
        Returns:
            Device string: "mps", "cuda", or "cpu"
        """
        # Check for MPS (Apple Silicon GPU)
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("M2/M1 Mac detected - MPS (Metal) available")
                return "mps"
        except:
            pass
        
        # Check for CUDA (NVIDIA GPU)
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("NVIDIA GPU detected - CUDA available")
                return "cuda"
        except:
            pass
        
        # Default to CPU
        logger.info("Using CPU (no GPU detected)")
        return "cpu"
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> List[Dict[str, any]]:
        """
        Transcribe audio file and return timestamped segments.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, MP4, etc.)
            language: Language code (e.g., "en"). Auto-detects if None.
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            List of segments with keys: start, end, text
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing: {audio_path} (using {'faster-whisper' if self.use_faster_whisper else 'openai-whisper'})")
        
        try:
            if self.use_faster_whisper:
                # faster-whisper API
                segments_gen, info = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=word_timestamps,
                    beam_size=5  # Balance between speed and accuracy
                )
                
                # Convert generator to list
                segments = []
                for segment in segments_gen:
                    segment_data = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    }
                    
                    # Add word-level timestamps if available
                    if word_timestamps and hasattr(segment, 'words') and segment.words:
                        segment_data["words"] = [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end
                            }
                            for w in segment.words
                        ]
                    
                    segments.append(segment_data)
                
                logger.info(f"Transcription complete: {len(segments)} segments (detected language: {info.language}, probability: {info.language_probability:.2f})")
            else:
                # openai-whisper API (fallback)
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=word_timestamps,
                    verbose=False
                )
                
                # Extract segments
                segments = []
                for seg in result.get("segments", []):
                    segment_data = {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "").strip()
                    }
                    
                    # Add word-level timestamps if available
                    if word_timestamps and "words" in seg:
                        segment_data["words"] = [
                            {
                                "word": w.get("word", ""),
                                "start": w.get("start", 0.0),
                                "end": w.get("end", 0.0)
                            }
                            for w in seg.get("words", [])
                        ]
                    
                    segments.append(segment_data)
                
                logger.info(f"Transcription complete: {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_to_text(self, audio_path: str) -> str:
        """
        Transcribe audio and return plain text (no timestamps).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Full transcript text
        """
        segments = self.transcribe(audio_path, word_timestamps=False)
        return " ".join(seg["text"] for seg in segments)


def extract_audio_from_video(video_path: str, output_audio_path: Optional[str] = None) -> str:
    """
    Extract audio from video file using moviepy.
    
    Args:
        video_path: Path to video file
        output_audio_path: Optional output path. Auto-generates if None.
        
    Returns:
        Path to extracted audio file
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError("moviepy is required for video processing. Install with: pip install moviepy")
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_audio_path is None:
        output_audio_path = str(video_path.with_suffix(".wav"))
    
    logger.info(f"Extracting audio from {video_path} to {output_audio_path}")
    
    try:
        video = VideoFileClip(str(video_path))
        audio = video.audio
        audio.write_audiofile(output_audio_path, logger=None)
        audio.close()
        video.close()
        logger.info(f"Audio extracted: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        raise

