"""
Video frame extraction utility for visual content analysis.
Extracts key frames from videos at specified timestamps.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("moviepy not available. Install with: pip install moviepy")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Install with: pip install Pillow")


class FrameExtractor:
    """
    Extracts frames from video files at specified timestamps.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize frame extractor.
        
        Args:
            output_dir: Directory to save extracted frames. Uses temp dir if None.
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "moviepy is required for frame extraction. "
                "Install with: pip install moviepy"
            )
        
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required for image processing. "
                "Install with: pip install Pillow"
            )
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.gettempdir()) / "prompt2notes_frames"
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frame(
        self,
        video_path: str,
        timestamp: float,
        output_path: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Extract a single frame from video at specified timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            output_path: Optional path to save frame. Returns PIL Image if None.
            
        Returns:
            PIL Image object or None if extraction fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            video = VideoFileClip(str(video_path))
            duration = video.duration
            
            # Clamp timestamp to valid range
            timestamp = max(0.0, min(timestamp, duration - 0.1))
            
            # Extract frame
            frame = video.get_frame(timestamp)
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Save if output path provided
            if output_path:
                pil_image.save(output_path)
                logger.info(f"Frame saved: {output_path}")
            
            video.close()
            return pil_image
            
        except Exception as e:
            logger.error(f"Failed to extract frame at {timestamp}s: {e}")
            return None
    
    def extract_frames_at_timestamps(
        self,
        video_path: str,
        timestamps: List[float],
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract multiple frames at specified timestamps.
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            video_id: Optional video identifier for naming
            
        Returns:
            List of dicts with keys: timestamp, image, path
        """
        frames = []
        video_path_obj = Path(video_path)
        
        for i, timestamp in enumerate(timestamps):
            # Generate output path
            if video_id:
                frame_filename = f"{video_id}_frame_{i}_{int(timestamp)}s.jpg"
            else:
                frame_filename = f"frame_{i}_{int(timestamp)}s.jpg"
            
            output_path = self.output_dir / frame_filename
            
            # Extract frame
            image = self.extract_frame(video_path, timestamp, str(output_path))
            
            if image:
                frames.append({
                    "timestamp": timestamp,
                    "image": image,
                    "path": str(output_path),
                    "frame_id": f"{video_id}_frame_{i}" if video_id else f"frame_{i}"
                })
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def extract_key_frames(
        self,
        video_path: str,
        interval_secs: float = 30.0,
        video_id: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames at regular intervals (key frames).
        
        Args:
            video_path: Path to video file
            interval_secs: Interval between frames in seconds
            video_id: Optional video identifier
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame dicts with timestamp, image, path
        """
        try:
            video = VideoFileClip(str(video_path))
            duration = video.duration
            video.close()
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return []
        
        # Calculate timestamps
        timestamps = []
        current_time = 0.0
        
        while current_time < duration:
            timestamps.append(current_time)
            current_time += interval_secs
            
            if max_frames and len(timestamps) >= max_frames:
                break
        
        # Also include the last frame
        if timestamps and timestamps[-1] < duration - 1.0:
            timestamps.append(duration - 0.5)
        
        return self.extract_frames_at_timestamps(video_path, timestamps, video_id)
    
    def extract_frames_for_chunks(
        self,
        video_path: str,
        chunks: List[Dict[str, Any]],
        frames_per_chunk: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract frames corresponding to transcript chunks.
        Extracts frames at the start, middle, or end of each chunk.
        
        Args:
            video_path: Path to video file
            chunks: List of chunk dicts with 'start' and 'end' keys
            frames_per_chunk: Number of frames to extract per chunk (1-3)
            
        Returns:
            Dict mapping chunk_id to list of frame dicts
        """
        chunk_frames = {}
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            start = chunk.get("start", 0.0)
            end = chunk.get("end", 0.0)
            
            # Calculate timestamps for this chunk
            timestamps = []
            if frames_per_chunk >= 1:
                timestamps.append(start)  # Start frame
            if frames_per_chunk >= 2:
                timestamps.append((start + end) / 2)  # Middle frame
            if frames_per_chunk >= 3:
                timestamps.append(end - 0.5)  # End frame (slightly before end)
            
            # Extract frames
            video_id = chunk.get("video_id", "unknown")
            frames = self.extract_frames_at_timestamps(
                video_path,
                timestamps,
                video_id=f"{video_id}_{chunk_id}"
            )
            
            chunk_frames[chunk_id] = frames
        
        logger.info(f"Extracted frames for {len(chunk_frames)} chunks")
        return chunk_frames
    
    def cleanup_frames(self, frame_paths: List[str]) -> None:
        """
        Delete extracted frame files.
        
        Args:
            frame_paths: List of frame file paths to delete
        """
        for path in frame_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to delete frame {path}: {e}")

