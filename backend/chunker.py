"""
Timestamped chunking logic with sliding overlap for semantic continuity.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TimestampedChunker:
    """
    Chunks transcript segments with timestamp preservation and overlap.
    """
    
    def __init__(self, chunk_secs: float = 75.0, overlap_secs: float = 15.0):
        """
        Initialize chunker.
        
        Args:
            chunk_secs: Target chunk duration in seconds
            overlap_secs: Overlap duration in seconds between chunks
        """
        self.chunk_secs = chunk_secs
        self.overlap_secs = overlap_secs
        
        if overlap_secs >= chunk_secs:
            raise ValueError("overlap_secs must be less than chunk_secs")
    
    def chunk_segments(
        self,
        segments: List[Dict[str, Any]],
        video_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Chunk transcript segments with timestamps and overlap.
        
        Args:
            segments: List of segments with 'start', 'end', 'text' keys
            video_id: Identifier for the video
            
        Returns:
            List of chunks with metadata: chunk_id, start, end, text, video_id
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk_text = []
        current_chunk_start = segments[0]["start"]
        current_chunk_end = segments[0]["end"]
        chunk_id = 0
        
        # Track last chunk end for overlap calculation
        last_chunk_end = 0.0
        
        for i, segment in enumerate(segments):
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_text = segment["text"]
            
            # Check if adding this segment would exceed chunk size
            # or if we need to start a new chunk due to overlap
            segment_duration = seg_end - seg_start
            current_chunk_duration = current_chunk_end - current_chunk_start
            
            # Start new chunk if:
            # 1. Current chunk + segment would exceed chunk_secs, OR
            # 2. We've passed the overlap point from last chunk
            should_start_new = (
                (current_chunk_duration + segment_duration > self.chunk_secs) and
                current_chunk_text  # Only if we have content
            )
            
            if should_start_new:
                # Save current chunk
                chunk_text = " ".join(current_chunk_text)
                if chunk_text.strip():
                    chunks.append({
                        "chunk_id": f"{video_id}_chunk_{chunk_id}",
                        "start": current_chunk_start,
                        "end": current_chunk_end,
                        "text": chunk_text.strip(),
                        "video_id": video_id
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                # Calculate overlap start: go back by overlap_secs from current position
                overlap_start = max(0.0, current_chunk_end - self.overlap_secs)
                
                # Find segments that fall within overlap window
                overlap_text = []
                for prev_seg in segments:
                    if prev_seg["start"] >= overlap_start and prev_seg["end"] <= current_chunk_end:
                        overlap_text.append(prev_seg["text"])
                
                current_chunk_text = overlap_text + [seg_text]
                current_chunk_start = overlap_start
                current_chunk_end = seg_end
                last_chunk_end = current_chunk_end
            else:
                # Add segment to current chunk
                current_chunk_text.append(seg_text)
                current_chunk_end = seg_end
        
        # Add final chunk
        if current_chunk_text:
            chunk_text = " ".join(current_chunk_text)
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{video_id}_chunk_{chunk_id}",
                    "start": current_chunk_start,
                    "end": current_chunk_end,
                    "text": chunk_text.strip(),
                    "video_id": video_id
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks
    
    def chunk_text(
        self,
        text: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
        video_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Chunk plain text with estimated timestamps.
        
        Args:
            text: Plain text to chunk
            start_time: Start timestamp
            end_time: End timestamp
            video_id: Video identifier
            
        Returns:
            List of chunks
        """
        # Simple sentence-based chunking for plain text
        sentences = text.split(". ")
        segments = []
        total_duration = end_time - start_time if end_time > start_time else len(text) * 0.05  # Estimate
        
        if sentences:
            time_per_sentence = total_duration / len(sentences)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    seg_start = start_time + i * time_per_sentence
                    seg_end = seg_start + time_per_sentence
                    segments.append({
                        "start": seg_start,
                        "end": seg_end,
                        "text": sentence.strip() + "."
                    })
        
        return self.chunk_segments(segments, video_id)

