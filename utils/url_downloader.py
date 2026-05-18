"""
Utilities for downloading videos and images from URLs and YouTube links.
"""

import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import re

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available. Install with: pip install requests")

try:
    import yt_dlp  # type: ignore
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("yt-dlp not available. Install with: pip install yt-dlp")


def is_youtube_url(url: str) -> bool:
    """
    Check if URL is a YouTube link.
    
    Args:
        url: URL string
        
    Returns:
        True if YouTube URL
    """
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be)',
        r'youtube\.com/watch',
        r'youtu\.be/'
    ]
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available in the system.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_valid_url(url: str) -> bool:
    """
    Check if string is a valid URL.
    
    Args:
        url: URL string
        
    Returns:
        True if valid URL
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None


def download_youtube_video(
    url: str,
    output_dir: Optional[str] = None,
    quality: str = "best"
) -> Optional[str]:
    """
    Download video from YouTube URL.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save video. Uses temp dir if None.
        quality: Video quality ("best", "worst", or specific format)
        
    Returns:
        Path to downloaded video file, or None if failed
    """
    if not YT_DLP_AVAILABLE:
        raise ImportError(
            "yt-dlp is required for YouTube downloads. "
            "Install with: pip install yt-dlp"
        )
    
    if not is_youtube_url(url):
        raise ValueError(f"Not a valid YouTube URL: {url}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.gettempdir()) / "prompt2notes_downloads"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = None

    try:
        # Check if ffmpeg is available
        ffmpeg_available = check_ffmpeg_available()
        
        if ffmpeg_available:
            logger.info("ffmpeg detected - enabling HLS support and format merging")
            # With ffmpeg, we can use best quality formats including HLS
            format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/bestvideo+bestaudio/best'
            hls_prefer_native = True
        else:
            logger.info("ffmpeg not detected - avoiding HLS formats and merging")
            # Without ffmpeg, avoid HLS and formats that require merging
            # Use simpler format selector that's more permissive
            format_selector = 'best[ext=mp4]/best[ext=webm]/best'
            hls_prefer_native = False
        
        # Configure yt-dlp options
        # Try multiple clients in order: android (no token needed), web, then default
        ydl_opts = {
            'format': format_selector,
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            # Use android client first (no PO token required), fallback to web, then default
            'extractor_args': {'youtube': {'player_client': ['android', 'web', 'default']}},
            # Use native HLS downloader if ffmpeg is available
            'hls_prefer_native': hls_prefer_native,
            'prefer_insecure': False,
        }
        
        # Add postprocessor for merging if ffmpeg is available
        if ffmpeg_available:
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
        
        logger.info(f"Downloading YouTube video: {url}")
        
        # Track downloaded filename
        downloaded_file = [None]
        
        def progress_hook(d):
            if d['status'] == 'finished':
                downloaded_file[0] = d.get('filename')
        
        ydl_opts['progress_hooks'] = [progress_hook]
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download
            ydl.download([url])
            
            # Get the actual downloaded file path
            if downloaded_file[0]:
                final_path = Path(downloaded_file[0])
                if final_path.exists():
                    logger.info(f"YouTube video downloaded: {final_path}")
                    return str(final_path)
            
            # Fallback: search for recently downloaded files in output directory
            # Get all files in output directory, sorted by modification time (newest first)
            all_files = sorted(
                output_dir.glob('*'),
                key=lambda p: p.stat().st_mtime if p.is_file() else 0,
                reverse=True
            )
            
            # Filter for video files
            video_extensions = {'.mp4', '.webm', '.mkv', '.flv', '.m4a', '.ts', '.mov'}
            video_files = [f for f in all_files if f.is_file() and f.suffix.lower() in video_extensions]
            
            if video_files:
                # Get the most recently modified video file
                final_path = video_files[0]
                logger.info(f"YouTube video downloaded: {final_path}")
                return str(final_path)
            else:
                raise FileNotFoundError(
                    f"Downloaded file not found in {output_dir}. "
                    f"Download may have failed or file was saved elsewhere."
                )
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to download YouTube video: {e}")

        # YouTube frequently blocks automated downloads (SABR / PO token / 403).
        # Convert these into a clear, user-facing error with next steps.
        lower = error_msg.lower()
        if (
            "http error 403" in lower
            or "403" in lower and "forbidden" in lower
            or "po token" in lower
            or "sabr" in lower
            or "missing a url" in lower
            or "sign in to confirm" in lower
            or "this video is unavailable" in lower
            or "private video" in lower
            or "members-only" in lower
            or "age-restricted" in lower
        ):
            raise RuntimeError(
                "YouTube blocked this download (permission/anti-bot restriction).\n\n"
                "What you can do:\n"
                "- Try a different YouTube video (public, non-age-restricted, not members-only).\n"
                "- Download the video locally (e.g., using your browser or a logged-in tool) and upload the file here.\n"
                "- If you control the environment, updating `yt-dlp` can help when YouTube changes formats.\n\n"
                f"Technical detail: {error_msg}"
            ) from e
        
        # Check if error is related to HLS/fragments/ffmpeg
        if any(keyword in error_msg.lower() for keyword in ['fragment', 'hls', 'm3u8', 'empty', 'ffmpeg']):
            ffmpeg_available = check_ffmpeg_available()
            if not ffmpeg_available:
                raise RuntimeError(
                    f"YouTube download failed: {error_msg}\n\n"
                    "💡 Tip: This video requires HLS format which needs ffmpeg for proper download.\n"
                    "Install ffmpeg to enable YouTube downloads:\n"
                    "  macOS: brew install ffmpeg\n"
                    "  Linux: sudo apt-get install ffmpeg\n"
                    "  Windows: Download from https://ffmpeg.org/download.html\n\n"
                    "After installing ffmpeg, restart the app and try again.\n"
                    "Alternatively, try downloading the video manually and uploading it to the app."
                )
            else:
                # ffmpeg is available but still failed - might be a different issue
                raise RuntimeError(
                    f"YouTube download failed: {error_msg}\n\n"
                    "ffmpeg is installed but the download still failed. "
                    "This might be a temporary YouTube issue. Try again later or download manually."
                )
        raise


def download_image_from_url(
    url: str,
    output_dir: Optional[str] = None,
    timeout: int = 30,
    max_bytes: int = 25 * 1024 * 1024
) -> Optional[str]:
    """
    Download image from URL.
    
    Args:
        url: Image URL
        output_dir: Directory to save image. Uses temp dir if None.
        timeout: Request timeout in seconds
        
    Returns:
        Path to downloaded image file, or None if failed
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "requests is required for image downloads. "
            "Install with: pip install requests"
        )
    
    if not is_valid_url(url):
        raise ValueError(f"Not a valid URL: {url}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.gettempdir()) / "prompt2notes_downloads"
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = None

    try:
        logger.info(f"Downloading image from URL: {url}")
        
        # Download image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        if total_size and total_size > max_bytes:
            raise ValueError(f"Image is too large ({total_size / 1024 / 1024:.1f} MB)")
        
        # Determine file extension from URL or Content-Type
        content_type = response.headers.get('Content-Type', '')
        if 'image/jpeg' in content_type or 'image/jpg' in content_type:
            ext = 'jpg'
        elif 'image/png' in content_type:
            ext = 'png'
        elif 'image/gif' in content_type:
            ext = 'gif'
        elif 'image/webp' in content_type:
            ext = 'webp'
        else:
            # Try to get from URL
            url_path = Path(url.split('?')[0])  # Remove query params
            ext = url_path.suffix.lower().lstrip('.')
            if not ext or ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = 'jpg'  # Default
        
        # Generate filename
        filename = f"downloaded_image_{hash(url) % 10000}.{ext}"
        output_path = output_dir / filename
        
        # Save image
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"Image exceeds size limit ({max_bytes / 1024 / 1024:.1f} MB)")
                    f.write(chunk)
        
        logger.info(f"Image downloaded: {output_path}")
        return str(output_path)
        
    except Exception as e:
        if output_path and output_path.exists():
            output_path.unlink()
        logger.error(f"Failed to download image: {e}")
        raise


def download_video_from_url(
    url: str,
    output_dir: Optional[str] = None,
    max_bytes: int = 500 * 1024 * 1024
) -> Optional[str]:
    """
    Download video from URL (non-YouTube).
    
    Args:
        url: Video URL
        output_dir: Directory to save video. Uses temp dir if None.
        
    Returns:
        Path to downloaded video file, or None if failed
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "requests is required for video downloads. "
            "Install with: pip install requests"
        )
    
    if not is_valid_url(url):
        raise ValueError(f"Not a valid URL: {url}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.gettempdir()) / "prompt2notes_downloads"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = None

    try:
        logger.info(f"Downloading video from URL: {url}")
        
        # Download video
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=300, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        if total_size and total_size > max_bytes:
            raise ValueError(f"Video is too large ({total_size / 1024 / 1024:.1f} MB)")
        
        # Determine file extension
        content_type = response.headers.get('Content-Type', '')
        if 'video/mp4' in content_type:
            ext = 'mp4'
        elif 'video/webm' in content_type:
            ext = 'webm'
        elif 'video/quicktime' in content_type:
            ext = 'mov'
        else:
            # Try to get from URL
            url_path = Path(url.split('?')[0])
            ext = url_path.suffix.lower().lstrip('.')
            if not ext or ext not in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                ext = 'mp4'  # Default
        
        # Generate filename
        filename = f"downloaded_video_{hash(url) % 10000}.{ext}"
        output_path = output_dir / filename
        
        # Save video (streaming for large files)
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"Video exceeds size limit ({max_bytes / 1024 / 1024:.1f} MB)")
                    f.write(chunk)
        
        logger.info(f"Video downloaded: {output_path} ({downloaded / 1024 / 1024:.2f} MB)")
        return str(output_path)
        
    except Exception as e:
        if output_path and output_path.exists():
            output_path.unlink()
        logger.error(f"Failed to download video: {e}")
        raise
