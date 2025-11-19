"""
Prompt2Notes - Streamlit Web UI
Single-page application for video/image upload, processing, query, and PDF export.
"""

import streamlit as st
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy imports - only import when needed to speed up startup
# Heavy modules (models, ML libraries) are imported on-demand

# Configuration constants
MAX_VIDEO_DURATION_MINUTES = 60  # Maximum video duration in minutes (configurable)
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB (Streamlit default is 200MB, but can be increased)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Page configuration
st.set_page_config(
    page_title="Prompt2Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = {}
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "processing_log" not in st.session_state:
    st.session_state.processing_log = []
if "visual_frames" not in st.session_state:
    st.session_state.visual_frames = {}  # Maps chunk_id to list of frames
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None
if "frame_paths" not in st.session_state:
    st.session_state.frame_paths = []  # Store frame file paths for cleanup


def log_message(message: str, level: str = "info"):
    """Add message to processing log."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.processing_log.append(f"[{timestamp}] {message}")
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)


def initialize_components():
    """Initialize backend components (lazy loading)."""
    try:
        # Lazy import to avoid loading heavy dependencies on startup
        from backend.vectorstore import VectorStore
        from backend.embedder import Embedder
        from backend.rag import RAGOrchestrator

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = VectorStore()
            log_message("Vector store initialized")

        if st.session_state.rag is None:
            if st.session_state.vectorstore is not None:
                embedder = Embedder()
                # Check for LLM API keys (priority: Gemini > OpenAI > Local)
                if os.getenv("GEMINI_API_KEY"):
                    llm_provider = "gemini"
                elif os.getenv("OPENAI_API_KEY"):
                    llm_provider = "openai"
                else:
                    llm_provider = "local"
                st.session_state.rag = RAGOrchestrator(
                    vectorstore=st.session_state.vectorstore,
                    embedder=embedder,
                    top_k=5,
                    llm_provider=llm_provider
                )
                log_message(f"RAG orchestrator initialized (provider: {llm_provider})")
    except Exception as e:
        log_message(f"Failed to initialize components: {e}", "error")
        st.error(f"Initialization error: {e}")


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds
    """
    try:
        # Lazy import moviepy
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        return duration
    except Exception as e:
        logger.warning(f"Could not get video duration: {e}")
        return 0.0


def validate_video_file(file, file_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate video file size and duration.

    Args:
        file: File object or None
        file_path: File path if file is a string

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    if file and hasattr(file, 'size'):
        file_size_mb = file.size / 1024 / 1024
        if file_size_mb > MAX_FILE_SIZE_MB:
            return False, f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed ({MAX_FILE_SIZE_MB} MB)"

    # Check video duration if it's a video file
    video_path = file_path if file_path else (file.name if hasattr(file, 'name') else None)
    if video_path:
        file_ext = Path(video_path).suffix.lower()
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            try:
                duration = get_video_duration(video_path)
                duration_minutes = duration / 60
                if duration > 0 and duration_minutes > MAX_VIDEO_DURATION_MINUTES:
                    return False, f"Video duration ({duration_minutes:.1f} minutes) exceeds maximum allowed ({MAX_VIDEO_DURATION_MINUTES} minutes)"
            except Exception as e:
                logger.warning(f"Could not validate video duration: {e}")
                # Don't fail validation if we can't check duration

    return True, None


def process_video(file, video_id: str) -> bool:
    """
    Process uploaded video: extract audio, transcribe, chunk, embed, store.

    Args:
        file: Uploaded file object or file path (str)
        video_id: Unique video identifier

    Returns:
        True if successful
    """
    try:
        # Handle both file objects and file paths
        if isinstance(file, str):
            # File path provided
            tmp_path = file
            file_name = Path(file).name
            log_message(f"Starting processing for: {file_name}")

            # Validate file
            is_valid, error_msg = validate_video_file(None, file_path=file)
            if not is_valid:
                log_message(f"Validation failed: {error_msg}", "error")
                st.error(f"❌ {error_msg}")
                return False
        else:
            # File object provided
            file_name = file.name
            log_message(f"Starting processing for: {file_name}")

            # Validate file size
            is_valid, error_msg = validate_video_file(file)
            if not is_valid:
                log_message(f"Validation failed: {error_msg}", "error")
                st.error(f"❌ {error_msg}")
                return False

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                if hasattr(file, 'getvalue'):
                    tmp_file.write(file.getvalue())
                elif hasattr(file, 'read'):
                    tmp_file.write(file.read())
                else:
                    raise ValueError("File object must have getvalue() or read() method")
                tmp_path = tmp_file.name

        # Check video duration after saving
        file_ext = Path(tmp_path).suffix.lower()
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            try:
                duration = get_video_duration(tmp_path)
                duration_minutes = duration / 60
                if duration > 0:
                    log_message(f"Video duration: {duration_minutes:.1f} minutes ({duration:.1f} seconds)")
                    if duration_minutes > MAX_VIDEO_DURATION_MINUTES:
                        error_msg = f"Video duration ({
                            duration_minutes:.1f} minutes) exceeds maximum ({MAX_VIDEO_DURATION_MINUTES} minutes)"
                        log_message(error_msg, "error")
                        st.error(f"❌ {error_msg}")
                        st.info(f"💡 For longer videos, consider splitting them into smaller segments.")
                        return False
                    elif duration_minutes > MAX_VIDEO_DURATION_MINUTES * 0.8:
                        st.warning(
                            f"⚠️ Video is {
                                duration_minutes:.1f} minutes. Processing may take a while (~{
                                duration_minutes * 3:.0f}-{
                                duration_minutes * 5:.0f} minutes).")
            except Exception as e:
                logger.warning(f"Could not check video duration: {e}")

        if not isinstance(file, str):
            log_message("File saved temporarily")

        # Initialize cache
        from utils.cache import FileSystemCache
        from utils.file_hash import compute_file_hash
        cache = FileSystemCache()
        file_hash = compute_file_hash(tmp_path)

        # Check cache for transcript
        cached_transcript = cache.get(f"transcript_{file_hash}")
        audio_path = None
        if cached_transcript:
            log_message("Using cached transcript")
            segments = cached_transcript
        else:
            # Extract audio if video
            file_ext = Path(tmp_path).suffix.lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                log_message("Extracting audio from video...")
                from backend.asr import extract_audio_from_video
                audio_path = extract_audio_from_video(tmp_path)
            else:
                audio_path = tmp_path

            # Transcribe
            log_message("Transcribing with faster-whisper (optimized for speed)...")
            from backend.asr import ASRTranscriber
            transcriber = ASRTranscriber(model_name="tiny", use_faster_whisper=True)
            segments = transcriber.transcribe(audio_path)

            # Validate transcript
            if not segments or len(segments) == 0:
                error_msg = "Transcription returned no segments. Video may have no audio track or audio is too quiet."
                log_message(error_msg, "error")
                st.error(f"❌ {error_msg}")
                return False

            # Cache transcript
            cache.set(f"transcript_{file_hash}", segments)
            log_message(f"Transcription complete: {len(segments)} segments")

        st.session_state.transcript = segments

        # Chunk transcript
        log_message("Chunking transcript...")
        from backend.chunker import TimestampedChunker
        chunker = TimestampedChunker(chunk_secs=75.0, overlap_secs=15.0)
        chunks = chunker.chunk_segments(segments, video_id=video_id)
        st.session_state.chunks = chunks
        log_message(f"Created {len(chunks)} chunks")

        # Validate we have chunks
        if not chunks:
            error_msg = "No chunks created from transcript. Video may have no audio or transcription failed."
            log_message(error_msg, "error")
            st.error(f"❌ {error_msg}")
            return False

        # Extract video frames for visual context (if video file)
        visual_frames = {}
        file_ext = Path(tmp_path).suffix.lower()
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            try:
                log_message("Extracting video frames for visual analysis...")
                from backend.frame_extractor import FrameExtractor
                frame_extractor = FrameExtractor()
                # Extract 1 frame per chunk (at the start of each chunk)
                visual_frames = frame_extractor.extract_frames_for_chunks(
                    tmp_path,
                    chunks,
                    frames_per_chunk=1
                )
                st.session_state.visual_frames = visual_frames
                log_message(f"Extracted frames for {len(visual_frames)} chunks")

                # Store frame paths for cleanup later
                frame_paths = []
                for chunk_frames in visual_frames.values():
                    for frame in chunk_frames:
                        if "path" in frame:
                            frame_paths.append(frame["path"])
                st.session_state.frame_paths = frame_paths
            except Exception as e:
                log_message(f"Frame extraction failed (continuing without visual context): {e}", "warning")
                visual_frames = {}
                st.session_state.frame_paths = []

        # Generate embeddings
        log_message("Generating embeddings...")
        from backend.embedder import Embedder
        embedder = Embedder()
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.encode(chunk_texts, show_progress=False)
        log_message(f"Generated {len(embeddings)} embeddings")

        # Store in vector database
        log_message("Storing in vector database...")
        initialize_components()
        embeddings_list = embeddings.tolist()
        st.session_state.vectorstore.add_chunks(chunks, embeddings_list)
        log_message("Stored in vector database")

        # Cleanup (only if we created the temp file)
        if not isinstance(file, str):
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        if audio_path and audio_path != tmp_path and os.path.exists(audio_path):
            os.unlink(audio_path)

        st.session_state.processed_videos[video_id] = {
            "filename": file_name if not isinstance(file, str) else Path(file).name,
            "segments": len(segments),
            "chunks": len(chunks)
        }
        st.session_state.current_video_id = video_id

        log_message("Processing complete!")
        return True

    except Exception as e:
        log_message(f"Processing failed: {e}", "error")
        st.error(f"Processing error: {e}")
        return False


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def show_login_page():
    """Display login/registration page."""
    st.title("🔐 Prompt2Notes Login")
    st.markdown("**RAG-powered Video & Image Summariser**")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to Prompt2Notes")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", type="primary")

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    from backend.auth import get_auth_manager
                    auth_manager = get_auth_manager()
                    success, error = auth_manager.authenticate(username, password)

                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")

    with tab2:
        st.subheader("Create New Account")

        with st.form("register_form"):
            new_username = st.text_input("Username", placeholder="Choose a username (min 3 characters)")
            new_password = st.text_input(
                "Password",
                type="password",
                placeholder="Choose a password (min 4 characters)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            email = st.text_input("Email (Optional)", placeholder="your.email@example.com")
            submit_register = st.form_submit_button("Register", type="primary")

            if submit_register:
                if not new_username or not new_password:
                    st.error("Please enter both username and password")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    from backend.auth import get_auth_manager
                    auth_manager = get_auth_manager()
                    success, message = auth_manager.register_user(new_username, new_password, email)

                    if success:
                        st.success(f"✅ {message}")
                        st.info("You can now login with your new account")
                    else:
                        st.error(f"❌ {message}")


def show_main_app():
    """Display main application (protected content)."""
    # Main UI
    st.title("📝 Prompt2Notes")
    st.markdown("**RAG-powered Video & Image Summariser**")

    # Sidebar with user info and settings
    with st.sidebar:
        st.info(f"👤 Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.divider()

        st.header("Settings")

        # LLM Provider info
        if os.getenv("GEMINI_API_KEY"):
            llm_provider = "Google Gemini"
        elif os.getenv("OPENAI_API_KEY"):
            llm_provider = "OpenAI API"
        else:
            llm_provider = "Local (Fallback)"

        st.info(f"LLM Provider: {llm_provider}")

        if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ No API key found. Using local summarizer.")
            st.caption("Set GEMINI_API_KEY or OPENAI_API_KEY in .env for better summaries.")

        st.divider()

        # User management section
        st.subheader("👤 Account")
        if st.button("Change Password", type="secondary"):
            st.session_state.show_change_password = True

        if st.session_state.get("show_change_password", False):
            with st.expander("Change Password", expanded=True):
                with st.form("change_password_form"):
                    old_password = st.text_input("Current Password", type="password")
                    new_password = st.text_input("New Password", type="password")
                    confirm_new_password = st.text_input("Confirm New Password", type="password")
                    change_submit = st.form_submit_button("Change Password", type="primary")
                    cancel = st.form_submit_button("Cancel")

                    if cancel:
                        st.session_state.show_change_password = False
                        st.rerun()

                    if change_submit:
                        if not old_password or not new_password:
                            st.error("Please fill in all fields")
                        elif new_password != confirm_new_password:
                            st.error("New passwords do not match")
                        else:
                            from backend.auth import get_auth_manager
                            auth_manager = get_auth_manager()
                            success, message = auth_manager.change_password(
                                st.session_state.username,
                                old_password,
                                new_password
                            )
                            if success:
                                st.success(f"✅ {message}")
                                st.session_state.show_change_password = False
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")

        st.divider()

        # Video limits info
        st.subheader("📏 Limits")
        st.caption(f"Max file size: {MAX_FILE_SIZE_MB} MB")
        st.caption(f"Max video duration: {MAX_VIDEO_DURATION_MINUTES} minutes")
        st.caption("Processing time: ~2-5x video duration")

        st.divider()

        # Vector store info
        if st.session_state.vectorstore:
            try:
                info = st.session_state.vectorstore.get_collection_info()
                st.metric("Stored Chunks", info.get("chunk_count", 0))
            except BaseException:
                pass

        st.divider()

        # Clear cache button
        if st.button("Clear Cache", type="secondary"):
            from utils.cache import FileSystemCache
            cache = FileSystemCache()
            cache.clear()

            # Also cleanup extracted frames
            if st.session_state.frame_paths:
                from backend.frame_extractor import FrameExtractor
                frame_extractor = FrameExtractor()
                frame_extractor.cleanup_frames(st.session_state.frame_paths)
                st.session_state.frame_paths = []

            st.success("Cache cleared")
            st.rerun()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Query & Search", "📄 Export"])

    # Tab 1: Upload & Process
    with tab1:
        st.header("Upload Video, Audio, or Image")

        # URL input section
        st.subheader("📎 Or paste a URL")
        url_input = st.text_input(
            "Enter YouTube link or direct video/image URL:",
            placeholder="https://www.youtube.com/watch?v=... or https://example.com/video.mp4",
            help="Supports YouTube links and direct video/image URLs"
        )

        # File upload section
        st.subheader("📁 Or upload a file")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a video or audio file",
                type=["mp4", "avi", "mov", "mkv", "wav", "mp3", "m4a"],
                help=f"Upload a video or audio file for processing (Max: {MAX_FILE_SIZE_MB}MB, Max duration: {MAX_VIDEO_DURATION_MINUTES} minutes)"
            )

            with col2:
                uploaded_image = st.file_uploader(
                    "Or choose an image file",
                    type=["jpg", "jpeg", "png", "gif", "webp"],
                    help="Upload an image file for visual Q&A (requires Gemini API)"
                )

        # Process URL input
        if url_input:
            url_input = url_input.strip()
            if url_input:
                from utils.url_downloader import is_youtube_url, is_valid_url
                if is_youtube_url(url_input):
                    st.info(f"🎥 YouTube URL detected: {url_input}")
                    if st.button("📥 Download & Process YouTube Video", type="primary"):
                        with st.spinner("Downloading YouTube video (this may take a while)..."):
                            try:
                                log_message("Downloading YouTube video...")
                                from utils.url_downloader import download_youtube_video
                                video_path = download_youtube_video(url_input)
                                log_message(f"YouTube video downloaded: {video_path}")

                                # Process the downloaded video directly using file path
                                from utils.file_hash import compute_file_hash
                                video_id = compute_file_hash(video_path)
                                success = process_video(video_path, video_id)

                                if success:
                                    st.success("✅ YouTube video processed successfully!")
                                    st.balloons()

                                    # Cleanup downloaded file after processing (frames are already extracted)
                                    try:
                                        if os.path.exists(video_path):
                                            os.unlink(video_path)
                                            log_message("Cleaned up downloaded YouTube video file")
                                    except Exception as cleanup_error:
                                        log_message(f"Cleanup warning: {cleanup_error}", "warning")
                                else:
                                    st.error("❌ Processing failed")
                                    # Keep file if processing failed for debugging

                            except Exception as e:
                                log_message(f"YouTube download failed: {e}", "error")
                                st.error(f"Failed to download YouTube video: {e}")
                                st.info("💡 Make sure yt-dlp is installed: pip install yt-dlp")

                elif is_valid_url(url_input):
                    # Check if it's likely an image or video
                    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

                    url_lower = url_input.lower()
                    is_image_url = any(url_lower.endswith(ext) for ext in image_extensions) or 'image' in url_lower
                    is_video_url = any(url_lower.endswith(ext) for ext in video_extensions) or 'video' in url_lower

                    if is_image_url:
                        st.info(f"🖼️ Image URL detected: {url_input}")
                        if st.button("📥 Download & Load Image", type="primary"):
                            with st.spinner("Downloading image..."):
                                try:
                                    log_message("Downloading image from URL...")
                                    from utils.url_downloader import download_image_from_url
                                    image_path = download_image_from_url(url_input)
                                    log_message(f"Image downloaded: {image_path}")

                                    from PIL import Image as PILImage
                                    pil_image = PILImage.open(image_path)
                                    st.session_state.current_image = pil_image

                                    st.image(pil_image, caption="Downloaded Image", use_container_width=True)
                                    st.success("✅ Image loaded! You can now ask questions about it in the Query & Search tab.")

                                    # Cleanup
                                    try:
                                        if os.path.exists(image_path):
                                            os.unlink(image_path)
                                    except:
                                        pass

                                except Exception as e:
                                    log_message(f"Image download failed: {e}", "error")
                                    st.error(f"Failed to download image: {e}")

                    elif is_video_url:
                        st.info(f"🎥 Video URL detected: {url_input}")
                        if st.button("📥 Download & Process Video", type="primary"):
                            with st.spinner("Downloading video (this may take a while)..."):
                                try:
                                    log_message("Downloading video from URL...")
                                    from utils.url_downloader import download_video_from_url
                                    video_path = download_video_from_url(url_input)
                                    log_message(f"Video downloaded: {video_path}")

                                    # Process the downloaded video directly using file path
                                    from utils.file_hash import compute_file_hash
                                    video_id = compute_file_hash(video_path)
                                    success = process_video(video_path, video_id)

                                    if success:
                                        st.success("✅ Video processed successfully!")
                                        st.balloons()

                                        # Cleanup downloaded file after processing
                                        try:
                                            if os.path.exists(video_path):
                                                os.unlink(video_path)
                                                log_message("Cleaned up downloaded video file")
                                        except Exception as cleanup_error:
                                            log_message(f"Cleanup warning: {cleanup_error}", "warning")
                                    else:
                                        st.error("❌ Processing failed")
                                        # Keep file if processing failed for debugging

                                except Exception as e:
                                    log_message(f"Video download failed: {e}", "error")
                                    st.error(f"Failed to download video: {e}")

                    else:
                        st.warning("⚠️ Could not determine if URL is an image or video. Try a direct link to a .jpg, .png, .mp4, etc.")
            else:
                st.error("❌ Invalid URL format. Please enter a valid URL.")

        # Process image upload
        if uploaded_image:
            st.info(f"🖼️ Image: {uploaded_image.name} ({uploaded_image.size / 1024:.2f} KB)")

            try:
                from PIL import Image as PILImage
                import io

                # Load image
                image_bytes = uploaded_image.read()
                pil_image = PILImage.open(io.BytesIO(image_bytes))
                st.session_state.current_image = pil_image

                # Display image
                st.image(pil_image, caption=uploaded_image.name, use_container_width=True)

                st.success("✅ Image loaded! You can now ask questions about it in the Query & Search tab.")

            except Exception as e:
                st.error(f"Failed to load image: {e}")

        # Process video/audio upload
        if uploaded_file:
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"📁 File: {uploaded_file.name} ({file_size_mb:.2f} MB)")

            # Show warnings for large files
            if file_size_mb > MAX_FILE_SIZE_MB * 0.8:
                st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer.")
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File size ({file_size_mb:.1f} MB) exceeds maximum ({MAX_FILE_SIZE_MB} MB)")
                st.stop()

            # Generate video ID from filename and size
            from utils.file_hash import compute_string_hash
            video_id = compute_string_hash(uploaded_file.name + str(uploaded_file.size))

            if st.button("🚀 Start Processing", type="primary"):
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process video
                    success = process_video(uploaded_file, video_id)

                    progress_bar.progress(100)
                    if success:
                        status_text.success("✅ Processing complete!")
                        st.balloons()
                    else:
                        status_text.error("❌ Processing failed")

            # Show processing log
            if st.session_state.processing_log:
                with st.expander("📋 Processing Log", expanded=False):
                    for log_entry in st.session_state.processing_log[-20:]:  # Last 20 entries
                        st.text(log_entry)

    # Tab 2: Query & Search
    with tab2:
        st.header("Semantic Search & RAG Summarization")

        # Show status
        if st.session_state.current_image and not st.session_state.current_video_id:
            st.info("🖼️ Image loaded! Ask questions about the image. (Requires Gemini API)")
        elif st.session_state.current_video_id is None:
            st.warning("⚠️ Please upload and process a video or upload an image first.")

        if st.session_state.current_video_id or st.session_state.current_image:
            # Query input
            if st.session_state.current_image and not st.session_state.current_video_id:
                placeholder = "e.g., 'What is shown in this image?' or 'Describe the main elements in this picture'"
            else:
                placeholder = "e.g., 'Summarize this lecture into 5 concise bullet points'"
            
            query = st.text_area(
                "Enter your query or prompt:",
                placeholder=placeholder,
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                prompt_type = st.selectbox(
                    "Prompt Type",
                    ["summary", "notes", "qa"],
                    help="summary: concise summary, notes: structured notes, qa: question answering"
                )
            with col2:
                top_k = st.slider("Top K Results", 3, 10, 5)
            
            if st.button("🔍 Search & Generate", type="primary"):
                if query:
                    with st.spinner("Searching and generating..."):
                        try:
                            initialize_components()
                            st.session_state.rag.top_k = top_k
                            
                            # Prepare visual context
                            visual_context = None
                            
                            # If we have a processed video with frames, use them
                            if st.session_state.current_video_id and st.session_state.visual_frames:
                                visual_context = st.session_state.visual_frames
                            
                            # If we have an uploaded image, use it for direct Q&A
                            if st.session_state.current_image and not st.session_state.current_video_id:
                                # For image-only queries, create a special context
                                visual_context = {
                                    "image_query": [{
                                        "image": st.session_state.current_image,
                                        "timestamp": 0.0
                                    }]
                                }
                                # Use Gemini directly for image Q&A
                                if st.session_state.rag.llm_provider == "gemini":
                                    try:
                                        import google.generativeai as genai
                                        model = genai.GenerativeModel("gemini-2.5-flash")
                                        response = model.generate_content([
                                            query,
                                            st.session_state.current_image
                                        ])
                                        result = {
                                            "summary": response.text.strip(),
                                            "evidence": [],
                                            "query": query
                                        }
                                        # Store result for potential export
                                        st.session_state.last_query_result = result
                                    except Exception as e:
                                        st.error(f"Image Q&A failed: {e}")
                                        result = {
                                            "summary": "Image Q&A requires Gemini API. Please set GEMINI_API_KEY.",
                                            "evidence": [],
                                            "query": query
                                        }
                                        st.session_state.last_query_result = result
                                else:
                                    result = {
                                        "summary": "Image Q&A requires Gemini API. Please set GEMINI_API_KEY in .env",
                                        "evidence": [],
                                        "query": query
                                    }
                                    st.session_state.last_query_result = result
                            else:
                                # Normal video/transcript query
                                result = st.session_state.rag.generate(
                                    query=query,
                                    video_id=st.session_state.current_video_id,
                                    prompt_type=prompt_type,
                                    visual_context=visual_context
                                )
                            
                            # Store result for PDF export
                            st.session_state.last_query_result = result
                            
                            # Display summary
                            st.subheader("📊 Summary")
                            st.markdown(result["summary"])
                            
                        except Exception as e:
                            st.error(f"Query failed: {e}")
                            log_message(f"Query error: {e}", "error")
                else:
                    st.warning("Please enter a query.")
    
    # Tab 3: Export
    with tab3:
        st.header("Export Notes to PDF")
        
        if st.session_state.current_video_id is None:
            st.warning("⚠️ Please upload and process a video first.")
        else:
            # Get last query result if available
            export_title = st.text_input(
                "Document Title",
                value="Video Notes",
                help="Title for the PDF document"
            )
            
            if st.button("📥 Export PDF", type="primary"):
                try:
                    # Use last query result if available, otherwise generate default
                    if st.session_state.last_query_result:
                        result = st.session_state.last_query_result
                        query = result.get("query", "Video Summary")
                        st.info("📄 Using last query result for export")
                    elif st.session_state.chunks:
                        initialize_components()
                        st.info("💡 Generating default summary for export...")
                        
                        # Generate a default summary
                        query = "Summarize the main content"
                        result = st.session_state.rag.generate(
                            query=query,
                            video_id=st.session_state.current_video_id,
                            prompt_type="notes"
                        )
                    else:
                        st.error("❌ No content available. Please process a video first.")
                        st.stop()
                    
                    # Export PDF
                    from backend.pdf_export import PDFExporter
                    pdf_exporter = PDFExporter()
                    output_path = f"notes_{st.session_state.current_video_id[:8]}.pdf"
                    
                    pdf_exporter.export_notes(
                        output_path=output_path,
                        title=export_title,
                        summary=result["summary"],
                        evidence=result.get("evidence", []),
                        query=query,
                        video_id=st.session_state.current_video_id
                    )
                    
                    # Provide download
                    with open(output_path, "rb") as pdf_file:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_file.read(),
                            file_name=output_path,
                            mime="application/pdf"
                        )

                    st.success(f"✅ PDF generated: {output_path}")
                    
                except Exception as e:
                    st.error(f"Export failed: {e}")
                    log_message(f"Export error: {e}", "error")
    
    # Footer
    st.divider()
    st.caption("Prompt2Notes MVP - RAG-powered Video Summariser | CPU-friendly, Open-source")


# Show loading indicator only on first run
if "app_loaded" not in st.session_state:
    with st.spinner("🚀 Initializing app..."):
        time.sleep(0.1)  # Brief pause to show spinner
    st.session_state.app_loaded = True

# Authentication check
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# Main application (only shown if authenticated)
show_main_app()
