"""
Unit tests for RAG prompt formatting and orchestration.
"""

import pytest
from unittest.mock import Mock, MagicMock
from backend.rag import RAGOrchestrator, clean_generated_answer


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore."""
    vs = Mock()
    vs.query = Mock(return_value=[])
    return vs


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = Mock()
    embedder.encode_single = Mock(return_value=[0.1] * 384)  # Mock embedding
    return embedder


@pytest.fixture
def sample_retrieved_chunks():
    """Sample retrieved chunks for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"start": 0.0, "end": 10.0, "video_id": "video_1"}
        },
        {
            "chunk_id": "chunk_2",
            "text": "Neural networks are inspired by biological neurons.",
            "metadata": {"start": 10.0, "end": 20.0, "video_id": "video_1"}
        },
    ]


def test_rag_initialization_local(mock_vectorstore, mock_embedder):
    """Test RAG initialization with local provider."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        top_k=5,
        llm_provider="local"
    )
    
    assert rag.llm_provider == "local"
    assert rag.top_k == 5


def test_rag_format_prompt_summary(mock_vectorstore, mock_embedder, sample_retrieved_chunks):
    """Test prompt formatting for summary type."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    query = "Summarize the main concepts"
    prompt = rag.format_prompt(query, sample_retrieved_chunks, prompt_type="summary")
    
    assert "Context from video transcript" in prompt
    assert query in prompt
    assert "Summarize" in prompt or "summary" in prompt.lower()
    assert "chunk_1" in prompt or "Machine learning" in prompt
    assert "[Chunk" not in prompt
    assert "Do not mention chunks" in prompt


def test_clean_generated_answer_removes_internal_citations():
    """Model output should not leak retrieval labels to users."""
    text = (
        "Calmness creates space between stimulus and response "
        "(Chunk 1, 3). A pilot's tone can prevent panic "
        "(Chunk 6, visual 06:44)."
    )

    cleaned = clean_generated_answer(text)

    assert "Chunk" not in cleaned
    assert "visual" not in cleaned.lower()
    assert "(3)" not in cleaned
    assert cleaned == "Calmness creates space between stimulus and response. A pilot's tone can prevent panic."


def test_rag_format_prompt_assistant(mock_vectorstore, mock_embedder, sample_retrieved_chunks):
    """Unified assistant/chat prompt includes anti-filler and user message."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local",
    )
    query = "Summarise in 100 words"
    prompt = rag.format_prompt(query, sample_retrieved_chunks, prompt_type="assistant")
    assert "User message:" in prompt
    assert query in prompt
    assert "This video is about:" in prompt
    assert "Do not mention chunks, sources" in prompt


def test_rag_format_prompt_notes(mock_vectorstore, mock_embedder, sample_retrieved_chunks):
    """Test prompt formatting for notes type."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    query = "Create structured notes"
    prompt = rag.format_prompt(query, sample_retrieved_chunks, prompt_type="notes")
    
    assert "structured notes" in prompt.lower()
    assert "Overview" in prompt or "Key Concepts" in prompt
    assert "Action Items" in prompt


def test_rag_format_prompt_qa(mock_vectorstore, mock_embedder, sample_retrieved_chunks):
    """Test prompt formatting for Q&A type."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    query = "What is machine learning?"
    prompt = rag.format_prompt(query, sample_retrieved_chunks, prompt_type="qa")
    
    assert query in prompt
    assert "answer" in prompt.lower() or "question" in prompt.lower()


def test_rag_format_timestamp(mock_vectorstore, mock_embedder):
    """Test timestamp formatting."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    # Test various timestamps
    assert rag._format_timestamp(0.0) == "00:00"
    assert rag._format_timestamp(65.0) == "01:05"
    assert rag._format_timestamp(125.5) == "02:05"  # Rounds down


def test_rag_retrieve(mock_vectorstore, mock_embedder, sample_retrieved_chunks):
    """Test retrieval functionality."""
    # Setup mock to return chunks
    mock_vectorstore.query.return_value = sample_retrieved_chunks
    
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        top_k=5
    )
    
    query = "machine learning"
    results = rag.retrieve(query, video_id="video_1")
    
    # Verify embedder was called
    mock_embedder.encode_single.assert_called_once_with(query)
    
    # Verify vectorstore was called
    mock_vectorstore.query.assert_called_once()
    call_args = mock_vectorstore.query.call_args
    assert call_args[1]["top_k"] == 5
    assert call_args[1]["filter_dict"] == {"video_id": "video_1"}
    
    assert len(results) == len(sample_retrieved_chunks)


def test_rag_generate(mock_vectorstore, mock_embedder, sample_retrieved_chunks):
    """Test full RAG generation pipeline."""
    # Setup mocks
    mock_vectorstore.query.return_value = sample_retrieved_chunks
    
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        top_k=5,
        llm_provider="local"
    )
    
    query = "Summarize the content"
    result = rag.generate(query, video_id="video_1", prompt_type="summary")
    
    # Check result structure
    assert "summary" in result
    assert "evidence" in result
    assert "query" in result
    assert result["query"] == query
    assert len(result["evidence"]) == len(sample_retrieved_chunks)
    assert isinstance(result["summary"], str)


def test_rag_generate_no_results(mock_vectorstore, mock_embedder):
    """Test generation when no results are found."""
    mock_vectorstore.query.return_value = []
    
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    query = "test query"
    result = rag.generate(query)
    
    assert result["summary"] == "No relevant content found."
    assert result["evidence"] == []


def test_rag_local_summarizer(mock_vectorstore, mock_embedder):
    """Test local summarizer fallback."""
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    prompt = "Context:\nThis is some context text.\n\nQuery: What is this about?"
    summary = rag._local_summarizer(prompt)
    
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_rag_hierarchical_summarize(mock_vectorstore, mock_embedder):
    """Test hierarchical summarization."""
    chunks = [
        {
            "chunk_id": "chunk_1",
            "start": 0.0,
            "end": 10.0,
            "text": "First chunk about machine learning."
        },
        {
            "chunk_id": "chunk_2",
            "start": 10.0,
            "end": 20.0,
            "text": "Second chunk about neural networks."
        },
    ]
    
    rag = RAGOrchestrator(
        vectorstore=mock_vectorstore,
        embedder=mock_embedder,
        llm_provider="local"
    )
    
    result = rag.hierarchical_summarize(chunks, video_id="video_1")
    
    assert "micro_summaries" in result
    assert "meta_summary" in result
    assert "video_id" in result
    assert result["video_id"] == "video_1"
    assert len(result["micro_summaries"]) == len(chunks)
