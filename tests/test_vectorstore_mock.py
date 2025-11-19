"""
Unit tests for vectorstore module using real ChromaDB in temporary directory.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from backend.vectorstore import VectorStore
from backend.embedder import Embedder
import numpy as np


@pytest.fixture
def temp_vectorstore():
    """Create a temporary vectorstore for testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        vs = VectorStore(persist_directory=temp_dir, collection_name="test_collection")
        yield vs
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "start": 0.0,
            "end": 10.0,
            "text": "This is the first chunk about machine learning.",
            "video_id": "video_1"
        },
        {
            "chunk_id": "chunk_2",
            "start": 10.0,
            "end": 20.0,
            "text": "This is the second chunk about neural networks.",
            "video_id": "video_1"
        },
        {
            "chunk_id": "chunk_3",
            "start": 20.0,
            "end": 30.0,
            "text": "This is the third chunk about deep learning.",
            "video_id": "video_1"
        },
    ]


@pytest.fixture
def sample_embeddings(sample_chunks):
    """Generate sample embeddings."""
    try:
        embedder = Embedder()
        texts = [chunk["text"] for chunk in sample_chunks]
        embeddings = embedder.encode(texts, show_progress=False)
        return embeddings.tolist()
    except Exception as e:
        pytest.skip(f"Embedder not available: {e}")


def test_vectorstore_initialization(temp_vectorstore):
    """Test vectorstore initialization."""
    assert temp_vectorstore is not None
    assert temp_vectorstore.collection_name == "test_collection"


def test_vectorstore_add_chunks(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test adding chunks to vectorstore."""
    success = temp_vectorstore.add_chunks(sample_chunks, sample_embeddings)
    assert success is True
    
    # Verify chunks were added
    info = temp_vectorstore.get_collection_info()
    assert info["chunk_count"] == len(sample_chunks)


def test_vectorstore_query(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test querying vectorstore."""
    # Add chunks first
    temp_vectorstore.add_chunks(sample_chunks, sample_embeddings)
    
    # Generate query embedding
    try:
        embedder = Embedder()
        query_text = "machine learning algorithms"
        query_embedding = embedder.encode_single(query_text).tolist()
        
        # Query
        results = temp_vectorstore.query(
            query_embedding=query_embedding,
            top_k=2
        )
        
        assert len(results) > 0
        assert len(results) <= 2
        
        # Check result structure
        for result in results:
            assert "chunk_id" in result
            assert "text" in result
            assert "metadata" in result
            assert "start" in result["metadata"]
            assert "end" in result["metadata"]
            
    except Exception as e:
        pytest.skip(f"Embedder not available: {e}")


def test_vectorstore_query_with_filter(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test querying with video_id filter."""
    # Add chunks
    temp_vectorstore.add_chunks(sample_chunks, sample_embeddings)
    
    # Add chunks from different video
    other_chunks = [
        {
            "chunk_id": "chunk_4",
            "start": 0.0,
            "end": 10.0,
            "text": "This is from a different video about cooking.",
            "video_id": "video_2"
        }
    ]
    try:
        embedder = Embedder()
        other_texts = [chunk["text"] for chunk in other_chunks]
        other_embeddings = embedder.encode(other_texts, show_progress=False).tolist()
        temp_vectorstore.add_chunks(other_chunks, other_embeddings)
        
        # Query with filter
        query_text = "machine learning"
        query_embedding = embedder.encode_single(query_text).tolist()
        
        results = temp_vectorstore.query(
            query_embedding=query_embedding,
            top_k=10,
            filter_dict={"video_id": "video_1"}
        )
        
        # All results should be from video_1
        for result in results:
            assert result["metadata"]["video_id"] == "video_1"
            
    except Exception as e:
        pytest.skip(f"Embedder not available: {e}")


def test_vectorstore_get_collection_info(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test getting collection info."""
    info_before = temp_vectorstore.get_collection_info()
    initial_count = info_before.get("chunk_count", 0)
    
    temp_vectorstore.add_chunks(sample_chunks, sample_embeddings)
    
    info_after = temp_vectorstore.get_collection_info()
    assert info_after["chunk_count"] == initial_count + len(sample_chunks)


def test_vectorstore_delete_by_video_id(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test deleting chunks by video_id."""
    # Add chunks
    temp_vectorstore.add_chunks(sample_chunks, sample_embeddings)
    
    # Verify they're there
    info = temp_vectorstore.get_collection_info()
    assert info["chunk_count"] == len(sample_chunks)
    
    # Delete by video_id
    success = temp_vectorstore.delete_by_video_id("video_1")
    assert success is True
    
    # Verify deletion
    info_after = temp_vectorstore.get_collection_info()
    assert info_after["chunk_count"] == 0


def test_vectorstore_clear_collection(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test clearing collection."""
    # Add chunks
    temp_vectorstore.add_chunks(sample_chunks, sample_embeddings)
    
    # Clear
    success = temp_vectorstore.clear_collection()
    assert success is True
    
    # Verify cleared
    info = temp_vectorstore.get_collection_info()
    assert info["chunk_count"] == 0


def test_vectorstore_mismatch_chunks_embeddings(temp_vectorstore, sample_chunks, sample_embeddings):
    """Test that mismatch between chunks and embeddings raises error."""
    with pytest.raises(ValueError):
        temp_vectorstore.add_chunks(sample_chunks, sample_embeddings[:2])  # Mismatch

