"""
ChromaDB wrapper for vector storage and retrieval.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not available. Install with: pip install chromadb")


class VectorStore:
    """
    Wrapper for ChromaDB vector storage.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "prompt2notes"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. "
                "Install with: pip install chromadb"
            )
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized: {persist_directory}, collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dicts with keys: chunk_id, start, end, text, video_id
            embeddings: List of embedding vectors (lists of floats)
            
        Returns:
            True if successful
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embedding_list = []
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
                ids.append(chunk_id)
                documents.append(chunk.get("text", ""))
                metadatas.append({
                    "video_id": chunk.get("video_id", "unknown"),
                    "chunk_id": chunk_id,
                    "start": chunk.get("start", 0.0),
                    "end": chunk.get("end", 0.0)
                })
                embedding_list.append(embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_list
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"video_id": "video1"})
            
        Returns:
            List of results with keys: text, metadata, distance
        """
        try:
            # Build where clause for filtering
            where = filter_dict if filter_dict else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "chunk_id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "chunk_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def delete_by_video_id(self, video_id: str) -> bool:
        """
        Delete all chunks for a specific video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            True if successful
        """
        try:
            # ChromaDB doesn't have a direct delete by metadata, so we need to query first
            results = self.collection.get(where={"video_id": video_id})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for video_id: {video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

