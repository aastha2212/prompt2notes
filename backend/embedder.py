"""
Embedding generation using sentence-transformers with batching support.
"""

import logging
from typing import List, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class Embedder:
    """
    Wrapper for sentence-transformers with batched encoding.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: Sentence transformer model name
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts (batch_size={self.batch_size})")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings: shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string.
        
        Args:
            text: Text to encode
            
        Returns:
            Single embedding vector (1D array)
        """
        embeddings = self.encode([text])
        return embeddings[0]
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        # all-MiniLM-L6-v2 has 384 dimensions
        return self.model.get_sentence_embedding_dimension()

