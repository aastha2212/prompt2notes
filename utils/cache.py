"""
Simple filesystem cache for transcripts and embeddings.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class FileSystemCache:
    """
    Simple filesystem-based cache for storing transcripts and embeddings.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize cache directory.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, key: str, suffix: str = ".pkl") -> Path:
        """Get cache file path for a given key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}{suffix}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value.
        
        Args:
            key: Cache key (typically file hash)
            
        Returns:
            Cached value or None if not found
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {key}: {e}")
                return None
        return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save cache for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if cache entry exists."""
        return self._get_cache_path(key).exists()
    
    def clear(self, key: Optional[str] = None) -> bool:
        """
        Clear cache entry or entire cache.
        
        Args:
            key: Specific key to clear, or None to clear all
            
        Returns:
            True if successful
        """
        if key:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                return True
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            return True
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get JSON metadata for a cache entry."""
        metadata_path = self._get_cache_path(key, ".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {key}: {e}")
        return None
    
    def set_metadata(self, key: str, metadata: Dict) -> bool:
        """Store JSON metadata for a cache entry."""
        metadata_path = self._get_cache_path(key, ".json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata for {key}: {e}")
            return False

