"""
In-memory model storage management with LRU eviction and thread-safety.

Provides thread-safe storage for FacteurTabimetrique models with automatic
LRU (Least Recently Used) eviction when max capacity is reached.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import logging
from collections import OrderedDict
from datetime import datetime
from threading import Lock
from typing import Dict, Optional

from app.core.facteur_tabimetrique import FacteurTabimetrique


logger = logging.getLogger(__name__)


class ModelMetadata:
    """Metadata for stored model.
    
    Attributes:
        model_id: Model identifier
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        n_features: Number of features (if trained)
        trained: Whether model has been trained
    """
    
    def __init__(self, model_id: str):
        """Initialize metadata.
        
        Args:
            model_id: Model identifier
        """
        self.model_id = model_id
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.n_features: Optional[int] = None
        self.trained = False
    
    def touch(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata
        """
        return {
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "n_features": self.n_features,
            "trained": self.trained,
        }


class ModelStorage:
    """Thread-safe in-memory model storage with LRU eviction.
    
    Provides automatic LRU eviction when max capacity is reached.
    All operations are thread-safe via internal Lock.
    
    Attributes:
        max_models: Maximum number of models to keep in memory
        _models: OrderedDict for LRU tracking
        _metadata: Metadata for each model
        _lock: Thread-safe lock
    """
    
    def __init__(self, max_models: int = 50):
        """Initialize storage.
        
        Args:
            max_models: Maximum number of models to keep in memory (default 50)
            
        Raises:
            ValueError: If max_models < 1
        """
        if max_models < 1:
            raise ValueError("max_models must be >= 1")
        
        self.max_models = max_models
        self._models: OrderedDict[str, FacteurTabimetrique] = OrderedDict()
        self._metadata: Dict[str, ModelMetadata] = {}
        self._lock = Lock()
        
        logger.info(f"ModelStorage initialized with max_models={max_models}")
    
    def save(self, model_id: str, model: FacteurTabimetrique) -> str:
        """Save model to storage with LRU eviction if needed.
        
        If storage is at max capacity, removes least recently used model.
        Updates model access timestamp.
        
        Args:
            model_id: Model identifier (string key)
            model: FacteurTabimetrique model instance
            
        Returns:
            model_id: The model identifier
            
        Raises:
            ValueError: If model_id is empty or model is None
        """
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if model is None:
            raise ValueError("model cannot be None")
        
        with self._lock:
            # If model exists, remove it first to update LRU position
            if model_id in self._models:
                del self._models[model_id]
                logger.debug(f"Updated existing model: {model_id}")
            
            # Evict LRU model if at max capacity
            elif len(self._models) >= self.max_models:
                lru_model_id, _ = self._models.popitem(last=False)
                del self._metadata[lru_model_id]
                logger.info(f"LRU eviction: removed model '{lru_model_id}' "
                           f"(storage at {self.max_models} capacity)")
            
            # Save new model
            self._models[model_id] = model
            
            # Update or create metadata
            if model_id not in self._metadata:
                self._metadata[model_id] = ModelMetadata(model_id)
            else:
                self._metadata[model_id].touch()
            
            logger.debug(f"Saved model '{model_id}' (total: {len(self._models)}/{self.max_models})")
        
        return model_id
    
    def get(self, model_id: str) -> Optional[FacteurTabimetrique]:
        """Get model by ID and mark as recently used.
        
        Updates the model's last accessed timestamp (LRU tracking).
        
        Args:
            model_id: Model identifier
            
        Returns:
            FacteurTabimetrique instance or None if not found
        """
        with self._lock:
            if model_id not in self._models:
                logger.debug(f"Model not found: {model_id}")
                return None
            
            # Move to end (most recently used)
            self._models.move_to_end(model_id)
            
            # Update metadata
            self._metadata[model_id].touch()
            
            logger.debug(f"Retrieved model: {model_id}")
            return self._models[model_id]
    
    def delete(self, model_id: str) -> bool:
        """Delete model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if model_id not in self._models:
                logger.debug(f"Cannot delete: model not found '{model_id}'")
                return False
            
            del self._models[model_id]
            del self._metadata[model_id]
            
            logger.info(f"Deleted model: {model_id}")
            return True
    
    def list_all(self) -> Dict[str, Dict]:
        """List all models with metadata.
        
        Returns models ordered by access time (most recent first).
        
        Returns:
            Dictionary mapping model_id to model info dict containing:
                - model_id: Model identifier
                - created_at: Creation timestamp
                - last_accessed: Last access timestamp
                - n_features: Number of features (if trained)
                - trained: Whether model is trained
        """
        with self._lock:
            result = {}
            # Reverse iteration to show most recent first
            for model_id in reversed(self._models):
                result[model_id] = self._metadata[model_id].to_dict()
            
            logger.debug(f"Listed {len(result)} models")
            return result
    
    def count(self) -> int:
        """Get number of models in storage.
        
        Returns:
            Number of models currently stored
        """
        with self._lock:
            return len(self._models)
    
    def clear(self) -> None:
        """Clear all models from storage.
        
        Removes all models and metadata.
        """
        with self._lock:
            count = len(self._models)
            self._models.clear()
            self._metadata.clear()
            
            logger.info(f"Cleared storage: removed {count} models")
    
    def get_stats(self) -> Dict[str, any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage information:
                - total_models: Number of models currently stored
                - max_models: Maximum capacity
                - available_slots: Remaining capacity
                - utilization: Percentage utilization (0-100)
                - oldest_model: Oldest model by creation time
                - newest_model: Most recent model by last access
        """
        with self._lock:
            total = len(self._models)
            available = self.max_models - total
            utilization = (total / self.max_models * 100) if self.max_models > 0 else 0
            
            oldest_model = None
            newest_model = None
            
            if self._metadata:
                oldest_model = min(
                    self._metadata.items(),
                    key=lambda x: x[1].created_at
                )[0]
                newest_model = max(
                    self._metadata.items(),
                    key=lambda x: x[1].last_accessed
                )[0]
            
            stats = {
                "total_models": total,
                "max_models": self.max_models,
                "available_slots": available,
                "utilization": round(utilization, 2),
                "oldest_model": oldest_model,
                "newest_model": newest_model,
            }
            
            logger.debug(f"Storage stats: {total}/{self.max_models} models "
                        f"({utilization:.1f}% utilization)")
            return stats
    
    def exists(self, model_id: str) -> bool:
        """Check if model exists.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model exists, False otherwise
        """
        with self._lock:
            return model_id in self._models
    
    def update_metadata(self, model_id: str, trained: bool = None, 
                       n_features: int = None) -> None:
        """Update model metadata.
        
        Args:
            model_id: Model identifier
            trained: Whether model is trained
            n_features: Number of features
        """
        with self._lock:
            if model_id not in self._metadata:
                raise ValueError(f"Model not found: {model_id}")
            
            metadata = self._metadata[model_id]
            if trained is not None:
                metadata.trained = trained
            if n_features is not None:
                metadata.n_features = n_features
            
            logger.debug(f"Updated metadata for model: {model_id}")


# Global model storage instance with default configuration
model_storage = ModelStorage(max_models=50)
