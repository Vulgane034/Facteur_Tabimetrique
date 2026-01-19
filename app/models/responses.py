"""
Pydantic response schemas for Facteur Tabimetrique API.

Provides response validation and automatic documentation for all API endpoints.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Base Response Classes
# ============================================================================

class BaseResponse(BaseModel):
    """Base response with status and timestamp.
    
    Attributes:
        status: Response status (success, error, etc.)
        timestamp: ISO format timestamp (auto-generated)
    """
    
    status: str = Field(
        default="success",
        description="Response status",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp (UTC)",
    )
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Training Response
# ============================================================================

class TrainResponse(BaseResponse):
    """Response schema for model training.
    
    Attributes:
        model_id: Trained model identifier
        message: Human-readable training result message
        training_info: Dictionary with training metrics
            - n_features: Number of features
            - n_samples: Number of samples
            - final_loss: Final training loss
            - final_val_loss: Final validation loss
            - epochs_trained: Number of epochs completed
            - early_stopped: Whether early stopping was triggered
    """
    
    model_id: str = Field(
        ...,
        description="Trained model identifier",
    )
    message: str = Field(
        ...,
        description="Training result message",
    )
    training_info: Dict[str, Any] = Field(
        ...,
        description="Training metrics and information",
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-01-18T10:30:00",
                "model_id": "model_1",
                "message": "Model trained successfully on 100 samples",
                "training_info": {
                    "n_features": 5,
                    "n_samples": 100,
                    "final_loss": 0.0234,
                    "final_val_loss": 0.0267,
                    "epochs_trained": 95,
                    "early_stopped": True,
                },
            }
        }


# ============================================================================
# Scoring Response
# ============================================================================

class ScoreResponse(BaseResponse):
    """Response schema for FT score computation.
    
    Attributes:
        model_id: Model identifier used for scoring
        scores: List of FT scores for each feature
        statistics: Dictionary with score statistics
            - mean: Mean of scores
            - std: Standard deviation
            - min: Minimum score
            - max: Maximum score
            - median: Median score
    """
    
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    scores: List[float] = Field(
        ...,
        description="FT scores for each feature",
    )
    statistics: Dict[str, float] = Field(
        ...,
        description="Score statistics",
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-01-18T10:30:00",
                "model_id": "model_1",
                "scores": [0.85, 0.42, 0.91, 0.23, 0.67],
                "statistics": {
                    "mean": 0.616,
                    "std": 0.281,
                    "min": 0.23,
                    "max": 0.91,
                    "median": 0.67,
                },
            }
        }


# ============================================================================
# Feature Selection Response
# ============================================================================

class SelectResponse(BaseResponse):
    """Response schema for feature selection.
    
    Attributes:
        model_id: Model identifier
        selected_features: Names of selected features
        selected_indices: Column indices of selected features
        n_selected: Number of selected features
        n_total: Total number of features
        threshold: Threshold used for selection
    """
    
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    selected_features: List[str] = Field(
        ...,
        description="Names of selected features",
    )
    selected_indices: List[int] = Field(
        ...,
        description="Column indices of selected features",
    )
    n_selected: int = Field(
        ...,
        description="Number of selected features",
        ge=0,
    )
    n_total: int = Field(
        ...,
        description="Total number of features",
        ge=1,
    )
    threshold: float = Field(
        ...,
        description="Selection threshold used",
        ge=0.0,
        le=1.0,
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-01-18T10:30:00",
                "model_id": "model_1",
                "selected_features": ["feature_1", "feature_3", "feature_5"],
                "selected_indices": [0, 2, 4],
                "n_selected": 3,
                "n_total": 5,
                "threshold": 0.5,
            }
        }


# ============================================================================
# Feature Importance Response
# ============================================================================

class ImportanceRow(BaseModel):
    """Row in importance table.
    
    Attributes:
        feature: Feature name
        ft_score: Facteur Tabimetrique score
        abs_score: Absolute value of score
        tau: Kendall correlation
        zeta: Pearson correlation
        dcor: Distance correlation
        C: Transitive dependence
        S_lin: Linearity meta-feature
        S_norm: Normality meta-feature
        S_out: Outlier sensitivity meta-feature
    """
    
    feature: str
    ft_score: float
    abs_score: float
    tau: float
    zeta: float
    dcor: float
    C: float
    S_lin: float
    S_norm: float
    S_out: float


class ImportanceResponse(BaseResponse):
    """Response schema for feature importance report.
    
    Attributes:
        model_id: Model identifier
        importance_table: List of importance rows sorted by abs_score descending
    """
    
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    importance_table: List[ImportanceRow] = Field(
        ...,
        description="Feature importance details sorted by score",
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-01-18T10:30:00",
                "model_id": "model_1",
                "importance_table": [
                    {
                        "feature": "feature_1",
                        "ft_score": 0.91,
                        "abs_score": 0.91,
                        "tau": 0.88,
                        "zeta": 0.93,
                        "dcor": 0.90,
                        "C": 0.02,
                        "S_lin": 0.86,
                        "S_norm": 1.0,
                        "S_out": 0.05,
                    },
                    {
                        "feature": "feature_3",
                        "ft_score": 0.85,
                        "abs_score": 0.85,
                        "tau": 0.82,
                        "zeta": 0.87,
                        "dcor": 0.84,
                        "C": 0.03,
                        "S_lin": 0.76,
                        "S_norm": 1.0,
                        "S_out": 0.08,
                    },
                ],
            }
        }


# ============================================================================
# Method Comparison Response
# ============================================================================

class ComparisonRow(BaseModel):
    """Row in comparison table.
    
    Attributes:
        method: Method name (e.g., "Pearson", "Spearman", "Distance Correlation")
        scores: Correlation scores for each feature
        rank_correlation: Correlation of method rankings with FT rankings
        mean_score: Mean of method scores
        median_score: Median of method scores
    """
    
    method: str
    scores: List[float]
    rank_correlation: float
    mean_score: float
    median_score: float


class CompareResponse(BaseResponse):
    """Response schema for method comparison.
    
    Attributes:
        comparison_table: List of method comparison rows
        ft_scores: Original FT scores for reference
        n_features: Number of features compared
    """
    
    comparison_table: List[ComparisonRow] = Field(
        ...,
        description="Comparison of methods",
    )
    ft_scores: List[float] = Field(
        ...,
        description="FT scores for reference",
    )
    n_features: int = Field(
        ...,
        description="Number of features",
        ge=1,
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2026-01-18T10:30:00",
                "comparison_table": [
                    {
                        "method": "Pearson",
                        "scores": [0.93, 0.42, 0.88, 0.25, 0.71],
                        "rank_correlation": 0.95,
                        "mean_score": 0.638,
                        "median_score": 0.71,
                    },
                    {
                        "method": "Spearman",
                        "scores": [0.91, 0.40, 0.86, 0.22, 0.68],
                        "rank_correlation": 0.92,
                        "mean_score": 0.614,
                        "median_score": 0.68,
                    },
                    {
                        "method": "Distance Correlation",
                        "scores": [0.89, 0.44, 0.90, 0.28, 0.65],
                        "rank_correlation": 0.98,
                        "mean_score": 0.632,
                        "median_score": 0.65,
                    },
                ],
                "ft_scores": [0.85, 0.42, 0.91, 0.23, 0.67],
                "n_features": 5,
            }
        }


# ============================================================================
# Error Response
# ============================================================================

class ErrorResponse(BaseModel):
    """Response schema for errors.
    
    Attributes:
        status: Status ("error")
        message: Error message
        detail: Optional detailed error information
        timestamp: ISO format timestamp
    """
    
    status: str = Field(
        default="error",
        description="Response status",
    )
    message: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp (UTC)",
    )
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Model not found",
                "detail": "Model with id 'model_1' does not exist in storage",
                "timestamp": "2026-01-18T10:30:00",
            }
        }
