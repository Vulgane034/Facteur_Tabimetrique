"""
Pydantic request schemas for Facteur Tabimetrique API.

Provides request validation with Pydantic v2 including field validators
and JSON schema examples for API documentation.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TrainRequest(BaseModel):
    """Request schema for model training.
    
    Trains a new Facteur Tabimetrique weight model on provided data.
    
    Attributes:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target variable (list of floats), shape (n_samples,)
        feature_names: Optional names for features
        hidden_units: Number of units in first hidden layer of MLP
        dropout_rate: Dropout rate for regularization (0-1)
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        validation_split: Validation split ratio (0-1)
        early_stopping: Whether to use early stopping
        patience: Number of epochs patience for early stopping
    """
    
    X: List[List[float]] = Field(
        ...,
        description="Feature matrix",
        min_items=2,
    )
    y: List[float] = Field(
        ...,
        description="Target variable",
        min_items=2,
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Optional feature names",
    )
    hidden_units: int = Field(
        default=32,
        description="Units in first hidden layer",
        ge=16,
        le=256,
    )
    dropout_rate: float = Field(
        default=0.2,
        description="Dropout rate",
        ge=0.0,
        le=0.9,
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate",
        gt=0,
        le=0.1,
    )
    epochs: int = Field(
        default=100,
        description="Number of training epochs",
        ge=1,
        le=1000,
    )
    validation_split: float = Field(
        default=0.2,
        description="Validation split ratio",
        ge=0.05,
        le=0.5,
    )
    early_stopping: bool = Field(
        default=True,
        description="Enable early stopping",
    )
    patience: int = Field(
        default=10,
        description="Early stopping patience",
        ge=1,
        le=100,
    )
    
    @field_validator("X", "y")
    @classmethod
    def validate_data_not_empty(cls, v):
        """Validate that X and y are not empty."""
        if not v or (isinstance(v, list) and len(v) < 2):
            raise ValueError("Must contain at least 2 samples")
        return v
    
    @field_validator("X")
    @classmethod
    def validate_X_shape(cls, v):
        """Validate X is rectangular."""
        if isinstance(v, list) and len(v) > 0:
            first_len = len(v[0])
            if not all(len(row) == first_len for row in v):
                raise ValueError("All X rows must have same length")
        return v
    
    @field_validator("X", "y")
    @classmethod
    def validate_X_y_match(cls, v, info):
        """Validate X and y have matching sample count."""
        if info.data.get("X") and info.field_name == "y":
            if len(info.data["X"]) != len(v):
                raise ValueError("X and y must have same number of samples")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "X": [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                ],
                "y": [10.5, 15.2, 20.1, 25.3],
                "feature_names": ["income", "age", "experience"],
                "hidden_units": 32,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 100,
                "validation_split": 0.2,
                "early_stopping": True,
                "patience": 10,
            }
        }


class ScoreRequest(BaseModel):
    """Request schema for computing FT scores.
    
    Computes Facteur Tabimetrique scores using an already trained model.
    
    Attributes:
        model_id: Identifier of trained model
        X: Feature matrix to score
        y: Target variable
        use_tanh: Whether to apply tanh activation to raw scores
    """
    
    model_id: str = Field(
        ...,
        description="Trained model identifier",
        min_length=1,
        max_length=100,
    )
    X: List[List[float]] = Field(
        ...,
        description="Feature matrix",
        min_items=1,
    )
    y: List[float] = Field(
        ...,
        description="Target variable",
        min_items=1,
    )
    use_tanh: bool = Field(
        default=True,
        description="Apply tanh activation to scores",
    )
    
    @field_validator("X", "y")
    @classmethod
    def validate_data_not_empty(cls, v):
        """Validate that X and y are not empty."""
        if not v or len(v) < 1:
            raise ValueError("Must contain at least 1 sample")
        return v
    
    @field_validator("X")
    @classmethod
    def validate_X_shape(cls, v):
        """Validate X is rectangular."""
        if isinstance(v, list) and len(v) > 0:
            first_len = len(v[0])
            if not all(len(row) == first_len for row in v):
                raise ValueError("All X rows must have same length")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_id": "trained_model_1",
                "X": [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                ],
                "y": [10.5, 15.2, 20.1],
                "use_tanh": True,
            }
        }


class SelectRequest(BaseModel):
    """Request schema for feature selection.
    
    Selects features based on FT score threshold.
    
    Attributes:
        model_id: Identifier of trained model
        X: Feature matrix
        feature_names: Optional feature names
        threshold: Selection threshold (0-1)
    """
    
    model_id: str = Field(
        ...,
        description="Trained model identifier",
        min_length=1,
        max_length=100,
    )
    X: List[List[float]] = Field(
        ...,
        description="Feature matrix",
        min_items=1,
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Optional feature names",
    )
    threshold: float = Field(
        default=0.3,
        description="Selection threshold",
        ge=0.0,
        le=1.0,
    )
    
    @field_validator("X")
    @classmethod
    def validate_X_shape(cls, v):
        """Validate X is rectangular."""
        if isinstance(v, list) and len(v) > 0:
            first_len = len(v[0])
            if not all(len(row) == first_len for row in v):
                raise ValueError("All X rows must have same length")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_id": "trained_model_1",
                "X": [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                ],
                "feature_names": ["income", "age", "experience"],
                "threshold": 0.3,
            }
        }


class PipelineRequest(BaseModel):
    """Request schema for complete pipeline (train + score + select).
    
    Executes full Facteur Tabimetrique pipeline in one request.
    
    Attributes:
        X: Feature matrix
        y: Target variable
        feature_names: Optional feature names
        threshold: Selection threshold
        epochs: Training epochs
        hidden_units: MLP hidden units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        validation_split: Validation split
        early_stopping: Enable early stopping
        patience: Early stopping patience
    """
    
    X: List[List[float]] = Field(
        ...,
        description="Feature matrix",
        min_items=2,
    )
    y: List[float] = Field(
        ...,
        description="Target variable",
        min_items=2,
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Optional feature names",
    )
    threshold: float = Field(
        default=0.3,
        description="Selection threshold",
        ge=0.0,
        le=1.0,
    )
    epochs: int = Field(
        default=100,
        description="Training epochs",
        ge=1,
        le=1000,
    )
    hidden_units: int = Field(
        default=32,
        description="Hidden layer units",
        ge=16,
        le=256,
    )
    dropout_rate: float = Field(
        default=0.2,
        description="Dropout rate",
        ge=0.0,
        le=0.9,
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate",
        gt=0,
        le=0.1,
    )
    validation_split: float = Field(
        default=0.2,
        description="Validation split",
        ge=0.05,
        le=0.5,
    )
    early_stopping: bool = Field(
        default=True,
        description="Enable early stopping",
    )
    patience: int = Field(
        default=10,
        description="Early stopping patience",
        ge=1,
        le=100,
    )
    
    @field_validator("X", "y")
    @classmethod
    def validate_data_not_empty(cls, v):
        """Validate that X and y are not empty."""
        if not v or len(v) < 2:
            raise ValueError("Must contain at least 2 samples")
        return v
    
    @field_validator("X")
    @classmethod
    def validate_X_shape(cls, v):
        """Validate X is rectangular."""
        if isinstance(v, list) and len(v) > 0:
            first_len = len(v[0])
            if not all(len(row) == first_len for row in v):
                raise ValueError("All X rows must have same length")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "X": [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                ],
                "y": [10.5, 15.2, 20.1, 25.3],
                "feature_names": ["income", "age", "experience"],
                "threshold": 0.3,
                "epochs": 100,
                "hidden_units": 32,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "early_stopping": True,
                "patience": 10,
            }
        }


class CompareRequest(BaseModel):
    """Request schema for method comparison.
    
    Compares FT scores with Pearson, Spearman, and Distance Correlation.
    
    Attributes:
        X: Feature matrix
        y: Target variable
        feature_names: Optional feature names
    """
    
    X: List[List[float]] = Field(
        ...,
        description="Feature matrix",
        min_items=2,
    )
    y: List[float] = Field(
        ...,
        description="Target variable",
        min_items=2,
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Optional feature names",
    )
    
    @field_validator("X", "y")
    @classmethod
    def validate_data_not_empty(cls, v):
        """Validate that X and y are not empty."""
        if not v or len(v) < 2:
            raise ValueError("Must contain at least 2 samples")
        return v
    
    @field_validator("X")
    @classmethod
    def validate_X_shape(cls, v):
        """Validate X is rectangular."""
        if isinstance(v, list) and len(v) > 0:
            first_len = len(v[0])
            if not all(len(row) == first_len for row in v):
                raise ValueError("All X rows must have same length")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "X": [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                ],
                "y": [10.5, 15.2, 20.1, 25.3],
                "feature_names": ["income", "age", "experience"],
            }
        }
