"""
FastAPI routes for Facteur Tabimetrique API.

Provides 9 endpoints for training, scoring, feature selection, and model management
with comprehensive error handling and validation.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import io
import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import pandas as pd
import numpy as np

from app.models.requests import (
    TrainRequest,
    ScoreRequest,
    SelectRequest,
    PipelineRequest,
    CompareRequest,
)
from app.models.responses import (
    BaseResponse,
    TrainResponse,
    ScoreResponse,
    SelectResponse,
    ImportanceResponse,
    CompareResponse,
    ErrorResponse,
)
from app.services.ft_service import ft_service
from app.services.storage import model_storage


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Facteurs Tabimétriques"])


# ============================================================================
# Training Endpoints
# ============================================================================

@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=201,
    summary="Entraîner un nouveau modèle",
    description="Entraîne un nouveau modèle Facteur Tabimetrique et le stocke en mémoire",
)
async def train_model(request: TrainRequest) -> TrainResponse:
    """Train a new Facteur Tabimetrique model.
    
    Creates a new model instance, trains it on provided data,
    and stores it for future reference.
    
    Args:
        request: TrainRequest with features X, target y, and hyperparameters
        
    Returns:
        TrainResponse with model_id, message, and training_info
        
    Raises:
        HTTPException: 400 if data validation fails, 500 if training error
    """
    try:
        logger.info("POST /train: Starting model training")
        
        # Train model
        model_id, model, training_info = ft_service.train_model(
            X=request.X,
            y=request.y,
            feature_names=request.feature_names,
            hidden_units=request.hidden_units,
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            validation_split=request.validation_split,
            early_stopping=request.early_stopping,
            patience=request.patience,
        )
        
        # Update metadata
        model_storage.update_metadata(
            model_id,
            trained=True,
            n_features=len(request.feature_names or []) if request.feature_names else len(request.X[0]),
        )
        
        response = TrainResponse(
            status="success",
            model_id=model_id,
            message=f"Model '{model_id}' trained successfully on {training_info['n_samples']} samples with {training_info['n_features']} features",
            training_info=training_info,
        )
        
        logger.info(f"POST /train: ✓ Model '{model_id}' trained successfully")
        return response
        
    except ValueError as e:
        logger.error(f"POST /train: ✗ Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"POST /train: ✗ Training error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}",
        )


# ============================================================================
# Scoring Endpoints
# ============================================================================

@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=200,
    summary="Calculer les scores FT",
    description="Calcule les scores Facteur Tabimetrique pour les features",
)
async def score_features(request: ScoreRequest) -> ScoreResponse:
    """Compute Facteur Tabimetrique scores for features.
    
    Uses an existing trained model to compute feature importance scores.
    
    Args:
        request: ScoreRequest with model_id, features X, target y, use_tanh flag
        
    Returns:
        ScoreResponse with scores and statistics
        
    Raises:
        HTTPException: 404 if model not found, 400 if data invalid, 500 if error
    """
    try:
        logger.info(f"POST /score: Computing scores for model '{request.model_id}'")
        
        # Check model exists
        if not model_storage.exists(request.model_id):
            raise ValueError(f"Model not found: {request.model_id}")
        
        # Compute scores
        scores, statistics = ft_service.compute_scores(
            model_id=request.model_id,
            X=request.X,
            y=request.y,
            use_tanh=request.use_tanh,
        )
        
        response = ScoreResponse(
            status="success",
            model_id=request.model_id,
            scores=scores.tolist(),
            statistics=statistics,
        )
        
        logger.info(f"POST /score: ✓ Computed {len(scores)} scores")
        return response
        
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            logger.error(f"POST /score: ✗ Model not found: {request.model_id}")
            raise HTTPException(status_code=404, detail=error_msg)
        logger.error(f"POST /score: ✗ Validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.error(f"POST /score: ✗ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}",
        )


# ============================================================================
# Feature Selection Endpoints
# ============================================================================

@router.post(
    "/select",
    response_model=SelectResponse,
    status_code=200,
    summary="Sélectionner les features",
    description="Sélectionne les features importantes selon un seuil",
)
async def select_features(request: SelectRequest) -> SelectResponse:
    """Select important features based on FT scores.
    
    Filters features based on a threshold applied to FT scores.
    
    Args:
        request: SelectRequest with model_id, features X, target y, threshold, feature_names
        
    Returns:
        SelectResponse with selected features and indices
        
    Raises:
        HTTPException: 404 if model not found, 400 if data invalid, 500 if error
    """
    try:
        logger.info(f"POST /select: Selecting features from model '{request.model_id}' (threshold={request.threshold})")
        
        # Check model exists
        if not model_storage.exists(request.model_id):
            raise ValueError(f"Model not found: {request.model_id}")
        
        # Select features
        df_selected, selected_indices, selected_names = ft_service.select_features(
            model_id=request.model_id,
            X=request.X,
            y=request.y,
            threshold=request.threshold,
            feature_names=request.feature_names,
        )
        
        response = SelectResponse(
            status="success",
            model_id=request.model_id,
            selected_features=selected_names,
            selected_indices=selected_indices,
            n_selected=len(selected_names),
            n_total=len(request.X[0]),
            threshold=request.threshold,
        )
        
        logger.info(f"POST /select: ✓ Selected {len(selected_names)}/{len(request.X[0])} features")
        return response
        
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            logger.error(f"POST /select: ✗ Model not found: {request.model_id}")
            raise HTTPException(status_code=404, detail=error_msg)
        logger.error(f"POST /select: ✗ Validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.error(f"POST /select: ✗ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Feature selection failed: {str(e)}",
        )


# ============================================================================
# Pipeline Endpoints
# ============================================================================

@router.post(
    "/pipeline",
    response_model=Dict,
    status_code=201,
    summary="Pipeline complet",
    description="Entraîne modèle + calcule scores + sélectionne features",
)
async def pipeline(request: PipelineRequest) -> Dict:
    """Complete pipeline: train, score, and select features.
    
    Executes the full Facteur Tabimetrique workflow in one call.
    
    Args:
        request: PipelineRequest with features, target, and all hyperparameters
        
    Returns:
        Dictionary with pipeline results including model_id, scores, selected_features
        
    Raises:
        HTTPException: 400 if data invalid, 500 if error
    """
    try:
        logger.info("POST /pipeline: Starting complete pipeline")
        
        # Train model
        model_id, model, training_info = ft_service.train_model(
            X=request.X,
            y=request.y,
            feature_names=request.feature_names,
            hidden_units=request.hidden_units,
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            validation_split=request.validation_split,
            early_stopping=request.early_stopping,
            patience=request.patience,
        )
        
        # Compute scores
        scores, statistics = ft_service.compute_scores(
            model_id=model_id,
            X=request.X,
            y=request.y,
            use_tanh=True,
        )
        
        # Select features
        df_selected, selected_indices, selected_names = ft_service.select_features(
            model_id=model_id,
            X=request.X,
            y=request.y,
            threshold=request.threshold,
            feature_names=request.feature_names,
        )
        
        # Update metadata
        model_storage.update_metadata(
            model_id,
            trained=True,
            n_features=len(request.feature_names or []) if request.feature_names else len(request.X[0]),
        )
        
        response = {
            "status": "success",
            "model_id": model_id,
            "training_info": training_info,
            "scores": {
                "ft_scores": scores.tolist(),
                "statistics": statistics,
            },
            "selection": {
                "selected_features": selected_names,
                "selected_indices": selected_indices,
                "n_selected": len(selected_names),
                "n_total": len(request.X[0]),
                "threshold": request.threshold,
            },
        }
        
        logger.info(f"POST /pipeline: ✓ Pipeline completed for model '{model_id}'")
        return response
        
    except ValueError as e:
        logger.error(f"POST /pipeline: ✗ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"POST /pipeline: ✗ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}",
        )

