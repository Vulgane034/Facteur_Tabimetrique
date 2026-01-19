"""
Business logic service for Facteur Tabimetrique operations.

Provides high-level interface for training, scoring, feature selection,
and method comparison operations with automatic model storage management.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau


def distance_correlation(x, y):
    """Distance correlation implementation using double centered distances.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
        
    Returns:
        Distance correlation coefficient between 0 and 1
    """
    x = np.asarray(x).reshape(-1, 1) if np.asarray(x).ndim == 1 else np.asarray(x)
    y = np.asarray(y).reshape(-1, 1) if np.asarray(y).ndim == 1 else np.asarray(y)
    
    n = len(x)
    if n < 2:
        return 0.0
    
    # Euclidean distances
    dx = np.sqrt(np.sum((x[:, np.newaxis, :] - x[np.newaxis, :, :]) ** 2, axis=2))
    dy = np.sqrt(np.sum((y[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2))
    
    # Double centered distances
    row_mean_x = dx.mean(axis=1, keepdims=True)
    col_mean_x = dx.mean(axis=0, keepdims=True)
    grand_mean_x = dx.mean()
    aij = dx - row_mean_x - col_mean_x + grand_mean_x
    
    row_mean_y = dy.mean(axis=1, keepdims=True)
    col_mean_y = dy.mean(axis=0, keepdims=True)
    grand_mean_y = dy.mean()
    bij = dy - row_mean_y - col_mean_y + grand_mean_y
    
    # V-statistics
    V_xy_sq = np.sum(aij * bij) / (n ** 2)
    V_xx_sq = np.sum(aij ** 2) / (n ** 2)
    V_yy_sq = np.sum(bij ** 2) / (n ** 2)
    
    if V_xx_sq == 0 or V_yy_sq == 0:
        return 0.0
    
    return np.sqrt(V_xy_sq ** 2 / (V_xx_sq * V_yy_sq))


from app.core.facteur_tabimetrique import FacteurTabimetrique
from app.services.storage import model_storage


logger = logging.getLogger(__name__)


class FTService:
    """Service layer for Facteur Tabimetrique operations.
    
    Provides static methods for:
    - Model training with automatic storage
    - Feature importance computation
    - Feature selection
    - Method comparison (Pearson, Spearman, Distance Correlation)
    
    All operations are thread-safe and integrate with ModelStorage.
    """
    
    @staticmethod
    def train_model(
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
        hidden_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
    ) -> Tuple[str, FacteurTabimetrique, Dict]:
        """Train a new Facteur Tabimetrique model.
        
        Validates input dimensions, creates model, trains it, stores it,
        and returns model_id for future reference.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: Optional feature names (defaults to ['feature_0', ...])
            hidden_units: MLP hidden layer size (default 32)
            dropout_rate: Dropout rate for MLP (default 0.2)
            learning_rate: Learning rate for MLP (default 0.001)
            epochs: Number of training epochs (default 100)
            validation_split: Validation set proportion (default 0.2)
            early_stopping: Whether to use early stopping (default True)
            patience: Patience for early stopping (default 10)
            
        Returns:
            Tuple of:
                - model_id: Generated UUID string
                - model: Trained FacteurTabimetrique instance
                - training_info: Dictionary with training metrics
                
        Raises:
            ValueError: If X and y dimensions don't match or data is invalid
        """
        logger.info("Starting model training...")
        
        # Convert to numpy arrays
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        
        # Validate dimensions
        if X_array.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X_array.ndim}D")
        if y_array.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y_array.ndim}D")
        
        n_samples, n_features = X_array.shape
        if n_samples != len(y_array):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {n_samples}, y: {len(y_array)}"
            )
        
        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples, got {n_samples}")
        if n_features < 1:
            raise ValueError(f"Need at least 1 feature, got {n_features}")
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"n_features ({n_features})"
            )
        
        try:
            # Create model instance
            model = FacteurTabimetrique(
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                verbose=1,
            )
            
            logger.debug(f"Created model with {n_features} features")
            
            # Train model
            model.train_weight_model(
                X_array,
                y_array,
                epochs=epochs,
                validation_split=validation_split,
                early_stopping=early_stopping,
                patience=patience,
            )
            
            logger.debug("Model training completed")
            
            # Generate unique model ID
            model_id = f"ft_{uuid.uuid4().hex[:12]}"
            
            # Store model
            model_storage.save(model_id, model)
            
            # Prepare training info
            training_info = {
                "n_features": n_features,
                "n_samples": n_samples,
                "feature_names": feature_names,
                "hidden_units": hidden_units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "validation_split": validation_split,
                "early_stopping": early_stopping,
                "patience": patience,
            }
            
            logger.info(f"Model '{model_id}' trained successfully")
            return model_id, model, training_info
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    @staticmethod
    def compute_scores(
        model_id: str,
        X: List[List[float]],
        y: List[float],
        use_tanh: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute Facteur Tabimetrique scores for features.
        
        Args:
            model_id: Model identifier from storage
            X: Features (n_samples, n_features)
            y: Target variable (n_samples,)
            use_tanh: Whether to apply tanh transformation (default True)
            
        Returns:
            Tuple of:
                - scores: Array of FT scores for each feature
                - statistics: Dict with mean, std, min, max, median
                
        Raises:
            ValueError: If model not found or data invalid
        """
        logger.info(f"Computing scores for model '{model_id}'")
        
        # Retrieve model
        model = model_storage.get(model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")
        
        # Convert to numpy arrays
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        
        # Validate
        if X_array.ndim != 2 or y_array.ndim != 1:
            raise ValueError("X must be 2D, y must be 1D")
        
        if X_array.shape[0] != len(y_array):
            raise ValueError("X and y must have same number of samples")
        
        try:
            # Compute scores
            scores = model.compute_tabimetric_scores(
                X_array,
                y_array,
                use_tanh=use_tanh,
            )
            
            # Compute statistics
            statistics = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            }
            
            logger.debug(f"Computed {len(scores)} scores")
            return scores, statistics
            
        except Exception as e:
            logger.error(f"Error computing scores: {str(e)}")
            raise
    
    @staticmethod
    def select_features(
        model_id: str,
        X: List[List[float]],
        y: Optional[List[float]] = None,
        threshold: float = 0.5,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[int], List[str]]:
        """Select features based on FT scores.
        
        Args:
            model_id: Model identifier
            X: Features (n_samples, n_features)
            y: Target variable (required for score computation)
            threshold: Selection threshold (0-1, default 0.5)
            feature_names: Optional feature names
            
        Returns:
            Tuple of:
                - df_selected: DataFrame with selected features
                - selected_indices: List of selected column indices
                - selected_names: List of selected feature names
                
        Raises:
            ValueError: If model not found or data invalid
        """
        logger.info(f"Selecting features for model '{model_id}' (threshold={threshold})")
        
        if y is None:
            raise ValueError("y is required for feature selection")
        
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        
        # Compute scores
        scores, _ = FTService.compute_scores(model_id, X, y, use_tanh=True)
        
        X_array = np.array(X, dtype=np.float32)
        n_samples, n_features = X_array.shape
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Select features based on threshold
        abs_scores = np.abs(scores)
        selected_mask = abs_scores >= threshold
        selected_indices = np.where(selected_mask)[0].tolist()
        
        if len(selected_indices) == 0:
            logger.warning(f"No features selected with threshold {threshold}")
        
        # Create result dataframe
        selected_names = [feature_names[i] for i in selected_indices]
        X_selected = X_array[:, selected_indices]
        
        df_selected = pd.DataFrame(
            X_selected,
            columns=selected_names,
        )
        
        logger.info(f"Selected {len(selected_indices)}/{n_features} features")
        return df_selected, selected_indices, selected_names
    
    @staticmethod
    def get_importance(
        model_id: str,
        X: Optional[List[List[float]]] = None,
        y: Optional[List[float]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get detailed feature importance report.
        
        Args:
            model_id: Model identifier
            X: Features (required for full report)
            y: Target variable (required for full report)
            feature_names: Optional feature names
            
        Returns:
            DataFrame with importance metrics:
                - feature: Feature name
                - ft_score: Facteur Tabimetrique score
                - abs_score: Absolute score
                - tau: Kendall correlation
                - zeta: Pearson correlation
                - dcor: Distance correlation
                - C: Transitive dependence
                - S_lin: Linearity meta-feature
                - S_norm: Normality meta-feature
                - S_out: Outlier sensitivity meta-feature
        """
        logger.info(f"Getting importance report for model '{model_id}'")
        
        if X is None or y is None:
            raise ValueError("X and y are required for importance report")
        
        # Get importance dataframe
        importance_df = model_storage.get(model_id).get_feature_importance()
        
        # Add feature names if provided
        n_features = len(importance_df)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length must match number of features"
            )
        
        importance_df.insert(0, "feature", feature_names)
        
        # Sort by absolute score descending
        importance_df = importance_df.sort_values(
            "abs_score",
            ascending=False,
        ).reset_index(drop=True)
        
        logger.debug(f"Generated importance report with {len(importance_df)} features")
        return importance_df
    
    @staticmethod
    def compare_methods(
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare FT with Pearson, Spearman, and Distance Correlation.
        
        Trains a temporary FT model and compares its feature rankings
        with three alternative methods.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: Optional feature names
            
        Returns:
            DataFrame with comparison results:
                - feature: Feature name
                - ft_score: FT score (tanh transformed)
                - pearson: Pearson correlation
                - spearman: Spearman correlation
                - distance_corr: Distance correlation
                - ft_rank: Ranking by FT
                - pearson_rank: Ranking by Pearson
                - spearman_rank: Ranking by Spearman
                - dcor_rank: Ranking by Distance Correlation
        """
        logger.info("Starting method comparison analysis")
        
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        
        if X_array.ndim != 2 or y_array.ndim != 1:
            raise ValueError("X must be 2D, y must be 1D")
        
        if X_array.shape[0] != len(y_array):
            raise ValueError("X and y must have same number of samples")
        
        n_samples, n_features = X_array.shape
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError("feature_names length must match n_features")
        
        try:
            # Train temporary FT model
            temp_model_id, _, _ = FTService.train_model(
                X_array.tolist(),
                y_array.tolist(),
                feature_names=feature_names,
                epochs=100,
            )
            
            # Compute FT scores
            ft_scores, _ = FTService.compute_scores(
                temp_model_id,
                X_array.tolist(),
                y_array.tolist(),
                use_tanh=True,
            )
            
            # Compute alternative methods
            pearson_scores = []
            spearman_scores = []
            dcor_scores = []
            
            for j in range(n_features):
                X_j = X_array[:, j]
                
                # Pearson
                pearson_corr, _ = pearsonr(X_j, y_array)
                pearson_scores.append(abs(pearson_corr))
                
                # Spearman
                spearman_corr, _ = spearmanr(X_j, y_array)
                spearman_scores.append(abs(spearman_corr))
                
                # Distance Correlation
                dcor_val = distance_correlation(X_j, y_array)
                dcor_scores.append(dcor_val)
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                "feature": feature_names,
                "ft_score": ft_scores,
                "pearson": pearson_scores,
                "spearman": spearman_scores,
                "distance_corr": dcor_scores,
            })
            
            # Add rankings
            comparison_df["ft_rank"] = comparison_df["ft_score"].rank(
                ascending=False,
                method="min",
            ).astype(int)
            comparison_df["pearson_rank"] = comparison_df["pearson"].rank(
                ascending=False,
                method="min",
            ).astype(int)
            comparison_df["spearman_rank"] = comparison_df["spearman"].rank(
                ascending=False,
                method="min",
            ).astype(int)
            comparison_df["dcor_rank"] = comparison_df["distance_corr"].rank(
                ascending=False,
                method="min",
            ).astype(int)
            
            # Clean up temporary model
            model_storage.delete(temp_model_id)
            
            logger.info("Method comparison completed")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error during method comparison: {str(e)}")
            raise
    
    @staticmethod
    def get_model_info(model_id: str) -> Dict:
        """Get metadata about stored model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model information
            
        Raises:
            ValueError: If model not found
        """
        if not model_storage.exists(model_id):
            raise ValueError(f"Model not found: {model_id}")
        
        model_list = model_storage.list_all()
        return model_list.get(model_id, {})


# Global service instance
ft_service = FTService()
