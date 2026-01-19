"""
Core Facteur Tabimetrique implementation.

This module implements the theoretical foundation of Facteur Tabimetrique (FT),
an advanced variable selection method combining multiple correlation measures
and a learned weighting scheme via MLP.

Theory:
    FT_j = tanh(w_j·τ_j + (1-w_j)·ζ_j + γ_j·C_j)
    
    where:
    - ζ (zeta): Pearson correlation
    - τ (tau): Kendall correlation
    - dCor: Distance correlation
    - C: Transitive dependence
    - w, γ: weights learned by MLP

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from scipy.stats import kendalltau, pearsonr


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

logger = logging.getLogger(__name__)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class CorrelationMetrics:
    """Correlation metrics for a single feature.
    
    Attributes:
        tau: Kendall rank correlation coefficient
        zeta: Pearson correlation coefficient
        dcor: Distance correlation
        C: Transitive dependence = |dcor - max(|tau|, |zeta|)|
    """
    tau: float
    zeta: float
    dcor: float
    C: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "tau": self.tau,
            "zeta": self.zeta,
            "dcor": self.dcor,
            "C": self.C,
        }


@dataclass
class MetaFeatures:
    """Meta-characteristics of a feature.
    
    Attributes:
        S_lin: Linearity degree = ζ²
        S_norm: Normality test result (0 or 1)
        S_out: Outlier sensitivity ratio
    """
    S_lin: float
    S_norm: float
    S_out: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array.
        
        Returns:
            1D array [S_lin, S_norm, S_out]
        """
        return np.array([self.S_lin, self.S_norm, self.S_out], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "S_lin": self.S_lin,
            "S_norm": self.S_norm,
            "S_out": self.S_out,
        }


# ============================================================================
# Cache for correlation and meta-feature computation
# ============================================================================

class TabimetricCache:
    """Cache for correlation and meta-feature computations.
    
    Reduces computation time by caching expensive calculations.
    """
    
    def __init__(self) -> None:
        """Initialize empty cache."""
        self._correlations: Dict[int, CorrelationMetrics] = {}
        self._meta_features: Dict[int, MetaFeatures] = {}
    
    def get_correlation(self, feature_idx: int) -> Optional[CorrelationMetrics]:
        """Get cached correlation metrics.
        
        Args:
            feature_idx: Feature index
            
        Returns:
            CorrelationMetrics or None if not cached
        """
        return self._correlations.get(feature_idx)
    
    def set_correlation(
        self, feature_idx: int, metrics: CorrelationMetrics
    ) -> None:
        """Cache correlation metrics.
        
        Args:
            feature_idx: Feature index
            metrics: CorrelationMetrics to cache
        """
        self._correlations[feature_idx] = metrics
    
    def get_meta_features(self, feature_idx: int) -> Optional[MetaFeatures]:
        """Get cached meta-features.
        
        Args:
            feature_idx: Feature index
            
        Returns:
            MetaFeatures or None if not cached
        """
        return self._meta_features.get(feature_idx)
    
    def set_meta_features(
        self, feature_idx: int, features: MetaFeatures
    ) -> None:
        """Cache meta-features.
        
        Args:
            feature_idx: Feature index
            features: MetaFeatures to cache
        """
        self._meta_features[feature_idx] = features
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._correlations.clear()
        self._meta_features.clear()
    
    def size(self) -> Tuple[int, int]:
        """Get cache size.
        
        Returns:
            Tuple of (correlations_count, meta_features_count)
        """
        return len(self._correlations), len(self._meta_features)


# ============================================================================
# Main FacteurTabimetrique Class
# ============================================================================

class FacteurTabimetrique:
    """Advanced variable selection using Facteur Tabimetrique.
    
    Combines multiple correlation measures (Pearson, Kendall, Distance)
    with learned weights via MLP for adaptive feature importance scoring.
    
    Attributes:
        hidden_units: Number of units in hidden layers
        dropout_rate: Dropout rate for regularization
        learning_rate: MLP optimizer learning rate
        verbose: Verbosity level
        weight_model: Trained MLP model
        ft_scores: Computed tabimetric scores
        cache: Cache for expensive computations
    """
    
    def __init__(
        self,
        hidden_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        verbose: int = 0,
    ) -> None:
        """Initialize FacteurTabimetrique.
        
        Args:
            hidden_units: Units in first hidden layer
            dropout_rate: Dropout rate (0-1)
            learning_rate: MLP learning rate
            verbose: Verbosity (0, 1, or 2)
            
        Raises:
            ValueError: If parameters out of valid range
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1")
        
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        self.weight_model: Optional[tf.keras.Model] = None
        self.ft_scores: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.cache = TabimetricCache()
        
        logger.debug(
            f"FacteurTabimetrique initialized: "
            f"hidden_units={hidden_units}, dropout_rate={dropout_rate}"
        )
    
    def _build_weight_model(self, n_features: int) -> tf.keras.Model:
        """Build MLP model for weight learning.
        
        Architecture:
            Input[3] -> Dense[hidden_units] -> Dropout 
                     -> Dense[hidden_units//2] -> Dropout 
                     -> Output[2*n_features] (w, gamma)
        
        Args:
            n_features: Number of features
            
        Returns:
            Compiled TensorFlow Keras model
            
        Raises:
            ValueError: If n_features < 1
        """
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(
                self.hidden_units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(
                self.hidden_units // 2,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(2 * n_features, activation="sigmoid"),
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"],
        )
        
        logger.debug(f"Weight model built for {n_features} features")
        return model
    
    def _compute_correlations(
        self, X_j: np.ndarray, y: np.ndarray, idx: int
    ) -> CorrelationMetrics:
        """Compute correlation metrics for a single feature.
        
        Args:
            X_j: Single feature column (n_samples,)
            y: Target variable (n_samples,)
            idx: Feature index (for caching)
            
        Returns:
            CorrelationMetrics object
            
        Raises:
            ValueError: If shapes incompatible
        """
        if X_j.shape[0] != y.shape[0]:
            raise ValueError("X_j and y must have same length")
        
        # Check cache
        cached = self.cache.get_correlation(idx)
        if cached is not None:
            return cached
        
        try:
            # Pearson (zeta)
            zeta, _ = pearsonr(X_j, y)
            
            # Kendall (tau)
            tau, _ = kendalltau(X_j, y)
            
            # Distance correlation
            dcor = distance_correlation(X_j, y)
            
            # Transitive dependence
            C = np.abs(dcor) - np.maximum(np.abs(tau), np.abs(zeta))
            C = max(C, 0.0)  # No negative values
            
        except Exception as e:
            logger.warning(f"Error computing correlations for feature {idx}: {e}")
            zeta, tau, dcor, C = 0.0, 0.0, 0.0, 0.0
        
        metrics = CorrelationMetrics(tau=tau, zeta=zeta, dcor=dcor, C=C)
        self.cache.set_correlation(idx, metrics)
        
        return metrics
    
    def _compute_meta_features(
        self, X_j: np.ndarray, zeta: float, idx: int
    ) -> MetaFeatures:
        """Compute meta-characteristics for a single feature.
        
        Args:
            X_j: Single feature column (n_samples,)
            zeta: Pearson correlation already computed
            idx: Feature index (for caching)
            
        Returns:
            MetaFeatures object
            
        Raises:
            ValueError: If X_j invalid
        """
        if X_j.ndim != 1:
            raise ValueError("X_j must be 1-dimensional")
        
        # Check cache
        cached = self.cache.get_meta_features(idx)
        if cached is not None:
            return cached
        
        try:
            # S_lin = ζ² (linearity degree)
            S_lin = zeta ** 2
            
            # S_norm = normality test (Shapiro-Wilk)
            if len(X_j) >= 3:
                _, p_value = stats.shapiro(X_j)
                S_norm = 1.0 if p_value > 0.05 else 0.0
            else:
                S_norm = 0.5
            
            # S_out = outlier sensitivity (IQR method)
            Q1 = np.percentile(X_j, 25)
            Q3 = np.percentile(X_j, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_count = np.sum(
                    (X_j < Q1 - 1.5 * IQR) | (X_j > Q3 + 1.5 * IQR)
                )
                S_out = outlier_count / len(X_j)
            else:
                S_out = 0.0
            
        except Exception as e:
            logger.warning(f"Error computing meta-features for feature {idx}: {e}")
            S_lin, S_norm, S_out = 0.5, 0.5, 0.0
        
        features = MetaFeatures(S_lin=S_lin, S_norm=S_norm, S_out=S_out)
        self.cache.set_meta_features(idx, features)
        
        return features
    
    def _prepare_training_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for MLP training.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            
        Returns:
            Tuple of (meta_X, target_w, target_gamma)
            - meta_X: (n_features, 3) meta-features matrix
            - target_w: (n_features,) w weights
            - target_gamma: (n_features,) gamma weights
            
        Raises:
            ValueError: If inputs invalid
        """
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D, y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        n_features = X.shape[1]
        meta_features_list = []
        
        # Compute all correlations and meta-features
        for i in range(n_features):
            corr = self._compute_correlations(X[:, i], y, i)
            meta = self._compute_meta_features(X[:, i], corr.zeta, i)
            meta_features_list.append(meta.to_array())
        
        meta_X = np.array(meta_features_list, dtype=np.float32)
        
        # Initialize target weights (neutral values)
        target_w = np.ones((n_features, 1), dtype=np.float32) * 0.5
        target_gamma = np.ones((n_features, 1), dtype=np.float32) * 0.5
        
        return meta_X, target_w.flatten(), target_gamma.flatten()
    
    def train_weight_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Train the MLP weight model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            epochs: Number of training epochs
            validation_split: Validation split ratio
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Dictionary with training metrics
            
        Raises:
            ValueError: If inputs invalid
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0 < validation_split < 1:
            raise ValueError("validation_split must be in (0, 1)")
        
        logger.info(f"Training weight model on {X.shape}")
        
        n_features = X.shape[1]
        self.weight_model = self._build_weight_model(n_features)
        
        # Prepare data
        meta_X, target_w, target_gamma = self._prepare_training_data(X, y)
        
        # Combine targets: [w_0, ..., w_n, gamma_0, ..., gamma_n]
        targets = np.concatenate([target_w, target_gamma], axis=0).reshape(1, -1)
        targets = np.repeat(targets, n_features, axis=0)
        
        # Build callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            )
        
        try:
            history = self.weight_model.fit(
                meta_X,
                targets,
                epochs=epochs,
                batch_size=max(1, n_features // 2),
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=self.verbose,
            )
            
            logger.info("Weight model training completed")
            
            return {
                "final_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
                "epochs_trained": len(history.history["loss"]),
                "early_stopped": len(callbacks) > 0,
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def compute_tabimetric_scores(
        self, X: np.ndarray, y: np.ndarray, use_tanh: bool = True
    ) -> np.ndarray:
        """Compute Facteur Tabimetrique scores.
        
        Formula: FT_j = tanh(w_j·τ_j + (1-w_j)·ζ_j + γ_j·C_j)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            use_tanh: Whether to apply tanh activation
            
        Returns:
            FT scores (n_features,)
            
        Raises:
            RuntimeError: If weight model not trained
            ValueError: If inputs invalid
        """
        if self.weight_model is None:
            raise RuntimeError("Weight model not trained. Call train_weight_model first.")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D, y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        logger.info(f"Computing FT scores for {X.shape[1]} features")
        
        n_features = X.shape[1]
        correlations = []
        meta_features_list = []
        
        # Compute correlations and meta-features for all features
        for i in range(n_features):
            corr = self._compute_correlations(X[:, i], y, i)
            meta = self._compute_meta_features(X[:, i], corr.zeta, i)
            correlations.append(corr)
            meta_features_list.append(meta.to_array())
        
        meta_X = np.array(meta_features_list, dtype=np.float32)
        
        # Get weights from MLP
        weights_output = self.weight_model.predict(meta_X, verbose=0)
        w_values = weights_output[:, :n_features]
        gamma_values = weights_output[:, n_features:]
        
        # Compute FT scores
        tau = np.array([c.tau for c in correlations])
        zeta = np.array([c.zeta for c in correlations])
        C = np.array([c.C for c in correlations])
        
        ft_raw = (w_values[0] * tau + 
                  (1 - w_values[0]) * zeta + 
                  gamma_values[0] * C)
        
        if use_tanh:
            self.ft_scores = np.tanh(ft_raw)
        else:
            self.ft_scores = ft_raw
        
        logger.info("FT scores computed")
        return self.ft_scores
    
    def select_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.5,
        return_indices: bool = False,
    ) -> pd.DataFrame:
        """Select features based on FT score threshold.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Feature names
            threshold: Selection threshold (0-1)
            return_indices: If True, return indices instead of names
            
        Returns:
            DataFrame with selected features and their scores
            
        Raises:
            RuntimeError: If FT scores not computed
            ValueError: If threshold invalid
        """
        if self.ft_scores is None:
            raise RuntimeError("FT scores not computed. Call compute_tabimetric_scores first.")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if len(feature_names) != len(self.ft_scores):
            raise ValueError("feature_names length must match X columns")
        
        mask = np.abs(self.ft_scores) >= threshold
        selected_idx = np.where(mask)[0]
        
        if return_indices:
            selected_names = selected_idx.tolist()
        else:
            selected_names = [feature_names[i] for i in selected_idx]
        
        selected_scores = self.ft_scores[mask]
        
        df = pd.DataFrame({
            "feature": selected_names,
            "ft_score": selected_scores,
            "abs_score": np.abs(selected_scores),
        })
        
        df = df.sort_values("abs_score", ascending=False).reset_index(drop=True)
        
        logger.info(f"Selected {len(df)} features with threshold {threshold}")
        return df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get detailed feature importance report.
        
        Returns:
            DataFrame with all metrics for each feature
            
        Raises:
            RuntimeError: If FT scores not computed or feature names not set
        """
        if self.ft_scores is None:
            raise RuntimeError("FT scores not computed yet")
        if not self.feature_names:
            raise RuntimeError("Feature names not set")
        
        data = []
        for i, name in enumerate(self.feature_names):
            cached_corr = self.cache.get_correlation(i)
            cached_meta = self.cache.get_meta_features(i)
            
            if cached_corr and cached_meta:
                data.append({
                    "feature": name,
                    "ft_score": self.ft_scores[i],
                    "abs_score": np.abs(self.ft_scores[i]),
                    **cached_corr.to_dict(),
                    **cached_meta.to_dict(),
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values("abs_score", ascending=False).reset_index(drop=True)
        
        return df
    
    def get_mlp_weights(self) -> Dict[str, Any]:
        """Get MLP model weights and architecture.
        
        Returns:
            Dictionary with model information
            
        Raises:
            RuntimeError: If model not trained
        """
        if self.weight_model is None:
            raise RuntimeError("Weight model not trained")
        
        weights = []
        for layer in self.weight_model.layers:
            if hasattr(layer, "kernel"):
                weights.append({
                    "layer": layer.name,
                    "shape": layer.kernel.shape,
                    "dtype": str(layer.kernel.dtype),
                })
        
        return {
            "model_name": self.weight_model.name,
            "total_params": int(self.weight_model.count_params()),
            "layers": len(self.weight_model.layers),
            "weights_info": weights,
        }
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> pd.DataFrame:
        """Complete pipeline: train + compute scores + select features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: Optional feature names
            threshold: Selection threshold
            **kwargs: Additional arguments for train_weight_model
            
        Returns:
            DataFrame with selected features
            
        Raises:
            ValueError: If inputs invalid
        """
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D, y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same samples")
        
        # Set feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names
        
        # Train
        self.train_weight_model(X, y, **kwargs)
        
        # Compute scores
        self.compute_tabimetric_scores(X, y)
        
        # Select
        selected_df = self.select_features(X, feature_names, threshold)
        
        logger.info(f"Pipeline completed: {len(selected_df)} features selected")
        return selected_df
