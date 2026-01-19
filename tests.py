"""
Comprehensive test suite for Facteur Tabimetrique API.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.core.facteur_tabimetrique import FacteurTabimetrique
from app.services.storage import storage


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    return {
        "X": X.tolist(),
        "y": y.tolist(),
        "feature_names": [f"feature_{i}" for i in range(n_features)],
    }


class TestFacteurTabimetrique:
    """Test FacteurTabimetrique core class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = FacteurTabimetrique(model_id="test_model")
        assert model.model_id == "test_model"
        assert not model.is_trained
        assert model.ft_scores is None
    
    def test_compute_correlations(self, sample_data):
        """Test correlation computation."""
        model = FacteurTabimetrique(model_id="test_corr")
        X = np.array(sample_data["X"])
        y = np.array(sample_data["y"])
        
        correlations = model.compute_correlations(X, y)
        
        assert "zeta" in correlations
        assert "tau" in correlations
        assert "dcor" in correlations
        assert "transitive" in correlations
        
        assert len(correlations["zeta"]) == X.shape[1]
        assert all(-1 <= c <= 1 for c in correlations["zeta"])
    
    def test_compute_meta_features(self, sample_data):
        """Test meta-features computation."""
        model = FacteurTabimetrique(model_id="test_meta")
        X = np.array(sample_data["X"])
        y = np.array(sample_data["y"])
        
        model.compute_correlations(X, y)
        meta_features = model.compute_meta_features(X)
        
        assert "linearity" in meta_features
        assert "normality" in meta_features
        assert "outlier_sensitivity" in meta_features
        
        assert len(meta_features["linearity"]) == X.shape[1]
        assert all(0 <= v <= 1 for v in meta_features["linearity"])
    
    def test_training(self, sample_data):
        """Test model training."""
        model = FacteurTabimetrique(
            model_id="test_train",
            epochs=10,
            batch_size=32,
        )
        X = np.array(sample_data["X"])
        y = np.array(sample_data["y"])
        
        metrics = model.train(X, y, sample_data["feature_names"])
        
        assert model.is_trained
        assert "final_loss" in metrics
        assert "final_val_loss" in metrics
        assert "epochs_trained" in metrics
    
    def test_scoring(self, sample_data):
        """Test model scoring."""
        model = FacteurTabimetrique(
            model_id="test_score",
            epochs=10,
        )
        X = np.array(sample_data["X"])
        y = np.array(sample_data["y"])
        
        model.train(X, y)
        ft_scores = model.score(X)
        
        assert len(ft_scores) == X.shape[1]
        assert all(-1 <= s <= 1 for s in ft_scores)
    
    def test_feature_selection(self, sample_data):
        """Test feature selection."""
        model = FacteurTabimetrique(
            model_id="test_select",
            epochs=10,
        )
        X = np.array(sample_data["X"])
        y = np.array(sample_data["y"])
        
        model.train(X, y, sample_data["feature_names"])
        model.score(X)
        
        selected_names, selected_scores = model.select_features(threshold=0.3)
        
        assert isinstance(selected_names, list)
        assert len(selected_names) <= len(model.feature_names)
    
    def test_importance_report(self, sample_data):
        """Test importance report generation."""
        model = FacteurTabimetrique(
            model_id="test_importance",
            epochs=10,
        )
        X = np.array(sample_data["X"])
        y = np.array(sample_data["y"])
        
        model.train(X, y, sample_data["feature_names"])
        model.score(X)
        
        report = model.get_importance_report()
        
        assert isinstance(report, pd.DataFrame)
        assert "feature" in report.columns
        assert "ft_score" in report.columns
        assert len(report) == X.shape[1]


class TestModelStorage:
    """Test model storage service."""
    
    def test_create_model(self):
        """Test model creation."""
        storage.clear_all()
        
        model = storage.create_model(
            model_id="storage_test_1",
            epochs=10,
        )
        
        assert model.model_id == "storage_test_1"
        assert "storage_test_1" in storage.list_models()
    
    def test_get_model(self):
        """Test model retrieval."""
        storage.clear_all()
        
        created_model = storage.create_model(model_id="test_retrieve")
        retrieved_model = storage.get_model("test_retrieve")
        
        assert retrieved_model is not None
        assert retrieved_model.model_id == "test_retrieve"
    
    def test_delete_model(self):
        """Test model deletion."""
        storage.clear_all()
        
        storage.create_model(model_id="test_delete")
        assert "test_delete" in storage.list_models()
        
        storage.delete_model("test_delete")
        assert "test_delete" not in storage.list_models()
    
    def test_duplicate_model_error(self):
        """Test duplicate model error."""
        storage.clear_all()
        
        storage.create_model(model_id="duplicate_test")
        
        with pytest.raises(ValueError):
            storage.create_model(model_id="duplicate_test")
    
    def test_get_model_info(self):
        """Test getting model info."""
        storage.clear_all()
        
        model = storage.create_model(model_id="info_test")
        info = storage.get_model_info("info_test")
        
        assert info["model_id"] == "info_test"
        assert not info["is_trained"]
        assert info["n_features"] == 0


class TestTrainEndpoint:
    """Test training endpoint."""
    
    def test_train_success(self, client, sample_data):
        """Test successful training."""
        storage.clear_all()
        
        response = client.post(
            "/api/v1/train",
            json={
                "model_id": "endpoint_test_1",
                "X": sample_data["X"],
                "y": sample_data["y"],
                "feature_names": sample_data["feature_names"],
                "epochs": 10,
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["model_id"] == "endpoint_test_1"
        assert data["status"] == "trained"
    
    def test_train_invalid_data(self, client):
        """Test training with invalid data."""
        storage.clear_all()
        
        response = client.post(
            "/api/v1/train",
            json={
                "model_id": "invalid_test",
                "X": [[1.0]],
                "y": [1.0],
            },
        )
        
        assert response.status_code in [400, 500]


class TestScoreEndpoint:
    """Test scoring endpoint."""
    
    def test_score_success(self, client, sample_data):
        """Test successful scoring."""
        storage.clear_all()
        
        # Train first
        client.post(
            "/api/v1/train",
            json={
                "model_id": "score_test",
                "X": sample_data["X"],
                "y": sample_data["y"],
                "epochs": 10,
            },
        )
        
        # Score
        response = client.post(
            "/api/v1/score",
            json={
                "model_id": "score_test",
                "X": sample_data["X"],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "score_test"
        assert len(data["ft_scores"]) == len(sample_data["X"][0])
    
    def test_score_untrained_model(self, client, sample_data):
        """Test scoring untrained model."""
        storage.clear_all()
        
        response = client.post(
            "/api/v1/score",
            json={
                "model_id": "nonexistent",
                "X": sample_data["X"],
            },
        )
        
        assert response.status_code == 400


class TestSelectEndpoint:
    """Test feature selection endpoint."""
    
    def test_select_success(self, client, sample_data):
        """Test successful feature selection."""
        storage.clear_all()
        
        # Train
        client.post(
            "/api/v1/train",
            json={
                "model_id": "select_test",
                "X": sample_data["X"],
                "y": sample_data["y"],
                "feature_names": sample_data["feature_names"],
                "epochs": 10,
            },
        )
        
        # Score
        client.post(
            "/api/v1/score",
            json={
                "model_id": "select_test",
                "X": sample_data["X"],
            },
        )
        
        # Select
        response = client.post(
            "/api/v1/select",
            json={
                "model_id": "select_test",
                "threshold": 0.3,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "selected_features" in data
        assert data["count_total"] == len(sample_data["feature_names"])


class TestPipelineEndpoint:
    """Test pipeline endpoint."""
    
    def test_pipeline_success(self, client, sample_data):
        """Test successful pipeline."""
        storage.clear_all()
        
        response = client.post(
            "/api/v1/pipeline",
            json={
                "model_id": "pipeline_test",
                "X": sample_data["X"],
                "y": sample_data["y"],
                "feature_names": sample_data["feature_names"],
                "threshold": 0.3,
                "epochs": 10,
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "completed"
        assert "selected_features" in data


class TestImportanceEndpoint:
    """Test importance endpoint."""
    
    def test_importance_success(self, client, sample_data):
        """Test successful importance report."""
        storage.clear_all()
        
        # Train
        client.post(
            "/api/v1/train",
            json={
                "model_id": "importance_test",
                "X": sample_data["X"],
                "y": sample_data["y"],
                "feature_names": sample_data["feature_names"],
                "epochs": 10,
            },
        )
        
        response = client.get("/api/v1/importance/importance_test")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["importance_data"]) == len(sample_data["feature_names"])


class TestModelsEndpoint:
    """Test models listing endpoint."""
    
    def test_list_models_empty(self, client):
        """Test listing empty models."""
        storage.clear_all()
        
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
    
    def test_list_models_with_data(self, client, sample_data):
        """Test listing with models."""
        storage.clear_all()
        
        # Create models
        for i in range(3):
            client.post(
                "/api/v1/train",
                json={
                    "model_id": f"list_test_{i}",
                    "X": sample_data["X"],
                    "y": sample_data["y"],
                    "epochs": 10,
                },
            )
        
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3


class TestDeleteEndpoint:
    """Test model deletion endpoint."""
    
    def test_delete_success(self, client, sample_data):
        """Test successful deletion."""
        storage.clear_all()
        
        # Create model
        client.post(
            "/api/v1/train",
            json={
                "model_id": "delete_test",
                "X": sample_data["X"],
                "y": sample_data["y"],
                "epochs": 10,
            },
        )
        
        response = client.delete("/api/v1/models/delete_test")
        
        assert response.status_code == 204
        assert "delete_test" not in storage.list_models()


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
