"""
Usage examples for Facteur Tabimetrique API.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import requests
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000/api/v1"


class FTAPIClient:
    """Simple client for Facteur Tabimetrique API."""
    
    def __init__(self, base_url: str = BASE_URL):
        """Initialize client.
        
        Args:
            base_url: API base URL
        """
        self.base_url = base_url
    
    def train(
        self,
        model_id: str,
        X: list,
        y: list,
        feature_names: list | None = None,
        epochs: int = 100,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Train a model.
        
        Args:
            model_id: Model identifier
            X: Feature matrix
            y: Target variable
            feature_names: Feature names
            epochs: Training epochs
            threshold: Selection threshold
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/train",
            json={
                "model_id": model_id,
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "epochs": epochs,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def score(self, model_id: str, X: list) -> Dict[str, Any]:
        """Score with a model.
        
        Args:
            model_id: Model identifier
            X: Feature matrix
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/score",
            json={"model_id": model_id, "X": X},
        )
        response.raise_for_status()
        return response.json()
    
    def select_features(
        self, model_id: str, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Select features.
        
        Args:
            model_id: Model identifier
            threshold: Selection threshold
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/select",
            json={"model_id": model_id, "threshold": threshold},
        )
        response.raise_for_status()
        return response.json()
    
    def pipeline(
        self,
        model_id: str,
        X: list,
        y: list,
        feature_names: list | None = None,
        threshold: float = 0.5,
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """Run complete pipeline.
        
        Args:
            model_id: Model identifier
            X: Feature matrix
            y: Target variable
            feature_names: Feature names
            threshold: Selection threshold
            epochs: Training epochs
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/pipeline",
            json={
                "model_id": model_id,
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "threshold": threshold,
                "epochs": epochs,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def get_importance(self, model_id: str) -> Dict[str, Any]:
        """Get importance report.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Response dictionary
        """
        response = requests.get(f"{self.base_url}/importance/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def compare_methods(
        self, model_id: str, X: list, y: list
    ) -> Dict[str, Any]:
        """Compare with other methods.
        
        Args:
            model_id: Model identifier
            X: Feature matrix
            y: Target variable
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/compare",
            json={"model_id": model_id, "X": X, "y": y},
        )
        response.raise_for_status()
        return response.json()
    
    def upload_csv(
        self,
        model_id: str,
        file_path: str,
        target_column: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Upload CSV and run pipeline.
        
        Args:
            model_id: Model identifier
            file_path: Path to CSV file
            target_column: Target column name
            threshold: Selection threshold
            
        Returns:
            Response dictionary
        """
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "model_id": model_id,
                "target_column": target_column,
                "threshold": threshold,
            }
            response = requests.post(
                f"{self.base_url}/upload-csv",
                files=files,
                data=data,
            )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List all models.
        
        Returns:
            Response dictionary
        """
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model.
        
        Args:
            model_id: Model identifier
        """
        response = requests.delete(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
    
    def health_check(self) -> Dict[str, Any]:
        """Health check.
        
        Returns:
            Response dictionary
        """
        response = requests.get("http://localhost:8000/health")
        response.raise_for_status()
        return response.json()


def example_1_basic_usage():
    """Example 1: Basic training and scoring."""
    print("=" * 60)
    print("Example 1: Basic Training and Scoring")
    print("=" * 60)
    
    client = FTAPIClient()
    
    # Check API health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features).tolist()
    y = (X[i][0] + 0.5 * X[i][1] for i in range(n_samples))
    y = [sum([X[i][0], 0.5 * X[i][1], np.random.randn() * 0.1]) for i in range(n_samples)]
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Train model
    print("\n1. Training model...")
    train_response = client.train(
        model_id="example_model_1",
        X=X,
        y=y,
        feature_names=feature_names,
        epochs=50,
    )
    print(f"   Status: {train_response['status']}")
    print(f"   Features: {train_response['features']}")
    print(f"   Samples: {train_response['samples']}")
    
    # Score model
    print("\n2. Scoring model...")
    score_response = client.score(
        model_id="example_model_1",
        X=X,
    )
    print(f"   FT Scores: {[f'{s:.4f}' for s in score_response['ft_scores']]}")
    
    # List models
    print("\n3. Listing models...")
    models = client.list_models()
    print(f"   Total models: {models['count']}")
    for model_info in models['models']:
        print(f"   - {model_info['model_id']}: {model_info['n_features']} features")


def example_2_feature_selection():
    """Example 2: Feature selection with threshold."""
    print("\n" + "=" * 60)
    print("Example 2: Feature Selection")
    print("=" * 60)
    
    client = FTAPIClient()
    
    # Generate data
    np.random.seed(42)
    n_samples = 150
    n_features = 8
    
    X = np.random.randn(n_samples, n_features).tolist()
    y = [X[i][0] + X[i][1] + np.random.randn() * 0.1 for i in range(n_samples)]
    
    feature_names = [f"var_{i}" for i in range(n_features)]
    
    # Train
    print("\n1. Training with feature selection...")
    response = client.pipeline(
        model_id="example_model_2",
        X=X,
        y=y,
        feature_names=feature_names,
        threshold=0.4,
        epochs=50,
    )
    
    print(f"   Status: {response['status']}")
    print(f"   Total features: {response['count_total']}")
    print(f"   Selected features: {response['count_selected']}")
    print(f"   Selected: {response['selected_features']}")
    print(f"   Scores: {[f'{s:.4f}' for s in response['selected_scores']]}")


def example_3_importance_report():
    """Example 3: Detailed importance report."""
    print("\n" + "=" * 60)
    print("Example 3: Importance Report")
    print("=" * 60)
    
    client = FTAPIClient()
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features).tolist()
    y = [X[i][0] + 0.5 * X[i][1] + np.random.randn() * 0.1 for i in range(n_samples)]
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Train
    print("\n1. Training model...")
    client.train(
        model_id="example_model_3",
        X=X,
        y=y,
        feature_names=feature_names,
        epochs=50,
    )
    
    # Get importance
    print("\n2. Getting importance report...")
    importance = client.get_importance("example_model_3")
    
    print(f"\n   Importance Report ({importance['timestamp']}):")
    print(f"   {'Feature':<15} {'FT_Score':<12} {'Zeta':<12} {'Tau':<12}")
    print(f"   {'-'*51}")
    
    for item in importance['importance_data']:
        print(
            f"   {item['feature']:<15} "
            f"{item['ft_score']:<12.4f} "
            f"{item['zeta']:<12.4f} "
            f"{item['tau']:<12.4f}"
        )


def example_4_method_comparison():
    """Example 4: Compare with other correlation methods."""
    print("\n" + "=" * 60)
    print("Example 4: Method Comparison")
    print("=" * 60)
    
    client = FTAPIClient()
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    X = np.random.randn(n_samples, n_features).tolist()
    y = [X[i][0] + np.random.randn() * 0.1 for i in range(n_samples)]
    
    feature_names = [f"X{i}" for i in range(n_features)]
    
    # Train
    print("\n1. Training model...")
    client.train(
        model_id="example_model_4",
        X=X,
        y=y,
        feature_names=feature_names,
        epochs=50,
    )
    
    # Compare
    print("\n2. Comparing methods...")
    comparison = client.compare_methods(
        model_id="example_model_4",
        X=X,
        y=y,
    )
    
    print(f"\n   Method Comparison:")
    print(f"   {'Method':<20} {'Rank Correlation':<20}")
    print(f"   {'-'*40}")
    
    for method in comparison['comparison_methods']:
        print(
            f"   {method['method']:<20} "
            f"{method['rank_correlation']:<20.4f}"
        )
    
    print(f"\n   Summary: {comparison['summary']['note']}")


def example_5_csv_upload():
    """Example 5: CSV file upload."""
    print("\n" + "=" * 60)
    print("Example 5: CSV Upload and Analysis")
    print("=" * 60)
    
    client = FTAPIClient()
    
    # Create sample CSV
    print("\n1. Creating sample CSV file...")
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'target': np.random.randn(100),
    })
    
    csv_path = "sample_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"   Created {csv_path}")
    
    # Upload and run pipeline
    print("\n2. Uploading and analyzing...")
    try:
        response = client.upload_csv(
            model_id="example_csv_model",
            file_path=csv_path,
            target_column="target",
            threshold=0.3,
        )
        
        print(f"   Status: {response['status']}")
        print(f"   Selected: {response['selected_features']}")
    except requests.exceptions.ConnectionError:
        print("   Note: API not running. Run 'python run.py' to start the server.")


def example_6_cleanup():
    """Example 6: Model cleanup."""
    print("\n" + "=" * 60)
    print("Example 6: Model Management")
    print("=" * 60)
    
    client = FTAPIClient()
    
    # List models
    print("\n1. Current models:")
    models = client.list_models()
    print(f"   Total: {models['count']}")
    for model_info in models['models']:
        print(f"   - {model_info['model_id']}")
    
    # Delete specific model
    if models['count'] > 0:
        model_to_delete = models['models'][0]['model_id']
        print(f"\n2. Deleting {model_to_delete}...")
        client.delete_model(model_to_delete)
        print("   Deleted successfully")
    
    # List again
    print("\n3. Models after cleanup:")
    models = client.list_models()
    print(f"   Total: {models['count']}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Facteur Tabimetrique API - Usage Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        example_1_basic_usage()
        example_2_feature_selection()
        example_3_importance_report()
        example_4_method_comparison()
        example_5_csv_upload()
        example_6_cleanup()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n" + "!" * 60)
        print("ERROR: Cannot connect to API")
        print("!" * 60)
        print("\nMake sure the API is running:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Start the server: python run.py")
        print("  3. Then run this script again in another terminal\n")


if __name__ == "__main__":
    main()
