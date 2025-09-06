import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

from app import EnhancedFoodAnalyzer

def test_analyzer_initialization():
    """Tests the initialization of the EnhancedFoodAnalyzer with Hugging Face models."""
    try:
        analyzer = EnhancedFoodAnalyzer()
        # The analyzer should initialize even without HF token for testing
        assert analyzer is not None
    except Exception as e:
        pytest.fail(f"EnhancedFoodAnalyzer initialization failed: {e}")

def test_health_check():
    """Tests the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "services" in data
    # Note: These may be False if no HF token is provided, which is okay for testing
    assert "huggingface_models" in data["services"]
    assert "huggingface_token" in data["services"]

def test_root_endpoint():
    """Tests the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Food Nutrition Analyzer API with Hugging Face"
    assert data["version"] == "3.0.0"

# Note: The test for the /analyze-food endpoint requires a valid image and HF token
# It can be manually tested using the API documentation at /docs
