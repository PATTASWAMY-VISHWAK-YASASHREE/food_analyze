# Food Analysis AI System - Copilot Instructions

## Architecture Overview

This is an **enhanced food nutrition analyzer** using state-of-the-art Vision Transformer models. The system has undergone significant upgrades from basic CNN to advanced models with 89.6% accuracy.

### Core Components
- **`app.py`**: FastAPI server with structured Pydantic models for comprehensive nutrition analysis
- **`hf_food_analyzer_enhanced.py`**: Enhanced analyzer using `ashaduzzaman/vit-finetuned-food101` (89.6% accuracy)
- **`hf_food_analyzer.py`**: Legacy analyzer using `nateraw/food` (kept for fallback)
- **Web UI**: `test_ui.html` provides complete testing interface with drag-drop functionality

### Model Architecture
```
Enhanced System: Vision Transformer (ViT) → 89.6% accuracy
Legacy System:   Basic CNN → Lower accuracy
Nutrition NER:   sgarbi/bert-fda-nutrition-ner (FDA data)
```

## Critical Development Patterns

### 1. Model Initialization & Fallback Strategy
```python
# Always implement graceful degradation
try:
    self.hf_analyzer = EnhancedHuggingFaceFoodAnalyzer(hf_token=self.hf_token)
    logger.info("Enhanced analyzer initialized successfully")
except Exception as e:
    logger.error(f"Could not initialize: {e}")
    self.hf_analyzer = None  # Fallback to basic classification
```

### 2. API Response Structure (Critical for Frontend)
The API returns structured data with **specific key names** that frontend depends on:
```python
# Frontend expects these exact keys:
{
    "food_identification": {"primary_dish": ..., "confidence": ...},
    "macronutrients": {"calories": ..., "protein": ...},
    "health_assessment": {"status": ..., "score": ...}
}
```

### 3. Environment Configuration Pattern
```python
# Support both development and production environments
port = int(os.getenv("PORT", 8000))  # Cloud Run uses PORT env var
hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Optional but enhances accuracy
```

## Essential Development Commands

### Local Development
```bash
# Start development server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest test_app.py -v

# Test with web UI (requires separate server)
python -m http.server 3000  # Serve test_ui.html
```

### Testing & Validation
```bash
# Complete system demo
python demo_enhanced_system.py

# API validation
python test_api.py

# Direct model testing
python -c "from hf_food_analyzer_enhanced import EnhancedHuggingFaceFoodAnalyzer; analyzer = EnhancedHuggingFaceFoodAnalyzer()"
```

### Deployment Commands
```bash
# Google Cloud deployment (automated)
chmod +x deploy.sh && ./deploy.sh

# Manual deployment
gcloud builds submit --config cloudbuild.yaml .

# Local Docker testing
docker build -t food-analyzer . && docker run -p 8080:8080 food-analyzer
```

## Project-Specific Conventions

### Response Format Consistency
- **Always map API responses to frontend expectations** in `test_ui.html` JavaScript
- Use `result.food_identification` not `result.classification`
- Use `result.macronutrients` not `result.nutrition`

### Model Version Management
- **Enhanced models** are in `hf_food_analyzer_enhanced.py` with **dynamic nutrition calculation**
- **Legacy models** are in `hf_food_analyzer.py` (fallback only)
- Always prefer enhanced models unless debugging legacy issues
- **Critical**: Enhanced analyzer uses `_calculate_dynamic_nutrition()` for ANY food the model detects

### Nutrition Calculation Pattern
```python
# WRONG: Static lookup that ignores model predictions
if food_label in static_database: return static_database[food_label]

# CORRECT: Dynamic calculation based on actual model output
def _calculate_dynamic_nutrition(self, food_label: str) -> Dict[str, float]:
    food_lower = food_label.lower()
    if any(meat in food_lower for meat in ["chicken", "beef", "pork"]):
        return {"calories": 250, "protein": 25, ...}  # Calculated values
```

### Error Handling Pattern
```python
# Use structured error responses
raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Log errors with context
logger.error(f"Error in food analysis: {e}", exc_info=True)
```

### File Organization
- **Analysis results**: Auto-saved to `analysis_results/` with timestamps
- **Test files**: Prefixed with `test_` and include both API and UI testing
- **Demo files**: Prefixed with `demo_` for system demonstrations

## Integration Points

### Hugging Face Models
- **Primary**: `ashaduzzaman/vit-finetuned-food101` (Vision Transformer, 89.6% accuracy)
- **Nutrition**: `sgarbi/bert-fda-nutrition-ner` (FDA nutrition extraction)
- **Token**: Optional but significantly improves model access and accuracy

### Cloud Services
- **Google Cloud Run**: Production deployment with 2Gi memory requirement for ML models
- **Cloud Build**: Automated CI/CD via `cloudbuild.yaml`
- **Container Registry**: Docker image storage

### Frontend-Backend Communication
- **CORS**: Configured for `*` origins (adjust for production)
- **File Upload**: Supports JPEG, PNG, GIF, WebP (NOT AVIF due to PIL limitations)
- **Response Structure**: Deeply nested JSON requiring careful frontend mapping

## Debugging & Troubleshooting

### Common Issues
1. **Model loading failures**: Check `HUGGINGFACE_TOKEN` and network connectivity
2. **Frontend response mapping**: Verify API response structure matches JavaScript expectations
3. **Memory issues**: ML models require 2Gi+ memory in production
4. **Image format errors**: Ensure PIL supports the uploaded image format
5. **Static nutrition bug**: Ensure `_calculate_dynamic_nutrition()` is used, not static lookups

### Health Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# Verify model status
curl http://localhost:8000/health | jq '.services'
```

### Performance Optimization
- **Model caching**: Models are loaded once at startup
- **Async processing**: All endpoints use FastAPI async patterns  
- **Response compression**: Large JSON responses from detailed nutrition analysis
- **Analysis speed**: Enhanced models process images in ~0.3s

## Production Considerations

- **Security**: Uses non-root Docker user, validates file types
- **Monitoring**: Health checks at `/health`, detailed logs to Cloud Logging
- **Scaling**: Configured for 80 concurrent requests per Cloud Run instance
- **Environment**: Supports both development (port 8000) and production (PORT env var)