# 🍽️ Food Nutrition Analyzer API with Hugging Face

A powerful FastAPI service that analyzes food images using Hugging Face AI models and provides comprehensive nutritional information with local nutrition database.

## ✨ Features

- **🤖 AI-Powered Food Recognition**: Uses Hugging Face models for food image classification
- **🔍 Ingredient Detection**: Identifies multiple ingredients in food images
- **📊 Comprehensive Nutrition Data**: Detailed macronutrients, vitamins, and minerals
- **🌐 Local Nutrition Database**: Built-in nutrition database for common foods
- **🚀 Fast API**: High-performance async API with automatic documentation
- **📱 Easy Integration**: RESTful API with JSON responses
- **🔧 Fallback Support**: Works even without internet access using fallback classification
- **🆓 No API Keys Required**: Uses open-source Hugging Face models

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Optional: Hugging Face account and token for enhanced model access

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd food_analyze
pip install -r requirements.txt
```

### 2. Environment Setup (Optional)
Create a `.env` file in the root of the project for enhanced functionality:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

**Note:** The application works without a Hugging Face token using fallback classification, but providing a token enables access to more advanced models.

### 3. Run the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Alternative Docs**: http://localhost:8000/redoc

## 📖 API Endpoints

### POST `/analyze-food`
Upload a food image for comprehensive analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze-food" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_food_image.jpg"
```

**Response:**
```json
{
  "food_identification": {
    "primary_dish": "mixed food",
    "confidence": 80.0,
    "alternative_names": ["prepared meal", "vegetable"],
    "food_category": "other",
    "cuisine_type": null
  },
  "detected_ingredients": [
    {
      "name": "mixed food",
      "confidence": 80.0,
      "estimated_weight": 80.0,
      "nutritional_category": "other"
    }
  ],
  "macronutrients": {
    "calories": 150.0,
    "protein": 5.0,
    "carbohydrates": 20.0,
    "fiber": 2.0,
    "sugars": 0.0,
    "fat": 5.0,
    "saturated_fat": 0.0,
    "sodium": 0.0
  },
  "vitamins": [],
  "minerals": [],
  "total_estimated_weight": 100.0,
  "calorie_density": 150.0,
  "analysis_metadata": {
    "model_used": "nateraw/food",
    "confidence_threshold": 0.3,
    "nutrition_source": "local_database"
  }
}
```

### GET `/health`
API health status and system information.

### GET `/`
Basic API information and version.

## 🏗️ Project Structure

```
food_analyze/
├── app.py                    # Main FastAPI application
├── hf_food_analyzer.py       # Hugging Face food analyzer core logic
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── test_app.py              # Automated tests
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_TOKEN` | No | Hugging Face API token for enhanced model access |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### Hugging Face Models Used

- **Food Classification**: `nateraw/food` (primary model)
- **Fallback**: Local heuristic-based classification when models are unavailable

## 🧪 Testing

Run the automated test suite:
```bash
pytest test_app.py -v
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Set up environment variables securely
- Enable HTTPS in production
- Implement rate limiting if needed
- Add authentication for sensitive deployments
- Monitor API usage and performance
- Consider GPU deployment for faster inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Food Classification Model](https://huggingface.co/nateraw/food)

## 🚀 Model Information

This application uses the `nateraw/food` model from Hugging Face for food classification. The model can identify various food items with confidence scores. When the model is not available (due to network issues or missing tokens), the application falls back to a local classification system that still provides useful nutrition analysis.

### Supported Food Categories

The local nutrition database includes common foods across these categories:
- **Proteins**: chicken, beef, salmon, eggs, tofu
- **Carbohydrates**: rice, pasta, bread, potato, quinoa  
- **Vegetables**: broccoli, carrots, spinach, tomato, cucumber
- **Fruits**: apple, banana, orange, strawberry
- **Dairy**: cheese, milk, yogurt
- **Nuts & Seeds**: almonds, walnuts
- **Prepared Foods**: pizza, burger, salad, sandwich, soup

The system provides estimated portion sizes, calorie calculations, and nutritional breakdowns for all supported foods.
