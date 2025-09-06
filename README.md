# 🍽️ Food Nutrition Analyzer API with USDA

A powerful FastAPI service that analyzes food images and provides comprehensive nutritional information using the USDA FoodData Central API for accurate, official nutrition data.

## ✨ Features

- **🔍 Food Image Analysis**: Processes food images for identification and analysis
- **🏛️ USDA Official Data**: Uses USDA FoodData Central for accurate nutrition information
- **📊 Comprehensive Nutrition**: Detailed macronutrients, vitamins, and minerals from official sources
- **🔎 Advanced Food Matching**: Smart food matching with USDA database
- **🚀 Fast API**: High-performance async API with automatic documentation
- **📱 Easy Integration**: RESTful API with JSON responses
- **🔧 Fallback Support**: Works even without USDA API key using fallback nutrition data
- **🆓 Free USDA API**: Uses official USDA API (free registration required)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- USDA FoodData Central API key (free registration at https://fdc.nal.usda.gov/api-key-signup.html)

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd food_analyze
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root of the project:
```
USDA_API_KEY=your_usda_api_key_here
```

**Note:** The application works without a USDA API key using fallback nutrition data, but providing a key enables access to official USDA nutritional information.

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
    "primary_dish": "mixed vegetables",
    "confidence": 75.0,
    "alternative_names": ["chicken breast", "rice"],
    "food_category": "vegetable",
    "cuisine_type": null
  },
  "detected_ingredients": [
    {
      "name": "mixed vegetables",
      "confidence": 75.0,
      "estimated_weight": 75.0,
      "nutritional_category": "vegetable",
      "usda_fdc_id": 123456,
      "usda_description": "Vegetables, mixed, frozen"
    }
  ],
  "macronutrients": {
    "calories": 112.5,
    "protein": 6.0,
    "carbohydrates": 15.0,
    "fiber": 2.25,
    "sugars": 3.75,
    "fat": 3.75,
    "saturated_fat": 1.125,
    "sodium": 150.0
  },
  "vitamins": [
    {
      "name": "Vitamin A",
      "amount": 835,
      "unit": "mcg",
      "daily_value_percentage": 93
    }
  ],
  "minerals": [
    {
      "name": "Iron",
      "amount": 2.7,
      "unit": "mg",
      "daily_value_percentage": 15
    }
  ],
  "usda_matches": [
    {
      "fdc_id": 123456,
      "description": "Vegetables, mixed, frozen",
      "match_confidence": 85.0,
      "food_category": "Vegetables and Vegetable Products",
      "ingredients": "Carrots, Green Beans, Corn, Peas"
    }
  ],
  "total_estimated_weight": 100.0,
  "calorie_density": 150.0,
  "analysis_metadata": {
    "usda_api_used": true,
    "nutrition_source": "usda_fooddata_central",
    "api_version": "v1"
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
├── usda_food_analyzer.py     # USDA food analyzer core logic
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
| `USDA_API_KEY` | No | USDA FoodData Central API key for official nutrition data |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### USDA FoodData Central

This application integrates with the USDA FoodData Central API to provide official nutrition data:

- **API Documentation**: https://fdc.nal.usda.gov/api-guide.html
- **API Key Registration**: https://fdc.nal.usda.gov/api-key-signup.html (free)
- **Food Search**: Advanced food matching with USDA database
- **Fallback**: Local nutrition data when API is unavailable

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
- Set up environment variables securely (especially USDA API key)
- Enable HTTPS in production
- Implement rate limiting if needed
- Add authentication for sensitive deployments
- Monitor API usage and performance
- Consider caching USDA API responses to reduce API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [USDA FoodData Central](https://fdc.nal.usda.gov/)
- [USDA API Documentation](https://fdc.nal.usda.gov/api-guide.html)
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
