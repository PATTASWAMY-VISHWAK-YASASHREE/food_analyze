# 🍽️ Enhanced Food Nutrition Analyzer API

A powerful FastAPI service that analyzes food images using AI and provides comprehensive nutritional information integrated with the USDA FoodData Central database.

## ✨ Features

- **� Multi-Food Detection**: Identifies multiple ingredients in a single image (salmon, vegetables, etc.)
- **📊 Comprehensive Nutrition Data**: Detailed macronutrients, vitamins, and minerals
- **🏛️ USDA Integration**: Official government nutrition database
- **🚀 Fast API**: High-performance async API with automatic documentation
- **📱 Easy Integration**: RESTful API with JSON responses
- **🔒 Secure**: Environment-based configuration for API keys

## �🚀 Quick Start

### Prerequisites
- Python 3.8+
- USDA API Key ([Get one here](https://fdc.nal.usda.gov/api-guide.html))

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd food-nutrition-analyzer
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your USDA API key
# USDA_API_KEY=your_actual_api_key_here
```

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
  "analysis_id": "uuid-string",
  "timestamp": "2025-09-06T10:30:00Z",
  "detected_foods": [
    {
      "name": "salmon",
      "confidence": 0.95,
      "weight_grams": 150
    }
  ],
  "total_nutrition": {
    "calories": 231,
    "protein": 31.62,
    "fat": 10.43,
    "carbohydrates": 0
  },
  "usda_matches": [...],
  "health_insights": {...}
}
```

### GET `/health`
API health status and system information.

## 🏗️ Project Structure

```
food-nutrition-analyzer/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── test_client.py       # Test client for manual testing
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `USDA_API_KEY` | Yes | USDA FoodData Central API key |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### Mock Vision API

Currently uses a mock vision API for demonstration. To integrate with Google Cloud Vision:

1. Set up Google Cloud Vision API
2. Download service account credentials
3. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
4. Replace `MockVisionAPI` with actual Google Cloud Vision client

## 🧪 Testing

Run the test client:
```bash
python test_client.py
```

This will test the API with sample food images and display the results.

## 📊 Nutrition Data Sources

- **USDA FoodData Central**: Official US government nutrition database
- **Comprehensive Analysis**: 100+ nutrients including vitamins, minerals, amino acids
- **Multiple Food Types**: Supports all food categories in USDA database

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
- Use environment variables for all configuration
- Enable HTTPS in production
- Implement rate limiting
- Add authentication if needed
- Monitor API usage and performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud Vision API](https://cloud.google.com/vision)

---

**Built with ❤️ using FastAPI, Python, and USDA FoodData Central**

### 2. Start the Server
```powershell
python start_server.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📋 API Endpoints

### Main Analysis Endpoint
**POST** `/analyze-food`
- Upload a food image and get comprehensive nutritional analysis
- Returns detailed JSON with detected ingredients, nutrition facts, and USDA matches

**Example Request:**
```python
import requests

with open('food_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze-food', files=files)
    result = response.json()
```

### USDA Database Search
**GET** `/search-usda/{query}`
- Search the USDA FoodData Central database
- Example: `/search-usda/salmon?page_size=10`

### Food Details
**GET** `/food-details/{fdc_id}`
- Get detailed nutrition for a specific USDA FDC ID
- Example: `/food-details/175167`

### Batch Analysis
**POST** `/batch-analyze`
- Analyze multiple food images at once (max 10)

### Food Comparison
**GET** `/nutrition-compare?food1=salmon&food2=chicken`
- Compare nutritional values between two foods

## 🔧 Configuration

### Environment Variables (Optional)
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "path/to/service-account.json"
$env:USDA_API_KEY = "your-usda-api-key"
```

### API Keys
The service includes embedded API keys for demonstration. For production use:
1. Get your own Google Cloud Vision API credentials
2. Register for a USDA FoodData Central API key
3. Update the credentials in `app.py`

## 📊 Response Format

### Enhanced JSON Response Example
```json
{
  "food_identification": {
    "primary_dish": "Salmon with Asparagus",
    "confidence": 87.5,
    "alternative_names": ["salmon", "asparagus", "lemon"],
    "food_category": "balanced meal",
    "cuisine_type": null
  },
  "detected_ingredients": [
    {
      "name": "salmon",
      "confidence": 92.3,
      "category": "protein",
      "estimated_portion_grams": 150,
      "usda_fdc_id": 175167,
      "scientific_name": "Salmo salar"
    },
    {
      "name": "asparagus",
      "confidence": 85.7,
      "category": "vegetable",
      "estimated_portion_grams": 80,
      "scientific_name": "Asparagus officinalis"
    }
  ],
  "macronutrients": {
    "calories": {
      "name": "Energy",
      "amount": 328.0,
      "unit": "kcal",
      "percent_daily_value": 16.4
    },
    "protein": {
      "name": "Protein",
      "amount": 32.4,
      "unit": "g",
      "percent_daily_value": 64.8
    },
    "carbohydrates": {
      "name": "Total Carbohydrate",
      "amount": 3.1,
      "unit": "g",
      "percent_daily_value": 1.0
    },
    "total_fat": {
      "name": "Total Fat",
      "amount": 20.2,
      "unit": "g",
      "percent_daily_value": 31.1
    },
    "fiber": {
      "name": "Dietary Fiber",
      "amount": 1.7,
      "unit": "g",
      "percent_daily_value": 6.8
    }
  },
  "usda_matches": [
    {
      "fdc_id": 175167,
      "description": "Fish, salmon, Atlantic, farmed, cooked, dry heat",
      "data_type": "SR Legacy",
      "publication_date": "2019-04-01",
      "match_confidence": 85.0
    }
  ],
  "total_estimated_weight": 230.0,
  "calorie_density": 142.6,
  "analysis_metadata": {
    "analysis_timestamp": "2024-12-19T10:30:00",
    "vision_api_used": true,
    "usda_search_enabled": true,
    "detection_methods": ["google_vision", "color_analysis"],
    "confidence_threshold": 0.3
  }
}
```

## 🔍 Key Features

### Enhanced Detection
- **Google Cloud Vision API**: Advanced image recognition
- **Color-based Analysis**: Detects vegetables by color patterns
- **Shape Recognition**: Identifies asparagus, cherry tomatoes, etc.
- **Multiple Detection Methods**: Combines various AI techniques

### USDA Integration
- **Real-time Search**: Query USDA FoodData Central
- **Comprehensive Data**: Access to Foundation Foods, SR Legacy, FNDDS
- **Detailed Nutrition**: Full nutrient profiles with units and daily values
- **Scientific Names**: Botanical/scientific classification

### Smart Analysis
- **Portion Estimation**: AI-powered serving size calculation
- **Confidence Scoring**: Reliability metrics for all detections
- **Category Classification**: Automatic food categorization
- **Allergen Detection**: Identifies common allergens

### Health Insights
- **Nutrient Density**: Overall nutritional quality score
- **Dietary Flags**: Vegetarian, vegan, gluten-free identification
- **Daily Value Percentages**: Nutrition facts label format
- **Calorie Density**: Energy content per gram

## 🛠️ Development

### Running in Development Mode
```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API
```python
import requests
import json

# Test with a sample image
def test_food_analysis():
    url = "http://localhost:8000/analyze-food"
    
    with open("sample_food.jpg", "rb") as f:
        files = {"file": f}
        data = {
            "enable_detailed_nutrients": True,
            "enable_usda_matching": True,
            "dietary_preferences": json.dumps(["vegetarian"])
        }
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Test USDA search
def test_usda_search():
    url = "http://localhost:8000/search-usda/salmon"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['total_results']} results")
        for food in result['results'][:3]:
            print(f"- {food['description']} (FDC ID: {food['fdc_id']})")

if __name__ == "__main__":
    test_usda_search()
    # test_food_analysis()  # Uncomment when you have a test image
```

## 📝 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test all endpoints directly in your browser.

## 🔧 Troubleshooting

### Common Issues

1. **Google Vision API Errors**
   - Ensure service account JSON is valid
   - Check internet connectivity
   - Verify API quotas

2. **USDA API Timeouts**
   - Check USDA API key validity
   - Verify network connectivity
   - Consider rate limiting

3. **Image Processing Errors**
   - Ensure image files are valid (JPEG, PNG)
   - Check file size limits (max 10MB)
   - Verify image format compatibility

### Performance Optimization

- Use smaller image files when possible
- Enable caching for repeated USDA queries
- Consider batch processing for multiple images
- Monitor API rate limits

## 🚀 Production Deployment

For production deployment, consider:

1. **Environment Variables**: Use proper environment variable management
2. **API Keys**: Secure credential storage
3. **Load Balancing**: Multiple server instances
4. **Caching**: Redis for USDA query caching
5. **Monitoring**: Health checks and logging
6. **Rate Limiting**: Implement request throttling

## 📄 License

This project is provided as-is for educational and demonstration purposes.
