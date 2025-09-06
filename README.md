# 🍽️ Enhanced Food Nutrition Analyzer API

A powerful FastAPI service that analyzes food images using AI and provides comprehensive nutritional information integrated with the USDA FoodData Central database.

## ✨ Features

- **📸 Multi-Food Detection**: Identifies multiple ingredients in a single image using Google Cloud Vision API.
- **📊 Comprehensive Nutrition Data**: Detailed macronutrients, vitamins, and minerals from the USDA database.
- **🏛️ USDA Integration**: Official government nutrition database.
- **🚀 Fast API**: High-performance async API with automatic documentation.
- **📱 Easy Integration**: RESTful API with JSON responses.
- **🔒 Secure**: Environment-based configuration for API keys.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Cloud Account with Vision API enabled.
- USDA API Key ([Get one here](https://fdc.nal.usda.gov/api-guide.html)).

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd food-nutrition-analyzer
pip install -r requirements.txt
```

### 2. Environment Setup
1.  **USDA API Key**:
    Create a `.env` file in the root of the project and add your USDA API key to it:
    ```
    USDA_API_KEY=your_actual_api_key_here
    ```
2.  **Google Cloud Credentials**:
    Create a file named `gcp_credentials.json` in the root of the project and paste your Google Cloud service account credentials into it.

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

### GET `/health`
API health status and system information.

## 🏗️ Project Structure

```
food-nutrition-analyzer/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables for USDA API key
├── gcp_credentials.json   # Google Cloud credentials
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── test_app.py            # Automated tests
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `USDA_API_KEY` | Yes | USDA FoodData Central API key |

### Google Cloud Credentials
The application uses the `gcp_credentials.json` file to authenticate with the Google Cloud Vision API.

## 🧪 Testing

The project includes a suite of automated tests using `pytest`. To run the tests, execute the following command in the root of the project:
```bash
pytest
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
- Use a secure method for managing secrets, such as a secret manager.
- Enable HTTPS in production.
- Implement rate limiting.
- Add authentication if needed.
- Monitor API usage and performance.

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
