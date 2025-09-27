# 🍽️ Food Nutrition Analyzer API

A powerful FastAPI service that analyzes food images using Google's Gemini AI and provides comprehensive nutritional information.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- Google Gemini API key

### Environment Setup
Set your Gemini API key:
```bash
export GEMINI_API_KEY=your_api_key_here
```

### Installation
```bash
git clone <your-repo-url>
cd food_analyze
pip install -r requirements.txt
```

### Run Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Access
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 API Endpoints

### POST `/analyze`
Upload a food image for analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@your_food_image.jpg"
```

**Response:**
```json
{
  "dish_name": "Grilled Chicken Salad",
  "ingredients": ["chicken", "lettuce", "tomatoes", "olive oil"],
  "nutrition": {
    "calories": {"amount": 250, "unit": "kcal"},
    "protein": {"amount": 25, "unit": "g"},
    "carbs": {"amount": 10, "unit": "g"},
    "fat": {"amount": 15, "unit": "g"},
    "fiber": {"amount": 3, "unit": "g"}
  }
}
```

### GET `/health`
API health status.

### GET `/`
Basic API information.

## 🏗️ Project Structure

```
food_analyze/
├── app.py                    # Main FastAPI application with Gemini AI
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container configuration
├── cloudbuild.yaml          # Google Cloud Build configuration
├── .gcloudignore           # Google Cloud deployment ignore rules
├── .gitignore              # Git ignore rules
└── README.md               # This documentation
```

## 🔧 Configuration

### Required Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Optional Environment Variables
- `PORT`: Server port (default: 8080)
- `LOG_LEVEL`: Logging level (default: INFO)

## 🚀 Deployment

### Google Cloud Run
Set your Gemini API key as a substitution variable:
```bash
gcloud builds submit --config cloudbuild.yaml --substitutions _GEMINI_API_KEY=your_key_here .
```

### Docker
```bash
docker build -t food-analyzer .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key_here food-analyzer
```

## 📄 License

MIT License