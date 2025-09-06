from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import os
import io
import json
import base64
import requests
import aiohttp
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mock Vision API implementation (replace with actual Google Cloud Vision if needed)
class MockVisionAPI:
    def analyze_food_image(self, image_content):
        # Mock food detection - in real implementation, use Google Cloud Vision
        return {
            "detected_foods": ["salmon", "asparagus", "vegetables"],
            "confidence_scores": [0.95, 0.87, 0.82],
            "text_annotations": [],
            "label_annotations": [
                {"description": "food", "score": 0.98},
                {"description": "dish", "score": 0.95},
                {"description": "meal", "score": 0.92}
            ]
        }

vision_client = MockVisionAPI()
import asyncio
import aiofiles
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Food Nutrition Analyzer API",
    description="AI-powered food image analysis with comprehensive nutritional data from USDA database",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class NutrientInfo(BaseModel):
    name: str = Field(..., description="Nutrient name")
    amount: float = Field(..., description="Amount in grams or milligrams")
    unit: str = Field(..., description="Unit of measurement")
    daily_value_percentage: Optional[float] = Field(None, description="Percentage of daily value")

class MacronutrientBreakdown(BaseModel):
    calories: float = Field(..., description="Total calories")
    protein: float = Field(..., description="Protein in grams")
    carbohydrates: float = Field(..., description="Carbohydrates in grams")
    fiber: float = Field(..., description="Fiber in grams")
    sugars: float = Field(..., description="Sugars in grams")
    fat: float = Field(..., description="Total fat in grams")
    saturated_fat: float = Field(..., description="Saturated fat in grams")
    sodium: float = Field(..., description="Sodium in milligrams")

class DetectedIngredient(BaseModel):
    name: str = Field(..., description="Ingredient name")
    confidence: float = Field(..., description="Detection confidence percentage")
    estimated_weight: float = Field(..., description="Estimated weight in grams")
    nutritional_category: str = Field(..., description="Category (protein, vegetable, etc.)")

class FoodIdentification(BaseModel):
    primary_dish: str = Field(..., description="Main identified dish")
    confidence: float = Field(..., description="Overall confidence percentage")
    alternative_names: List[str] = Field(default_factory=list, description="Alternative dish names")
    food_category: str = Field(..., description="Category of the dish")
    cuisine_type: Optional[str] = Field(None, description="Cuisine type if detected")

class USDAFoodMatch(BaseModel):
    fdc_id: int = Field(..., description="USDA Food Data Central ID")
    description: str = Field(..., description="Food description")
    match_confidence: float = Field(..., description="Match confidence percentage")
    food_category: str = Field(..., description="USDA food category")
    ingredients: Optional[str] = Field(None, description="Ingredient list")

class HealthMetrics(BaseModel):
    glycemic_load: Optional[float] = Field(None, description="Estimated glycemic load")
    protein_quality_score: Optional[float] = Field(None, description="Protein quality score")
    nutrient_density_score: float = Field(..., description="Overall nutrient density")
    allergen_warnings: List[str] = Field(default_factory=list, description="Potential allergens")
    dietary_flags: List[str] = Field(default_factory=list, description="Dietary compliance flags")

class NutritionalAnalysis(BaseModel):
    food_identification: FoodIdentification = Field(..., description="Identified food information")
    detected_ingredients: List[DetectedIngredient] = Field(..., description="Individual ingredients detected")
    macronutrients: MacronutrientBreakdown = Field(..., description="Macronutrient breakdown")
    vitamins: List[NutrientInfo] = Field(default_factory=list, description="Vitamin content")
    minerals: List[NutrientInfo] = Field(default_factory=list, description="Mineral content")
    usda_matches: List[USDAFoodMatch] = Field(default_factory=list, description="USDA database matches")
    total_estimated_weight: float = Field(..., description="Total estimated weight in grams")
    calorie_density: float = Field(..., description="Calories per 100g")
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")

class AnalysisRequest(BaseModel):
    enable_detailed_nutrients: bool = Field(True, description="Include detailed vitamin/mineral analysis")
    enable_usda_matching: bool = Field(True, description="Enable USDA database matching")
    portion_size_grams: Optional[float] = Field(None, description="Override portion size estimation")
    dietary_preferences: List[str] = Field(default_factory=list, description="Dietary preferences")
    detailed_analysis: bool = Field(True, description="Enable detailed analysis")
    include_allergens: bool = Field(True, description="Include allergen detection")

class EnhancedFoodAnalyzer:
    def __init__(self):
        """Initialize the enhanced food analyzer with USDA integration"""
        # API Configuration - Load from environment variable
        self.usda_api_key = os.getenv("USDA_API_KEY")
        if not self.usda_api_key:
            logger.warning("USDA_API_KEY environment variable not set. Some features may be limited.")
        
        # Initialize Vision API (using mock for now)
        try:
            self.vision_client = MockVisionAPI()
            logger.info("Mock Vision API initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize Vision API: {e}")
            self.vision_client = None

        # USDA API endpoints
        self.usda_search_endpoint = "https://api.nal.usda.gov/fdc/v1/foods/search"
        self.usda_details_endpoint = "https://api.nal.usda.gov/fdc/v1/food"
        
        # Enhanced detection parameters
        self.label_detection_threshold = 0.3
        
        # Load nutrition database
        self.nutrition_db = self._load_nutrition_database()
        
        logger.info("Enhanced Food Nutrition Analyzer initialized successfully")

    def _load_nutrition_database(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive nutrition database"""
        return {
            # Proteins
            "salmon": {"category": "protein", "calories_per_100g": 208, "protein": 25.4, "fat": 12.4},
            "chicken": {"category": "protein", "calories_per_100g": 239, "protein": 27.3, "fat": 13.6},
            "beef": {"category": "protein", "calories_per_100g": 250, "protein": 26.1, "fat": 15.4},
            # Vegetables
            "asparagus": {"category": "vegetable", "calories_per_100g": 20, "protein": 2.2, "carbs": 3.9},
            "broccoli": {"category": "vegetable", "calories_per_100g": 34, "protein": 2.8, "carbs": 7.0},
            "tomato": {"category": "vegetable", "calories_per_100g": 18, "protein": 0.9, "carbs": 3.9},
            "vegetables": {"category": "vegetable", "calories_per_100g": 25, "protein": 2.0, "carbs": 5.0},
        }

    async def analyze_food_image(self, image_bytes: bytes, analysis_request: AnalysisRequest) -> NutritionalAnalysis:
        """Main analysis function that processes food images and returns comprehensive nutritional data"""
        try:
            # Step 1: Detect food items
            detected_items = await self._detect_food_items(image_bytes)
            
            # Step 2: Identify primary dish and ingredients
            food_identification = await self._identify_primary_dish(detected_items)
            
            # Step 3: Get detailed ingredient analysis
            detailed_ingredients = await self._analyze_ingredients(detected_items, image_bytes)
            
            # Step 4: Calculate nutrition
            nutrition_data = await self._calculate_nutrition(detailed_ingredients, analysis_request)
            
            # Step 5: Get USDA matches if enabled
            usda_matches = []
            if analysis_request.enable_usda_matching:
                usda_matches = await self._search_usda_database(food_identification.primary_dish)
            
            # Step 6: Build response
            return self._build_analysis_response(
                food_identification, detailed_ingredients, nutrition_data, usda_matches
            )
            
        except Exception as e:
            logger.error(f"Error in food analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def _detect_food_items(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Enhanced food detection using mock vision API"""
        if not self.vision_client:
            raise HTTPException(status_code=503, detail="Vision API not available")
        
        try:
            # Use mock vision API
            response = self.vision_client.analyze_food_image(image_bytes)
            
            detected_items = []
            
            # Process detected foods from mock API
            for i, food in enumerate(response["detected_foods"]):
                confidence = response["confidence_scores"][i] if i < len(response["confidence_scores"]) else 0.8
                detected_items.append({
                    'name': food.lower(),
                    'confidence': round(confidence * 100, 2),
                    'type': 'food_item',
                    'bounding_box': None
                })
            
            # Process label annotations from mock API
            for label in response.get("label_annotations", []):
                detected_items.append({
                    'name': label["description"].lower(),
                    'confidence': round(label["score"] * 100, 2),
                    'type': 'label',
                    'bounding_box': None
                })
            
            return detected_items
            
        except Exception as e:
            logger.error(f"Vision API detection failed: {e}")
            return []

    async def _identify_primary_dish(self, detected_items: List[Dict[str, Any]]) -> FoodIdentification:
        """Identify the primary dish from detected items"""
        
        if not detected_items:
            return FoodIdentification(
                primary_dish="unknown food",
                confidence=0.0,
                alternative_names=[],
                food_category="unknown",
                cuisine_type=None
            )
        
        # Find the highest confidence food item
        food_items = [item for item in detected_items if item['type'] == 'food_item']
        
        if food_items:
            primary = max(food_items, key=lambda x: x['confidence'])
            primary_dish = primary['name']
            confidence = primary['confidence']
        else:
            primary_dish = detected_items[0]['name']
            confidence = detected_items[0]['confidence']
        
        # Determine food category
        food_category = "main course"
        if any(item['name'] in ['vegetables', 'broccoli', 'asparagus'] for item in detected_items):
            if any(item['name'] in ['salmon', 'chicken', 'beef'] for item in detected_items):
                food_category = "balanced meal"
            else:
                food_category = "vegetarian dish"
        
        return FoodIdentification(
            primary_dish=primary_dish,
            confidence=round(confidence, 2),
            alternative_names=[item['name'] for item in detected_items[:3]],
            food_category=food_category,
            cuisine_type=None
        )

    async def _analyze_ingredients(self, detected_items: List[Dict[str, Any]], image_bytes: bytes) -> List[DetectedIngredient]:
        """Analyze individual ingredients with portion estimation"""
        
        ingredients = []
        
        for item in detected_items:
            if item['type'] == 'food_item':
                name = item['name']
                confidence = item['confidence']
                
                # Estimate weight based on food type and image analysis
                estimated_weight = self._estimate_ingredient_weight(name)
                
                # Determine nutritional category
                category = "other"
                if name in self.nutrition_db:
                    category = self.nutrition_db[name].get('category', 'other')
                
                ingredients.append(DetectedIngredient(
                    name=name,
                    confidence=confidence,
                    estimated_weight=estimated_weight,
                    nutritional_category=category
                ))
        
        return ingredients

    def _estimate_ingredient_weight(self, ingredient_name: str) -> float:
        """Estimate ingredient weight based on typical portions"""
        weight_estimates = {
            "salmon": 150.0,  # Typical fillet
            "chicken": 120.0,
            "beef": 100.0,
            "asparagus": 80.0,  # Several spears
            "broccoli": 60.0,
            "tomato": 40.0,
            "vegetables": 80.0
        }
        return weight_estimates.get(ingredient_name, 50.0)

    async def _calculate_nutrition(self, ingredients: List[DetectedIngredient], analysis_request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate comprehensive nutrition from ingredients"""
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        total_weight = 0
        
        for ingredient in ingredients:
            if ingredient.name in self.nutrition_db:
                data = self.nutrition_db[ingredient.name]
                weight_factor = ingredient.estimated_weight / 100.0  # Per 100g
                
                total_calories += data.get('calories_per_100g', 0) * weight_factor
                total_protein += data.get('protein', 0) * weight_factor
                total_carbs += data.get('carbs', 0) * weight_factor
                total_fat += data.get('fat', 0) * weight_factor
                total_weight += ingredient.estimated_weight
        
        return {
            'macronutrients': {
                'calories': total_calories,
                'protein': total_protein,
                'carbohydrates': total_carbs,
                'fat': total_fat,
                'fiber': total_carbs * 0.1,  # Estimate
                'sugars': total_carbs * 0.2,  # Estimate
                'saturated_fat': total_fat * 0.3,  # Estimate
                'sodium': 100  # Estimate
            },
            'total_weight': total_weight,
            'vitamins': [],
            'minerals': []
        }

    async def _search_usda_database(self, query: str) -> List[USDAFoodMatch]:
        """Search USDA database for food matches"""
        try:
            url = f"{self.usda_search_endpoint}?api_key={self.usda_api_key}&query={query}&pageSize=5"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches = []
                        
                        for food in data.get('foods', []):
                            matches.append(USDAFoodMatch(
                                fdc_id=food.get('fdcId'),
                                description=food.get('description', ''),
                                match_confidence=85.0,  # Mock confidence
                                food_category=food.get('foodCategory', 'Unknown'),
                                ingredients=food.get('ingredients', None)
                            ))
                        
                        return matches[:3]  # Return top 3 matches
        except Exception as e:
            logger.error(f"USDA search failed: {e}")
        
        return []

    def _build_analysis_response(self, food_identification: FoodIdentification, 
                                ingredients: List[DetectedIngredient], 
                                nutrition_data: Dict[str, Any], 
                                usda_matches: List[USDAFoodMatch]) -> NutritionalAnalysis:
        """Build the final analysis response"""
        
        macros = nutrition_data['macronutrients']
        
        macronutrients = MacronutrientBreakdown(
            calories=round(macros['calories'], 1),
            protein=round(macros['protein'], 1),
            carbohydrates=round(macros['carbohydrates'], 1),
            fiber=round(macros['fiber'], 1),
            sugars=round(macros['sugars'], 1),
            fat=round(macros['fat'], 1),
            saturated_fat=round(macros['saturated_fat'], 1),
            sodium=round(macros['sodium'], 1)
        )
        
        return NutritionalAnalysis(
            food_identification=food_identification,
            detected_ingredients=ingredients,
            macronutrients=macronutrients,
            vitamins=nutrition_data.get('vitamins', []),
            minerals=nutrition_data.get('minerals', []),
            usda_matches=usda_matches,
            total_estimated_weight=nutrition_data['total_weight'],
            calorie_density=round(macros['calories'] / max(nutrition_data['total_weight'], 1) * 100, 2),
            analysis_metadata={
                'detection_methods': ['mock_vision', 'color_analysis'],
                'confidence_threshold': self.label_detection_threshold
            }
        )

# Initialize the analyzer
analyzer = EnhancedFoodAnalyzer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced Food Nutrition Analyzer API",
        "version": "2.0.0",
        "description": "AI-powered food image analysis with USDA database integration",
        "endpoints": {
            "analyze": "/analyze-food",
            "search_usda": "/search-usda/{query}",
            "food_details": "/food-details/{fdc_id}",
            "health": "/health"
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "vision_api": analyzer.vision_client is not None,
            "usda_api": analyzer.usda_api_key is not None
        }
    }

@app.post("/analyze-food", response_model=NutritionalAnalysis)
async def analyze_food(
    file: UploadFile = File(..., description="Food image file"),
    enable_detailed_nutrients: bool = True,
    enable_usda_matching: bool = True,
    portion_size_grams: Optional[float] = None,
    dietary_preferences: str = "[]"
):
    """
    Analyze a food image and return comprehensive nutritional information
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Parse dietary preferences
        try:
            dietary_prefs = json.loads(dietary_preferences) if dietary_preferences != "[]" else []
        except json.JSONDecodeError:
            dietary_prefs = []
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            enable_detailed_nutrients=enable_detailed_nutrients,
            enable_usda_matching=enable_usda_matching,
            portion_size_grams=portion_size_grams,
            dietary_preferences=dietary_prefs,
            detailed_analysis=True,
            include_allergens=True
        )
        
        # Analyze the image
        result = await analyzer.analyze_food_image(image_data, analysis_request)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_food endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
