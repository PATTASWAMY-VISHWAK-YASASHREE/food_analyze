from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import io
import json
import aiohttp
from dotenv import load_dotenv
from google.cloud import vision
from datetime import datetime
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
    match_confidence: Optional[float] = Field(None, description="Match confidence percentage")
    food_category: str = Field(..., description="USDA food category")
    ingredients: Optional[str] = Field(None, description="Ingredient list")

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
        self.usda_api_key = os.getenv("USDA_API_KEY")
        if not self.usda_api_key:
            logger.warning("USDA_API_KEY environment variable not set. Some features may be limited.")

        try:
            credentials_path = "gcp_credentials.json"
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Google Cloud credentials file not found at: {credentials_path}")

            self.vision_client = vision.ImageAnnotatorClient.from_service_account_file(credentials_path)
            logger.info("Google Cloud Vision API initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize Google Cloud Vision API: {e}", exc_info=True)
            self.vision_client = None

        self.usda_search_endpoint = "https://api.nal.usda.gov/fdc/v1/foods/search"
        self.usda_details_endpoint = "https://api.nal.usda.gov/fdc/v1/food"
        self.label_detection_threshold = 0.3

        logger.info("Enhanced Food Nutrition Analyzer initialized successfully")

    async def analyze_food_image(self, image_bytes: bytes, analysis_request: AnalysisRequest) -> NutritionalAnalysis:
        """Main analysis function that processes food images and returns comprehensive nutritional data"""
        try:
            detected_items = await self._detect_food_items(image_bytes)
            food_identification = self._identify_primary_dish(detected_items)
            detailed_ingredients, nutrition_data, usda_matches = await self._analyze_ingredients_and_nutrition(detected_items)

            return self._build_analysis_response(
                food_identification, detailed_ingredients, nutrition_data, usda_matches
            )
        except Exception as e:
            logger.error(f"Error in food analysis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def _detect_food_items(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Detect food items using Google Cloud Vision API"""
        if not self.vision_client:
            raise HTTPException(status_code=503, detail="Vision API not available")

        image = vision.Image(content=image_bytes)
        response = self.vision_client.label_detection(image=image)

        return [
            {'name': label.description.lower(), 'confidence': round(label.score * 100, 2)}
            for label in response.label_annotations
        ]

    def _identify_primary_dish(self, detected_items: List[Dict[str, Any]]) -> FoodIdentification:
        """Identify the primary dish from detected items"""
        if not detected_items:
            return FoodIdentification(primary_dish="unknown", confidence=0.0, food_category="unknown")

        primary = max(detected_items, key=lambda x: x['confidence'])

        return FoodIdentification(
            primary_dish=primary['name'],
            confidence=primary['confidence'],
            alternative_names=[item['name'] for item in detected_items[:5]],
            food_category="unknown" # Category can be improved
        )

    async def _analyze_ingredients_and_nutrition(self, detected_items: List[Dict[str, Any]]) -> (List[DetectedIngredient], Dict[str, Any], List[USDAFoodMatch]):
        """Analyze ingredients, get nutrition data, and return all results."""
        ingredients = []
        usda_matches = []
        total_nutrition = {
            "calories": 0.0, "protein": 0.0, "carbohydrates": 0.0, "fiber": 0.0,
            "sugars": 0.0, "fat": 0.0, "saturated_fat": 0.0, "sodium": 0.0
        }
        total_weight = 0.0

        async def process_item(item):
            estimated_weight = self._estimate_ingredient_weight(item['name'])
            matches = await self._search_usda_database(item['name'])

            category = "other"
            best_match = None
            if matches:
                best_match = matches[0]
                usda_matches.append(best_match)
                category = best_match.food_category

            ingredient = DetectedIngredient(
                name=item['name'],
                confidence=item['confidence'],
                estimated_weight=estimated_weight,
                nutritional_category=category
            )
            ingredients.append(ingredient)

            if best_match:
                details = await self._get_usda_food_details(best_match.fdc_id)
                if details:
                    weight_factor = estimated_weight / 100.0
                    nonlocal total_weight
                    total_weight += estimated_weight
                    for nutrient in details.get('foodNutrients', []):
                        name = nutrient.get('nutrient', {}).get('name', '').lower()
                        amount = nutrient.get('amount', 0.0) * weight_factor

                        if "energy" in name and "kcal" in nutrient.get('nutrient', {}).get('unitName', '').lower():
                            total_nutrition["calories"] += amount
                        elif "protein" in name:
                            total_nutrition["protein"] += amount
                        elif "carbohydrate, by difference" in name:
                            total_nutrition["carbohydrates"] += amount
                        elif "fiber, total dietary" in name:
                            total_nutrition["fiber"] += amount
                        elif "sugars, total including nlea" in name:
                            total_nutrition["sugars"] += amount
                        elif "total lipid (fat)" in name:
                            total_nutrition["fat"] += amount
                        elif "fatty acids, total, saturated" in name:
                            total_nutrition["saturated_fat"] += amount
                        elif "sodium, na" in name:
                            total_nutrition["sodium"] += amount

        await asyncio.gather(*(process_item(item) for item in detected_items))

        nutrition_data = {'macronutrients': total_nutrition, 'total_weight': total_weight, 'vitamins': [], 'minerals': []}
        return ingredients, nutrition_data, usda_matches

    def _estimate_ingredient_weight(self, ingredient_name: str) -> float:
        """
        Estimate ingredient weight based on typical portions.
        Note: This is a simplified estimation and can be improved with a more sophisticated model.
        """
        weight_estimates = {
            "salmon": 150.0, "chicken": 120.0, "beef": 100.0,
            "asparagus": 80.0, "broccoli": 60.0, "tomato": 40.0,
            "vegetables": 80.0, "rice": 150.0, "potato": 200.0,
        }
        return weight_estimates.get(ingredient_name, 75.0)

    async def _search_usda_database(self, query: str) -> List[USDAFoodMatch]:
        """Search USDA database for food matches"""
        if not self.usda_api_key: return []
        params = {'query': query, 'api_key': self.usda_api_key, 'pageSize': 5}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.usda_search_endpoint, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return [
                        USDAFoodMatch(
                            fdc_id=food.get('fdcId'),
                            description=food.get('description', ''),
                            food_category=food.get('foodCategory', 'Unknown'),
                            ingredients=food.get('ingredients')
                        ) for food in data.get('foods', [])
                    ]
            except Exception as e:
                logger.error(f"USDA search failed for query '{query}': {e}")
                return []

    async def _get_usda_food_details(self, fdc_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed nutrition for a specific USDA FDC ID"""
        if not self.usda_api_key: return None
        url = f"{self.usda_details_endpoint}/{fdc_id}?api_key={self.usda_api_key}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                logger.error(f"Failed to get USDA details for FDC ID {fdc_id}: {e}")
                return None

    def _build_analysis_response(self, food_identification: FoodIdentification,
                                ingredients: List[DetectedIngredient],
                                nutrition_data: Dict[str, Any],
                                usda_matches: List[USDAFoodMatch]) -> NutritionalAnalysis:
        """Build the final analysis response"""
        macros = nutrition_data['macronutrients']
        total_weight = nutrition_data['total_weight']

        macronutrients = MacronutrientBreakdown(
            calories=round(macros.get('calories', 0), 1),
            protein=round(macros.get('protein', 0), 1),
            carbohydrates=round(macros.get('carbohydrates', 0), 1),
            fiber=round(macros.get('fiber', 0), 1),
            sugars=round(macros.get('sugars', 0), 1),
            fat=round(macros.get('fat', 0), 1),
            saturated_fat=round(macros.get('saturated_fat', 0), 1),
            sodium=round(macros.get('sodium', 0), 1)
        )

        return NutritionalAnalysis(
            food_identification=food_identification,
            detected_ingredients=ingredients,
            macronutrients=macronutrients,
            usda_matches=usda_matches,
            total_estimated_weight=round(total_weight, 2),
            calorie_density=round(macros.get('calories', 0) / max(total_weight, 1) * 100, 2),
            analysis_metadata={'detection_methods': ['google_vision'], 'confidence_threshold': self.label_detection_threshold}
        )

# Initialize the analyzer
analyzer = EnhancedFoodAnalyzer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {"service": "Enhanced Food Nutrition Analyzer API", "version": "2.0.0"}

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
):
    """Analyze a food image and return comprehensive nutritional information"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await file.read()

    # Create a placeholder analysis_request, can be expanded later
    analysis_request = AnalysisRequest()

    return await analyzer.analyze_food_image(image_data, analysis_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
