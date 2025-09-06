from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import io
import json
from dotenv import load_dotenv
from datetime import datetime
import logging

# Import our Hugging Face food analyzer
from hf_food_analyzer import HuggingFaceFoodAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Food Nutrition Analyzer API with Hugging Face",
    description="AI-powered food image analysis using Hugging Face models for food identification and nutrition calculation",
    version="3.0.0"
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
        """Initialize the enhanced food analyzer with Hugging Face models"""
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        try:
            self.hf_analyzer = HuggingFaceFoodAnalyzer(hf_token=self.hf_token)
            logger.info("Hugging Face Food Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize Hugging Face Food Analyzer: {e}", exc_info=True)
            self.hf_analyzer = None

        logger.info("Enhanced Food Nutrition Analyzer (HF version) initialized successfully")

    async def analyze_food_image(self, image_bytes: bytes, analysis_request: AnalysisRequest) -> NutritionalAnalysis:
        """Main analysis function that processes food images and returns comprehensive nutritional data"""
        try:
            if not self.hf_analyzer:
                raise HTTPException(status_code=503, detail="Food analyzer not available")
            
            # Use Hugging Face analyzer
            analysis_result = self.hf_analyzer.analyze_food_image(image_bytes)
            
            # Convert to our response format
            return self._convert_hf_result_to_response(analysis_result)
            
        except Exception as e:
            logger.error(f"Error in food analysis: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def _convert_hf_result_to_response(self, hf_result: Dict[str, Any]) -> NutritionalAnalysis:
        """Convert Hugging Face analyzer result to our response format"""
        
        # Extract food identification
        food_id = hf_result["food_identification"]
        food_identification = FoodIdentification(
            primary_dish=food_id["primary_dish"],
            confidence=food_id["confidence"],
            alternative_names=food_id["alternative_names"],
            food_category=food_id["food_category"],
            cuisine_type=food_id.get("cuisine_type")
        )
        
        # Extract detected ingredients
        detected_ingredients = []
        for ingredient in hf_result["detected_ingredients"]:
            detected_ingredients.append(DetectedIngredient(
                name=ingredient["name"],
                confidence=ingredient["confidence"],
                estimated_weight=ingredient["estimated_weight"],
                nutritional_category=ingredient["nutritional_category"]
            ))
        
        # Extract macronutrients
        macros = hf_result["macronutrients"]
        macronutrients = MacronutrientBreakdown(
            calories=macros["calories"],
            protein=macros["protein"],
            carbohydrates=macros["carbohydrates"],
            fiber=macros["fiber"],
            sugars=macros["sugars"],
            fat=macros["fat"],
            saturated_fat=macros["saturated_fat"],
            sodium=macros["sodium"]
        )
        
        # Extract vitamins
        vitamins = []
        for vitamin in hf_result["vitamins"]:
            vitamins.append(NutrientInfo(
                name=vitamin["name"],
                amount=vitamin["amount"],
                unit=vitamin["unit"],
                daily_value_percentage=vitamin.get("daily_value_percentage")
            ))
        
        # Extract minerals
        minerals = []
        for mineral in hf_result["minerals"]:
            minerals.append(NutrientInfo(
                name=mineral["name"],
                amount=mineral["amount"],
                unit=mineral["unit"],
                daily_value_percentage=mineral.get("daily_value_percentage")
            ))
        
        return NutritionalAnalysis(
            food_identification=food_identification,
            detected_ingredients=detected_ingredients,
            macronutrients=macronutrients,
            vitamins=vitamins,
            minerals=minerals,
            usda_matches=[],  # Not using USDA in HF version
            total_estimated_weight=hf_result["total_estimated_weight"],
            calorie_density=hf_result["calorie_density"],
            analysis_metadata=hf_result["analysis_metadata"]
        )

# Initialize the analyzer
analyzer = EnhancedFoodAnalyzer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {"service": "Food Nutrition Analyzer API with Hugging Face", "version": "3.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "huggingface_models": analyzer.hf_analyzer is not None,
            "huggingface_token": analyzer.hf_token is not None
        }
    }

@app.post("/analyze-food", response_model=NutritionalAnalysis)
async def analyze_food(
    file: UploadFile = File(..., description="Food image file"),
):
    """Analyze a food image and return comprehensive nutritional information"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await file.read()
        
        # Create a placeholder analysis_request, can be expanded later
        analysis_request = AnalysisRequest()
        
        result = await analyzer.analyze_food_image(image_data, analysis_request)
        
        # Save result to JSON file with timestamp
        timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
        with open(f"analysis_{timestamp}.json", "w") as f:
            json.dump(result.dict(), f, indent=4)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in food analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
