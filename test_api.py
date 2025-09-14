#!/usr/bin/env python3
"""
API Test Script - Test the enhanced food analysis API
"""

import requests
import json
from io import BytesIO
from PIL import Image
import base64

def create_sample_food_image():
    """Create a simple sample food image for testing"""
    # Create a simple colored image that might represent food
    img = Image.new('RGB', (224, 224), color=(255, 180, 100))  # Orange-ish color
    
    # Save to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_api_endpoints():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Enhanced Food Analysis API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\n📍 Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Status: {response.status_code}")
        print(f"📋 Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Health check
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Status: {response.status_code}")
        print(f"📋 Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Food analysis with sample image
    print("\n🍎 Testing food analysis endpoint...")
    try:
        # Create sample image
        sample_image = create_sample_food_image()
        
        # Prepare the file for upload
        files = {
            'file': ('sample_food.jpg', sample_image, 'image/jpeg')
        }
        
        print("📤 Uploading sample image for analysis...")
        response = requests.post(f"{base_url}/analyze-food", files=files)
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n🎉 Analysis Results:")
            print("-" * 30)
            
            # Classification results
            classification = result.get('classification', {})
            print(f"🏷️  Primary: {classification.get('primary_classification')}")
            print(f"🔄 Alternative: {classification.get('alternative_classification')}")
            print(f"📊 Confidence: {classification.get('confidence', 0):.2%}")
            
            # Nutrition info
            nutrition = result.get('nutrition', {})
            print(f"\n🥗 Nutrition (per 100g):")
            print(f"   Calories: {nutrition.get('calories')} kcal")
            print(f"   Protein: {nutrition.get('protein')}g")
            print(f"   Carbs: {nutrition.get('carbs')}g")
            print(f"   Fat: {nutrition.get('fat')}g")
            
            # Health assessment
            health = result.get('health_assessment', {})
            print(f"\n❤️  Health Score: {health.get('health_score')}/10")
            print(f"📝 Assessment: {health.get('assessment')}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing food analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_api_endpoints()
    
    if success:
        print("\n🎉 All API tests passed!")
        print("\n🌐 Test your API with the web UI:")
        print("   1. Open: http://localhost:3000/test_ui.html")
        print("   2. Upload a food image")
        print("   3. Click 'Analyze Food'")
        print("\n📚 API Documentation:")
        print("   • Interactive docs: http://localhost:8000/docs")
        print("   • OpenAPI spec: http://localhost:8000/openapi.json")
    else:
        print("\n❌ Some tests failed. Check the server logs.")