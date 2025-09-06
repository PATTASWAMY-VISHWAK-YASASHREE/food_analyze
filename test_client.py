#!/usr/bin/env python3
"""
Food Nutrition Analyzer - Test Client
Interactive client for testing the Food Nutrition Analyzer API
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional

class FoodAnalyzerClient:
    """Client for interacting with the Food Nutrition Analyzer API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if the API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ API server is running!")
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                return True
            else:
                print(f"❌ API server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API server. Make sure it's running on http://localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Error checking API health: {e}")
            return False
    
    def analyze_food_image(self, image_path: str) -> Optional[dict]:
        """Analyze a food image and return nutrition data"""
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': image_file}
                response = self.session.post(f"{self.base_url}/analyze-food", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API returned error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
            return None
    
    def display_results(self, results: dict):
        """Display analysis results in a formatted way"""
        print("\n" + "="*60)
        print("🍽️  FOOD ANALYSIS RESULTS")
        print("="*60)
        
        # Basic info
        print(f"📅 Analysis ID: {results.get('analysis_id', 'N/A')}")
        print(f"⏰ Timestamp: {results.get('timestamp', 'N/A')}")
        
        # Detected foods
        detected_foods = results.get('detected_foods', [])
        if detected_foods:
            print(f"\n🔍 DETECTED FOODS ({len(detected_foods)} items):")
            for i, food in enumerate(detected_foods, 1):
                name = food.get('name', 'Unknown')
                confidence = food.get('confidence', 0) * 100
                weight = food.get('weight_grams', 0)
                category = food.get('nutritional_category', 'Unknown')
                print(f"   {i}. {name.title()} ({confidence:.1f}% confidence)")
                print(f"      Weight: {weight}g | Category: {category}")
        
        # Total nutrition
        nutrition = results.get('total_nutrition', {})
        if nutrition:
            print(f"\n📊 TOTAL NUTRITION:")
            print(f"   🔥 Calories: {nutrition.get('calories', 0):.1f} kcal")
            print(f"   🥩 Protein: {nutrition.get('protein', 0):.1f}g")
            print(f"   🍞 Carbs: {nutrition.get('carbohydrates', 0):.1f}g")
            print(f"   🥑 Fat: {nutrition.get('fat', 0):.1f}g")
            print(f"   🧂 Sodium: {nutrition.get('sodium', 0):.1f}mg")
            print(f"   🌾 Fiber: {nutrition.get('fiber', 0):.1f}g")
        
        # USDA matches
        usda_matches = results.get('usda_matches', [])
        if usda_matches:
            print(f"\n🏛️  USDA DATABASE MATCHES ({len(usda_matches)} found):")
            for i, match in enumerate(usda_matches[:3], 1):  # Show top 3
                description = match.get('description', 'Unknown')
                brand = match.get('brandName', '')
                if brand:
                    print(f"   {i}. {description} ({brand})")
                else:
                    print(f"   {i}. {description}")
        
        # Health insights
        health_insights = results.get('health_insights', {})
        if health_insights:
            print(f"\n💡 HEALTH INSIGHTS:")
            if 'allergen_warnings' in health_insights:
                allergens = health_insights['allergen_warnings']
                if allergens:
                    print(f"   ⚠️  Potential allergens: {', '.join(allergens)}")
            
            if 'dietary_tags' in health_insights:
                tags = health_insights['dietary_tags']
                if tags:
                    print(f"   🏷️  Dietary tags: {', '.join(tags)}")
        
        print("="*60)

def main():
    """Main interactive function"""
    print("🍽️  Food Nutrition Analyzer - Test Client")
    print("=========================================")
    
    # Initialize client
    client = FoodAnalyzerClient()
    
    # Health check
    if not client.health_check():
        print("\n💡 To start the server, run:")
        print("   uvicorn app:app --host 0.0.0.0 --port 8000")
        return
    
    # Interactive mode
    while True:
        print("\n" + "="*50)
        print("Select an option:")
        print("1. Analyze a food image")
        print("2. Check API health")
        print("3. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Get image path from user
            image_path = input("\nEnter the path to your food image: ").strip()
            
            # Handle common path formats
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]  # Remove quotes
            
            if not image_path:
                print("❌ No image path provided")
                continue
            
            # Convert to absolute path
            image_path = os.path.abspath(image_path)
            
            print(f"\n🔄 Analyzing image: {image_path}")
            results = client.analyze_food_image(image_path)
            
            if results:
                client.display_results(results)
                
                # Ask if user wants to save results
                save_choice = input("\nSave results to file? (y/N): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    filename = f"analysis_results_{results.get('analysis_id', 'unknown')}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"✅ Results saved to {filename}")
            
        elif choice == "2":
            client.health_check()
            
        elif choice == "3":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
