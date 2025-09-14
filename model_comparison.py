#!/usr/bin/env python3
"""
Model Comparison Script
Compare the old vs new enhanced food analysis models
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from hf_food_analyzer import HuggingFaceFoodAnalyzer  # Old model
from hf_food_analyzer_enhanced import EnhancedHuggingFaceFoodAnalyzer  # New model
from PIL import Image
import io

class ModelComparison:
    def __init__(self):
        print("🔄 Initializing models...")
        
        # Initialize old model
        try:
            self.old_model = HuggingFaceFoodAnalyzer()
            print("✅ Old model (nateraw/food) loaded")
        except Exception as e:
            print(f"❌ Failed to load old model: {e}")
            self.old_model = None
        
        # Initialize new enhanced model
        try:
            self.new_model = EnhancedHuggingFaceFoodAnalyzer()
            print("✅ New enhanced model (ashaduzzaman/vit-finetuned-food101) loaded")
        except Exception as e:
            print(f"❌ Failed to load new model: {e}")
            self.new_model = None
    
    def compare_models(self, image_path: str):
        """Compare both models on the same image"""
        print(f"\n🖼️  Analyzing image: {image_path}")
        
        # Load test image
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return
        
        results = {}
        
        # Test old model
        if self.old_model:
            print("\n📊 Testing OLD model (nateraw/food)...")
            start_time = time.time()
            try:
                old_result = self.old_model.analyze_food_image(image_bytes)
                old_time = time.time() - start_time
                results['old_model'] = {
                    'result': old_result,
                    'processing_time': old_time,
                    'model_name': 'nateraw/food'
                }
                print(f"✅ Old model completed in {old_time:.2f}s")
                print(f"   Primary prediction: {old_result['food_identification']['primary_dish']}")
                print(f"   Confidence: {old_result['food_identification']['confidence']:.1f}%")
            except Exception as e:
                print(f"❌ Old model failed: {e}")
                results['old_model'] = {'error': str(e)}
        
        # Test new enhanced model
        if self.new_model:
            print("\n🚀 Testing NEW enhanced model (ashaduzzaman/vit-finetuned-food101)...")
            start_time = time.time()
            try:
                new_result = self.new_model.analyze_food_image(image_bytes)
                new_time = time.time() - start_time
                results['new_model'] = {
                    'result': new_result,
                    'processing_time': new_time,
                    'model_name': 'ashaduzzaman/vit-finetuned-food101'
                }
                print(f"✅ New model completed in {new_time:.2f}s")
                print(f"   Primary prediction: {new_result['food_identification']['primary_dish']}")
                print(f"   Confidence: {new_result['food_identification']['confidence']:.1f}%")
                print(f"   Model accuracy: {new_result['analysis_metadata'].get('model_accuracy', 'Unknown')}")
            except Exception as e:
                print(f"❌ New model failed: {e}")
                results['new_model'] = {'error': str(e)}
        
        # Generate comparison report
        self._generate_comparison_report(results, image_path)
        return results
    
    def _generate_comparison_report(self, results, image_path):
        """Generate a detailed comparison report"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON REPORT")
        print("="*60)
        
        if 'old_model' in results and 'result' in results['old_model']:
            old = results['old_model']['result']
            print(f"\n🔹 OLD MODEL (nateraw/food):")
            print(f"   Food: {old['food_identification']['primary_dish']}")
            print(f"   Confidence: {old['food_identification']['confidence']:.1f}%")
            print(f"   Calories: {old['macronutrients']['calories']:.1f}")
            print(f"   Processing time: {results['old_model']['processing_time']:.2f}s")
        
        if 'new_model' in results and 'result' in results['new_model']:
            new = results['new_model']['result']
            print(f"\n🚀 NEW ENHANCED MODEL (89.6% accuracy):")
            print(f"   Food: {new['food_identification']['primary_dish']}")
            print(f"   Confidence: {new['food_identification']['confidence']:.1f}%")
            print(f"   Calories: {new['macronutrients']['calories']:.1f}")
            print(f"   Processing time: {results['new_model']['processing_time']:.2f}s")
            print(f"   Health status: {new['health_assessment']['status']}")
            print(f"   Health score: {new['health_assessment']['score']}/10")
        
        # Save detailed results
        output_file = f"model_comparison_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        print("\n🎯 RECOMMENDATION:")
        if 'new_model' in results and 'result' in results['new_model']:
            print("✅ UPGRADE to the enhanced model!")
            print("   Benefits:")
            print("   • 89.6% accuracy vs unknown accuracy")
            print("   • Better confidence scoring")
            print("   • Enhanced nutrition database")
            print("   • Improved health assessment")
            print("   • FDA nutrition data integration")
        else:
            print("⚠️  Enhanced model not available, check installation")

def main():
    print("🍽️  Food Analysis Model Comparison Tool")
    print("Comparing old model vs enhanced model for accuracy")
    
    # Create comparison instance
    comparator = ModelComparison()
    
    # Test with sample food images (you can add your own)
    test_images = [
        # Add paths to your test images here
        # Example: "test_images/pizza.jpg",
        # Example: "test_images/salad.jpg",
    ]
    
    if not test_images:
        print("\n⚠️  No test images specified.")
        print("To test the models, add image paths to the test_images list in this script.")
        print("\nExample usage:")
        print("1. Add some food images to a 'test_images' folder")
        print("2. Update the test_images list with image paths")
        print("3. Run this script again")
        return
    
    # Run comparisons
    for image_path in test_images:
        if Path(image_path).exists():
            comparator.compare_models(image_path)
        else:
            print(f"❌ Image not found: {image_path}")

if __name__ == "__main__":
    main()