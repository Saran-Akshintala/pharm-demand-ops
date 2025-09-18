"""
Test that the existing model gets registered correctly.
"""

import sys
sys.path.append('.')

from retrain import ModelRegistry
import os

def test_existing_model_registration():
    """Test that existing model gets registered automatically."""
    
    # Clean up any existing registry for fresh test
    registry_path = "models/registry.json"
    if os.path.exists(registry_path):
        os.remove(registry_path)
        print("🗑️ Cleaned up existing registry for fresh test")
    
    # Create new registry - should auto-detect existing model
    print("🔍 Creating ModelRegistry...")
    registry = ModelRegistry("models")
    
    # Check if model was registered
    active_model = registry.get_active_model()
    all_models = registry.get_all_models()
    
    print(f"📊 Registry status:")
    print(f"   Total models: {len(all_models)}")
    print(f"   Active model: {active_model['version'] if active_model else 'None'}")
    
    if active_model:
        print(f"✅ SUCCESS: Existing model registered as {active_model['version']}")
        print(f"   Path: {active_model['path']}")
        print(f"   Metrics: {active_model.get('metrics', {})}")
        return True
    else:
        print("❌ FAILED: No active model found")
        return False

if __name__ == "__main__":
    print("🧪 Testing Existing Model Registration")
    print("=" * 50)
    
    success = test_existing_model_registration()
    
    if success:
        print("\n🎉 Test passed! Existing model registration works correctly.")
    else:
        print("\n❌ Test failed! Check the implementation.")
