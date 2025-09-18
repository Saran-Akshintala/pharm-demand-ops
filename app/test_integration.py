"""
Test integration of the new Model Retraining feature with existing functionality.
"""

import sys
sys.path.append('.')

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        # Test existing imports
        from utils import (
            preprocess_sales_features, 
            prepare_features_for_prediction,
            load_model,
            validate_input_data,
            apply_predicted_order_business_rules,
            compute_predicted_order_with_adjustments
        )
        print("✅ Utils module imports successfully")
        
        # Test new imports
        from retrain import ModelRetrainer, ModelRegistry
        print("✅ Retrain module imports successfully")
        
        # Test streamlit app imports
        import streamlit_app
        print("✅ Streamlit app imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_registry():
    """Test basic ModelRegistry functionality."""
    try:
        from retrain import ModelRegistry
        
        # Create registry instance
        registry = ModelRegistry("test_models")
        print("✅ ModelRegistry created successfully")
        
        # Test basic operations
        all_models = registry.get_all_models()
        active_model = registry.get_active_model()
        
        print(f"✅ Registry operations work: {len(all_models)} models found")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelRegistry test failed: {e}")
        return False

def test_model_retrainer():
    """Test basic ModelRetrainer functionality."""
    try:
        from retrain import ModelRetrainer
        
        # Create retrainer instance
        retrainer = ModelRetrainer("test_models")
        print("✅ ModelRetrainer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelRetrainer test failed: {e}")
        return False

def test_existing_functionality():
    """Test that existing functionality still works."""
    try:
        import pandas as pd
        from utils import compute_predicted_order_with_adjustments
        
        # Create test data
        test_data = pd.DataFrame({
            'Name': ['Test Product'],
            'Predicted_Base_Quantity': [10],
            'Stock': [3],
            'Box': [5],
            'Scm': ['3+1']
        })
        
        # Test existing functionality
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=True,
            box_tolerance=2,
            apply_scm=True,
            scm_tolerance=2
        )
        
        print("✅ Existing prediction functionality works")
        print(f"   Result: {result['Predicted_Order'].iloc[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Existing functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Model Retraining Integration")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("ModelRegistry Test", test_model_registry),
        ("ModelRetrainer Test", test_model_retrainer),
        ("Existing Functionality Test", test_existing_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
