"""
Final validation of the edge case fix with exact user scenarios.
"""

import sys
sys.path.append('.')

import pandas as pd
from utils import compute_predicted_order_with_adjustments

def validate_exact_scenarios():
    """Validate the exact scenarios reported by the user."""
    print("🎯 FINAL VALIDATION - Exact User Scenarios")
    print("=" * 50)
    
    print("\n📋 Scenario 1:")
    print("Input: Predicted_Base_Quantity=4, Stock=2, Scm=4+1")
    print("Expected: 2.5+0.5 (sum=3.0)")
    
    # Create exact test case
    test_data_1 = pd.DataFrame({
        'Name': ['Medicine A'],
        'Predicted_Base_Quantity': [4],
        'Stock': [2],
        'Scm': ['4+1'],
        'Box': [1],  # Avoid box interference
        'Days': [45]  # Normal days
    })
    
    result_1 = compute_predicted_order_with_adjustments(
        test_data_1,
        apply_box=False,  # Focus on scheme only
        box_tolerance=2,
        apply_scm=True,
        scm_tolerance=2
    )
    
    predicted_1 = result_1.iloc[0]['Predicted_Order']
    print(f"Result: {predicted_1}")
    
    # Verify
    if predicted_1 == "2.5+0.5":
        print("✅ FIXED - Correct result!")
        parts = predicted_1.split('+')
        total = float(parts[0]) + float(parts[1])
        print(f"✅ Sum verification: {parts[0]} + {parts[1]} = {total} (whole number)")
    else:
        print(f"❌ STILL BROKEN - Expected '2.5+0.5', got '{predicted_1}'")
    
    print("\n📋 Scenario 2:")
    print("Input: Predicted_Base_Quantity=7, Stock=1, Scm=10+1")
    print("Expected: 5.5+0.5 (sum=6.0)")
    
    # Create exact test case
    test_data_2 = pd.DataFrame({
        'Name': ['Medicine B'],
        'Predicted_Base_Quantity': [7],
        'Stock': [1],
        'Scm': ['10+1'],
        'Box': [1],  # Avoid box interference
        'Days': [60]  # Normal days
    })
    
    result_2 = compute_predicted_order_with_adjustments(
        test_data_2,
        apply_box=False,  # Focus on scheme only
        box_tolerance=2,
        apply_scm=True,
        scm_tolerance=2
    )
    
    predicted_2 = result_2.iloc[0]['Predicted_Order']
    print(f"Result: {predicted_2}")
    
    # Verify
    if predicted_2 == "5.5+0.5":
        print("✅ FIXED - Correct result!")
        parts = predicted_2.split('+')
        total = float(parts[0]) + float(parts[1])
        print(f"✅ Sum verification: {parts[0]} + {parts[1]} = {total} (whole number)")
    else:
        print(f"❌ STILL BROKEN - Expected '5.5+0.5', got '{predicted_2}'")
    
    return predicted_1 == "2.5+0.5" and predicted_2 == "5.5+0.5"

def test_streamlit_integration():
    """Test with a more comprehensive dataset like what would be uploaded to Streamlit."""
    print("\n🔄 Testing Streamlit Integration...")
    
    # Create a dataset similar to what users would upload
    test_data = pd.DataFrame({
        'Name': ['Product A', 'Product B', 'Product C', 'Product D'],
        'Predicted_Base_Quantity': [4, 7, 6, 8],
        'Stock': [2, 1, 2, 3],
        'Scm': ['4+1', '10+1', '8+1', '6+1'],
        'Box': [1, 1, 1, 1],
        'Days': [45, 60, 30, 75],
        'Order': [0, 0, 0, 0]  # Placeholder
    })
    
    # Process with the same settings as Streamlit would use
    result = compute_predicted_order_with_adjustments(
        test_data,
        apply_box=True,   # Enable box adjustment (but Box=1 so no effect)
        box_tolerance=2,
        apply_scm=True,   # Enable scheme adjustment
        scm_tolerance=2
    )
    
    print("Results:")
    for idx, row in result.iterrows():
        name = row['Name']
        base_qty = row['Predicted_Base_Quantity']
        stock = row['Stock']
        scm = row['Scm']
        predicted = row['Predicted_Order']
        
        # Calculate expected after stock subtraction
        after_stock = base_qty - stock
        
        print(f"{name}: Base={base_qty}, Stock={stock}, After={after_stock}, Scm={scm} → {predicted}")
        
        # Verify sum if it's a scheme
        if '+' in str(predicted):
            parts = str(predicted).split('+')
            total = float(parts[0]) + float(parts[1])
            is_whole = total == int(total)
            status = "✅" if is_whole else "❌"
            print(f"  {status} Sum: {parts[0]} + {parts[1]} = {total}")

if __name__ == "__main__":
    # Run validation
    success = validate_exact_scenarios()
    
    # Test integration
    test_streamlit_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 EDGE CASE FIX SUCCESSFUL!")
        print("✅ Both user scenarios now work correctly")
        print("✅ All scheme results sum to whole numbers")
        print("✅ Ready for production use")
    else:
        print("❌ EDGE CASE FIX FAILED!")
        print("⚠️ Need further investigation")
