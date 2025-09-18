"""
Debug the edge case scenarios to identify the root cause.
"""

import sys
sys.path.append('.')

import pandas as pd
from utils import _apply_scheme_adjustment, compute_predicted_order_with_adjustments

def debug_scenario_1():
    """Debug: Predicted_Base_Quantity=4, Stock=2, Scm=4+1"""
    print("🔍 Debugging Scenario 1:")
    print("Predicted_Base_Quantity=4, Stock=2, Scm=4+1")
    print("Expected: After stock subtraction (4-2=2), should be 2.5+0.5")
    print()
    
    # Create test data
    test_data = pd.DataFrame({
        'Name': ['Test Product 1'],
        'Predicted_Base_Quantity': [4],
        'Stock': [2],
        'Scm': ['4+1'],
        'Box': [1],  # Use 1 to avoid box adjustment
        'Days': [30]  # Less than 90 to avoid Days constraint
    })
    
    # Test the full pipeline
    result = compute_predicted_order_with_adjustments(
        test_data,
        apply_box=False,  # Disable box adjustment
        box_tolerance=2,
        apply_scm=True,   # Enable scheme adjustment
        scm_tolerance=2
    )
    
    predicted_order = result.iloc[0]['Predicted_Order']
    print(f"RESULT: {predicted_order}")
    
    # Test the scheme adjustment function directly
    after_stock = 4 - 2  # = 2
    print(f"After stock subtraction: {after_stock}")
    
    df_scheme = pd.DataFrame({'Scm': ['4+1']})
    row_scheme = df_scheme.iloc[0]
    scheme_result = _apply_scheme_adjustment(after_stock, row_scheme, df_scheme, 2)
    print(f"Direct scheme adjustment result: {scheme_result}")
    
    # Check if it sums to whole number
    if '+' in scheme_result:
        parts = scheme_result.split('+')
        total = float(parts[0]) + float(parts[1])
        print(f"Sum check: {parts[0]} + {parts[1]} = {total}")
        print(f"Is whole number: {total == int(total)}")
    
    return predicted_order

def debug_scenario_2():
    """Debug: Predicted_Base_Quantity=7, Stock=1, Scm=10+1"""
    print("\n🔍 Debugging Scenario 2:")
    print("Predicted_Base_Quantity=7, Stock=1, Scm=10+1")
    print("Expected: After stock subtraction (7-1=6), should be 5.5+0.5")
    print()
    
    # Create test data
    test_data = pd.DataFrame({
        'Name': ['Test Product 2'],
        'Predicted_Base_Quantity': [7],
        'Stock': [1],
        'Scm': ['10+1'],
        'Box': [1],  # Use 1 to avoid box adjustment
        'Days': [30]  # Less than 90 to avoid Days constraint
    })
    
    # Test the full pipeline
    result = compute_predicted_order_with_adjustments(
        test_data,
        apply_box=False,  # Disable box adjustment
        box_tolerance=2,
        apply_scm=True,   # Enable scheme adjustment
        scm_tolerance=2
    )
    
    predicted_order = result.iloc[0]['Predicted_Order']
    print(f"RESULT: {predicted_order}")
    
    # Test the scheme adjustment function directly
    after_stock = 7 - 1  # = 6
    print(f"After stock subtraction: {after_stock}")
    
    df_scheme = pd.DataFrame({'Scm': ['10+1']})
    row_scheme = df_scheme.iloc[0]
    scheme_result = _apply_scheme_adjustment(after_stock, row_scheme, df_scheme, 2)
    print(f"Direct scheme adjustment result: {scheme_result}")
    
    # Check if it sums to whole number
    if '+' in scheme_result:
        parts = scheme_result.split('+')
        total = float(parts[0]) + float(parts[1])
        print(f"Sum check: {parts[0]} + {parts[1]} = {total}")
        print(f"Is whole number: {total == int(total)}")
    
    return predicted_order

if __name__ == "__main__":
    print("🐛 ROOT CAUSE ANALYSIS - Edge Case Debug")
    print("=" * 60)
    
    result1 = debug_scenario_1()
    result2 = debug_scenario_2()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Scenario 1 result: {result1}")
    print(f"Scenario 2 result: {result2}")
    
    # Check if results are as expected
    if result1 == "2.5+0.5":
        print("✅ Scenario 1: FIXED")
    else:
        print("❌ Scenario 1: STILL BROKEN")
    
    if result2 == "5.5+0.5":
        print("✅ Scenario 2: FIXED")
    else:
        print("❌ Scenario 2: STILL BROKEN")
