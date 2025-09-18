"""
Comprehensive test for the edge case fix in both scheme adjustment functions.
"""

import sys
sys.path.append('.')

import pandas as pd
from utils import compute_predicted_order_with_adjustments, _apply_scheme_adjustment_new

def test_user_scenarios():
    """Test the exact scenarios mentioned by the user."""
    print("🧪 Testing User-Reported Scenarios...")
    
    # Scenario 1: Predicted_Base_Quantity=4, Stock=2, Scm=4+1
    # Expected: 2.5+0.5 (sum=3)
    test_data_1 = pd.DataFrame({
        'Name': ['Product 1'],
        'Predicted_Base_Quantity': [4],
        'Stock': [2],
        'Scm': ['4+1'],
        'Box': [1],  # Avoid box adjustment
        'Days': [30]  # Avoid Days constraint
    })
    
    result_1 = compute_predicted_order_with_adjustments(
        test_data_1,
        apply_box=False,
        box_tolerance=2,
        apply_scm=True,
        scm_tolerance=2
    )
    
    predicted_order_1 = result_1.iloc[0]['Predicted_Order']
    print(f"Scenario 1: Predicted_Base_Quantity=4, Stock=2, Scm=4+1 → {predicted_order_1}")
    
    # Verify sum
    if '+' in predicted_order_1:
        parts = predicted_order_1.split('+')
        total = float(parts[0]) + float(parts[1])
        print(f"  Sum: {parts[0]} + {parts[1]} = {total} (whole: {total == int(total)})")
    
    # Scenario 2: Predicted_Base_Quantity=7, Stock=1, Scm=10+1
    # Expected: 5.5+0.5 (sum=6)
    test_data_2 = pd.DataFrame({
        'Name': ['Product 2'],
        'Predicted_Base_Quantity': [7],
        'Stock': [1],
        'Scm': ['10+1'],
        'Box': [1],  # Avoid box adjustment
        'Days': [30]  # Avoid Days constraint
    })
    
    result_2 = compute_predicted_order_with_adjustments(
        test_data_2,
        apply_box=False,
        box_tolerance=2,
        apply_scm=True,
        scm_tolerance=2
    )
    
    predicted_order_2 = result_2.iloc[0]['Predicted_Order']
    print(f"Scenario 2: Predicted_Base_Quantity=7, Stock=1, Scm=10+1 → {predicted_order_2}")
    
    # Verify sum
    if '+' in predicted_order_2:
        parts = predicted_order_2.split('+')
        total = float(parts[0]) + float(parts[1])
        print(f"  Sum: {parts[0]} + {parts[1]} = {total} (whole: {total == int(total)})")
    
    return predicted_order_1, predicted_order_2

def test_additional_edge_cases():
    """Test additional edge cases to ensure robustness."""
    print("\n🧪 Testing Additional Edge Cases...")
    
    edge_cases = [
        # (Predicted_Base_Quantity, Stock, Scm, Expected_Pattern)
        (6, 2, '8+1', 'Should be 4.5+0.5'),  # 6-2=4, 8+1 with 0.5x = 4+0.5 → 4.5+0.5
        (8, 3, '6+1', 'Should be 3.5+0.5'),  # 8-3=5, 6+1 with 0.5x = 3+0.5 → 3.5+0.5
        (12, 6, '12+1', 'Should be 6.5+0.5'), # 12-6=6, 12+1 with 0.5x = 6+0.5 → 6.5+0.5
    ]
    
    for base_qty, stock, scm, description in edge_cases:
        test_data = pd.DataFrame({
            'Name': ['Test Product'],
            'Predicted_Base_Quantity': [base_qty],
            'Stock': [stock],
            'Scm': [scm],
            'Box': [1],
            'Days': [30]
        })
        
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=False,
            box_tolerance=2,
            apply_scm=True,
            scm_tolerance=2
        )
        
        predicted_order = result.iloc[0]['Predicted_Order']
        print(f"Base={base_qty}, Stock={stock}, Scm={scm} → {predicted_order}")
        
        # Verify sum
        if '+' in predicted_order:
            parts = predicted_order.split('+')
            total = float(parts[0]) + float(parts[1])
            is_whole = total == int(total)
            status = "✅" if is_whole else "❌"
            print(f"  {status} Sum: {parts[0]} + {parts[1]} = {total} - {description}")

def test_normal_cases():
    """Ensure normal cases still work correctly."""
    print("\n🧪 Testing Normal Cases (Should Not Be Affected)...")
    
    normal_cases = [
        # (Predicted_Base_Quantity, Stock, Scm, Expected)
        (6, 0, '5+1', '5+1'),    # Exact match
        (8, 0, '3+1', '6+2'),    # 2x multiplier
        (5, 0, '10+1', '5'),     # Too small for scheme
    ]
    
    for base_qty, stock, scm, expected in normal_cases:
        test_data = pd.DataFrame({
            'Name': ['Normal Test'],
            'Predicted_Base_Quantity': [base_qty],
            'Stock': [stock],
            'Scm': [scm],
            'Box': [1],
            'Days': [30]
        })
        
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=False,
            box_tolerance=2,
            apply_scm=True,
            scm_tolerance=2
        )
        
        predicted_order = result.iloc[0]['Predicted_Order']
        status = "✅" if predicted_order == expected else f"❌ (expected {expected})"
        print(f"{status} Base={base_qty}, Stock={stock}, Scm={scm} → {predicted_order}")

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE EDGE CASE TEST")
    print("=" * 60)
    
    # Test user scenarios
    result1, result2 = test_user_scenarios()
    
    # Test additional edge cases
    test_additional_edge_cases()
    
    # Test normal cases
    test_normal_cases()
    
    print("\n" + "=" * 60)
    print("FINAL VALIDATION:")
    
    # Check user scenarios
    scenario1_fixed = result1 == "2.5+0.5"
    scenario2_fixed = result2 == "5.5+0.5"
    
    if scenario1_fixed and scenario2_fixed:
        print("🎉 ALL USER SCENARIOS FIXED!")
        print("✅ Scenario 1: 2.5+0.5 (sum=3.0)")
        print("✅ Scenario 2: 5.5+0.5 (sum=6.0)")
        print("✅ Edge case fix working correctly")
    else:
        print("❌ SOME SCENARIOS STILL BROKEN:")
        if not scenario1_fixed:
            print(f"❌ Scenario 1: Expected 2.5+0.5, got {result1}")
        if not scenario2_fixed:
            print(f"❌ Scenario 2: Expected 5.5+0.5, got {result2}")
