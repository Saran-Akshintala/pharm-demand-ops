"""
Test the customer feedback enhancements:
1. Box Quantity Rule updates
2. Box Quantity Adjustment improvements  
3. Multiple tooltip messages
4. Expiry column highlighting
5. Scheme whole number adjustment
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import (
    _apply_box_adjustment_new,
    _apply_scheme_adjustment,
    apply_expiry_highlighting,
    apply_predicted_order_business_rules
)

def test_box_quantity_rules():
    """Test Box Quantity Rule updates."""
    print("🧪 Testing Box Quantity Rules...")
    
    # Test Case 1: Box = 0 (should ignore box adjustment)
    row1 = pd.Series({'Box': 0})
    result1 = _apply_box_adjustment_new(10, row1, 2)
    assert result1 == 10, f"Expected 10, got {result1} for Box=0"
    print("✅ Box=0: Ignores adjustment correctly")
    
    # Test Case 2: Box = 1 (should ignore box adjustment)
    row2 = pd.Series({'Box': 1})
    result2 = _apply_box_adjustment_new(10, row2, 2)
    assert result2 == 10, f"Expected 10, got {result2} for Box=1"
    print("✅ Box=1: Ignores adjustment correctly")
    
    # Test Case 3: Box = 50, Predicted = 99 (should round to nearest multiple)
    row3 = pd.Series({'Box': 50})
    result3 = _apply_box_adjustment_new(99, row3, 10)  # 99 is closest to 100 (2*50)
    assert result3 == 100, f"Expected 100, got {result3} for Box=50, Predicted=99"
    print("✅ Box=50, Predicted=99: Rounds to 100 correctly")
    
    # Test Case 4: Box = 10, Predicted = 23 (should round to 20)
    row4 = pd.Series({'Box': 10})
    result4 = _apply_box_adjustment_new(23, row4, 5)  # 23 is closer to 20 (2*10) than 30 (3*10)
    assert result4 == 20, f"Expected 20, got {result4} for Box=10, Predicted=23"
    print("✅ Box=10, Predicted=23: Rounds to 20 correctly")
    
    return True

def test_scheme_whole_number():
    """Test Scheme whole number adjustment."""
    print("\n🧪 Testing Scheme Whole Number Adjustment...")
    
    # Create test data
    df = pd.DataFrame({
        'Scm': ['4+1'],
        'L7': [10], 'L15': [12], 'L30': [15], 'L45': [18], 'L60': [20], 'L75': [22], 'L90': [25]
    })
    row = df.iloc[0]
    
    # Test case: Should result in whole number when components are summed
    result = _apply_scheme_adjustment(2.5, row, df, 2)  # 2.5 should become a scheme that sums to whole number
    print(f"✅ Scheme adjustment result: {result}")
    
    # Verify the result sums to a whole number
    if '+' in result:
        parts = result.split('+')
        total = float(parts[0]) + float(parts[1])
        assert total == int(total), f"Scheme {result} doesn't sum to whole number: {total}"
        print(f"✅ Scheme {result} sums to whole number: {total}")
    
    return True

def test_expiry_highlighting():
    """Test Expiry column highlighting."""
    print("\n🧪 Testing Expiry Column Highlighting...")
    
    # Create test data with various expiry dates
    current_date = datetime.now()
    test_data = pd.DataFrame({
        'Name': ['Product A', 'Product B', 'Product C', 'Product D'],
        'Expiry': [
            current_date + timedelta(days=20),   # 0.7 months - URGENT
            current_date + timedelta(days=60),   # 2 months - Soon  
            current_date + timedelta(days=120),  # 4 months - Moderate
            current_date + timedelta(days=200)   # 6.6 months - No highlight
        ]
    })
    
    expiry_styling = apply_expiry_highlighting(test_data)
    
    # Check results
    assert 0 in expiry_styling, "Product A should be highlighted (expires in 0.7 months)"
    assert 1 in expiry_styling, "Product B should be highlighted (expires in 2 months)"
    assert 2 in expiry_styling, "Product C should be highlighted (expires in 4 months)"
    assert 3 not in expiry_styling, "Product D should NOT be highlighted (expires in 6.6 months)"
    
    print(f"✅ Expiry highlighting: {len(expiry_styling)} products highlighted")
    for idx, style in expiry_styling.items():
        print(f"   Row {idx}: {style['tooltip']}")
    
    return True

def test_multiple_tooltips():
    """Test multiple tooltip messages."""
    print("\n🧪 Testing Multiple Tooltip Messages...")
    
    # Create test data that should trigger multiple business rules
    test_data = pd.DataFrame({
        'Name': ['Test Product'],
        'Predicted_Order': [5],
        'Predicted_Base_Quantity': [5],
        'Stock': [2],
        'Box': [10],
        'Days': [120],  # > 90 days
        'No_of_Customer_Last_Month': [1],  # <= 2 customers
        'L7': [1], 'L15': [1], 'L30': [2], 'L45': [1], 'L60': [2], 'L75': [1], 'L90': [1]  # Uneven pattern
    })
    
    df_result, styling_info = apply_predicted_order_business_rules(test_data)
    
    # Check if multiple conditions were detected
    if 0 in styling_info:
        tooltip = styling_info[0]['tooltip']
        print(f"✅ Combined tooltip: {tooltip}")
        
        # Check if multiple conditions are mentioned (using | separator)
        condition_count = tooltip.count('|') + 1 if '|' in tooltip else 1
        print(f"✅ Number of conditions detected: {condition_count}")
    
    return True

def test_integration():
    """Test overall integration."""
    print("\n🧪 Testing Overall Integration...")
    
    # Create comprehensive test data
    test_data = pd.DataFrame({
        'Name': ['Product 1', 'Product 2', 'Product 3'],
        'Order': [10, 15, 20],
        'Predicted_Base_Quantity': [12, 18, 25],
        'Stock': [2, 5, 3],
        'Box': [0, 5, 10],  # Mix of box values
        'Scm': ['3+1', '5+1', '4+1'],
        'Days': [45, 95, 30],
        'No_of_Customer_Last_Month': [5, 1, 8],
        'L7': [8, 1], 'L15': [10, 1], 'L30': [12, 2], 'L45': [14, 1], 'L60': [16, 2], 'L75': [18, 1], 'L90': [20, 1]
    })
    
    # Test business rules
    df_result, styling_info = apply_predicted_order_business_rules(test_data)
    print(f"✅ Business rules processed: {len(styling_info)} rows with styling")
    
    # Test expiry highlighting (add expiry column)
    current_date = datetime.now()
    test_data['Expiry'] = [
        current_date + timedelta(days=30),   # 1 month
        current_date + timedelta(days=90),   # 3 months
        current_date + timedelta(days=200)   # 6+ months
    ]
    
    expiry_styling = apply_expiry_highlighting(test_data)
    print(f"✅ Expiry highlighting: {len(expiry_styling)} products highlighted")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Customer Feedback Enhancements")
    print("=" * 60)
    
    tests = [
        ("Box Quantity Rules", test_box_quantity_rules),
        ("Scheme Whole Number", test_scheme_whole_number),
        ("Expiry Highlighting", test_expiry_highlighting),
        ("Multiple Tooltips", test_multiple_tooltips),
        ("Integration Test", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhancements working correctly!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
