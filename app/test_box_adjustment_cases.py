"""
Test various box adjustment cases to ensure the logic is working correctly.
"""

import pandas as pd
import sys
sys.path.append('.')

try:
    from utils import compute_predicted_order_with_adjustments, apply_predicted_order_business_rules
    
    print('=== Testing Box Adjustment Cases ===')
    print()
    
    # Test cases
    test_cases = [
        {
            'name': 'Case 1: Should NOT adjust (difference > tolerance)',
            'data': {'Base': 8, 'Stock': 7, 'Box': 10, 'Scm': '0+0'},
            'expected': '1',  # 8-7=1, |1-10|=9 > 2, keep 1
            'description': '8-7=1, |1-10|=9 > 2, should keep 1'
        },
        {
            'name': 'Case 2: Should adjust (difference = tolerance)',
            'data': {'Base': 10, 'Stock': 2, 'Box': 10, 'Scm': '0+0'},
            'expected': '10',  # 10-2=8, |8-10|=2 = 2, adjust to 10
            'description': '10-2=8, |8-10|=2 = tolerance, should adjust to 10'
        },
        {
            'name': 'Case 3: Should adjust (difference < tolerance)',
            'data': {'Base': 11, 'Stock': 2, 'Box': 10, 'Scm': '0+0'},
            'expected': '10',  # 11-2=9, |9-10|=1 < 2, adjust to 10
            'description': '11-2=9, |9-10|=1 < tolerance, should adjust to 10'
        },
        {
            'name': 'Case 4: Should NOT adjust (difference > tolerance)',
            'data': {'Base': 15, 'Stock': 2, 'Box': 10, 'Scm': '0+0'},
            'expected': '13',  # 15-2=13, |13-10|=3 > 2, keep 13
            'description': '15-2=13, |13-10|=3 > tolerance, should keep 13'
        },
        {
            'name': 'Case 5: Exact match',
            'data': {'Base': 12, 'Stock': 2, 'Box': 10, 'Scm': '0+0'},
            'expected': '10',  # 12-2=10, |10-10|=0 < 2, adjust to 10 (already exact)
            'description': '12-2=10, |10-10|=0 = exact match, should stay 10'
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"{case['name']}:")
        print(f"  {case['description']}")
        
        test_data = pd.DataFrame({
            'Name': [f'Product {i}'],
            'Predicted_Base_Quantity': [case['data']['Base']],
            'Stock': [case['data']['Stock']],
            'Box': [case['data']['Box']],
            'Scm': [case['data']['Scm']]
        })
        
        # Apply our priority logic
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=True,
            box_tolerance=2,
            apply_scm=True,
            scm_tolerance=2
        )
        
        # Apply business rules
        result_final, styling_info = apply_predicted_order_business_rules(result)
        
        predicted_order = result_final['Predicted_Order'].iloc[0]
        expected = case['expected']
        
        if predicted_order == expected:
            print(f"  ✅ PASS: Got '{predicted_order}' (expected '{expected}')")
        else:
            print(f"  ❌ FAIL: Got '{predicted_order}' (expected '{expected}')")
            all_passed = False
        
        print()
    
    if all_passed:
        print('🎉 ALL TESTS PASSED! Box adjustment logic is working correctly.')
    else:
        print('❌ Some tests failed. Box adjustment logic needs review.')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
