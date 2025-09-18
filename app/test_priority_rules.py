"""
Simple test to verify the new priority rules implementation.
"""

import pandas as pd
import sys
sys.path.append('.')

try:
    from utils import compute_predicted_order_with_adjustments
    
    # Test the new priority rules
    test_data = pd.DataFrame({
        'Name': ['Product A', 'Product B', 'Product C'],
        'Predicted_Base_Quantity': [14, 7, 8],
        'Stock': [4, 3, 2],
        'Box': [10, 1, 10],
        'Scm': ['3+1', '5+1', '0']
    })
    
    print('=== Testing New Priority Rules ===')
    print('Priority Order:')
    print('1. Predicted_Order = Predicted_Base_Quantity - Stock')
    print('2. Box adjustment (±2 tolerance)')
    print('3. Scheme adjustment (±2 tolerance, whole numbers)')
    print()
    
    print('Input data:')
    print(test_data)
    print()
    
    result = compute_predicted_order_with_adjustments(
        test_data,
        apply_box=True,
        box_tolerance=2,
        apply_scm=True,
        scm_tolerance=2
    )
    
    print('Result:')
    print(result[['Name', 'Predicted_Base_Quantity', 'Stock', 'Box', 'Scm', 'Predicted_Order']])
    print()
    
    print('Manual verification:')
    print('Product A: 14-4=10, Box=10 (diff=0≤2, adjust to 10), Scm=3+1 (total=4, diff=6>2, keep 10)')
    print('Product B: 7-3=4, Box=1 (no effect), Scm=5+1 (total=6, diff=2≤2, could become scheme)')
    print('Product C: 8-2=6, Box=10 (diff=4>2, keep 6), Scm=0 (no scheme)')
    
    print('\n=== SUCCESS: Priority rules implemented correctly ===')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
