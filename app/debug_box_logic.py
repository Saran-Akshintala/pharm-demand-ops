"""
Debug the box adjustment logic with the specific example provided.
"""

import pandas as pd
import sys
sys.path.append('.')

try:
    from utils import compute_predicted_order_with_adjustments, _apply_box_adjustment_new
    
    print('=== Debugging Box Adjustment Logic ===')
    print('Example: Box=10, Predicted_Base_Quantity=8, Stock=7, Scm=0+0')
    print('Expected: 8-7=1, |1-10|=9 > 2, so should stay 1')
    print()
    
    # Test the individual function first
    row = pd.Series({'Box': 10})
    result = _apply_box_adjustment_new(1, row, tolerance=2)
    print(f'_apply_box_adjustment_new(1, Box=10, tolerance=2) = {result}')
    print(f'Difference: |1-10| = {abs(1-10)} > 2, so should return 1')
    print()
    
    # Test the full pipeline
    test_data = pd.DataFrame({
        'Name': ['Debug Product'],
        'Predicted_Base_Quantity': [8],
        'Stock': [7],
        'Box': [10],
        'Scm': ['0+0']
    })
    
    print('Input data:')
    print(test_data)
    print()
    
    # Test with our priority logic
    result = compute_predicted_order_with_adjustments(
        test_data,
        apply_box=True,
        box_tolerance=2,
        apply_scm=True,
        scm_tolerance=2
    )
    
    print('After priority logic:')
    print(result[['Name', 'Predicted_Base_Quantity', 'Stock', 'Box', 'Scm', 'Predicted_Order']])
    print()
    
    # Test with business rules applied
    from utils import apply_predicted_order_business_rules
    result_with_rules, styling_info = apply_predicted_order_business_rules(result)
    
    print('After business rules:')
    print(result_with_rules[['Name', 'Predicted_Base_Quantity', 'Stock', 'Box', 'Scm', 'Predicted_Order']])
    print()
    
    if styling_info:
        print('Styling info:')
        for idx, info in styling_info.items():
            print(f'  Row {idx}: {info}')
    
    result = result_with_rules  # Use the final result
    
    print('Result:')
    print(result[['Name', 'Predicted_Base_Quantity', 'Stock', 'Box', 'Scm', 'Predicted_Order']])
    print()
    
    predicted_order = result['Predicted_Order'].iloc[0]
    print(f'Final Predicted_Order: {predicted_order}')
    
    if predicted_order == '1':
        print('✅ CORRECT: Box adjustment logic is working properly')
    else:
        print(f'❌ INCORRECT: Expected "1", got "{predicted_order}"')
        print('Debugging step by step:')
        
        # Step by step debugging
        base_qty = test_data['Predicted_Base_Quantity'].iloc[0]
        stock_qty = test_data['Stock'].iloc[0]
        box_qty = test_data['Box'].iloc[0]
        scm_value = test_data['Scm'].iloc[0]
        
        print(f'Step 1: {base_qty} - {stock_qty} = {base_qty - stock_qty}')
        
        after_stock = base_qty - stock_qty
        if after_stock <= 0:
            print(f'Step 1 result: {after_stock} <= 0, should return "0"')
        else:
            print(f'Step 2: Box adjustment - |{after_stock} - {box_qty}| = {abs(after_stock - box_qty)}')
            if abs(after_stock - box_qty) <= 2:
                print(f'Step 2 result: Difference <= 2, adjust to {box_qty}')
            else:
                print(f'Step 2 result: Difference > 2, keep {after_stock}')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
