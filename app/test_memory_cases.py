"""
Test specific cases from memories to ensure the scheme logic works correctly.
"""

import pandas as pd
import sys
sys.path.append('.')

try:
    from utils import compute_predicted_order_with_adjustments
    
    print('=== Testing Memory Cases ===')
    print()
    
    # Memory cases from MEMORY[5787a517-7d56-486c-b7f4-2dec9064ab80]
    memory_cases = [
        {
            'name': 'Memory Case 1: Base=7, Stock=1, Scm=11+1',
            'base': 7, 'stock': 1, 'scm': '11+1',
            'expected_after_stock': 6,  # 7-1=6
            'expected_scheme': '5.5+0.5',  # 0.5x of 11+1, total=6
            'description': 'After 7-1=6, should use 0.5x multiplier: 5.5+0.5=6'
        },
        {
            'name': 'Memory Case 2: Base=6, Stock=2, Scm=5+1',
            'base': 6, 'stock': 2, 'scm': '5+1',
            'expected_after_stock': 4,  # 6-2=4
            'expected_scheme': '2.5+0.5',  # 0.5x of 5+1, but total=3, not 4. Should be simple number
            'description': 'After 6-2=4, 0.5x gives 2.5+0.5=3, diff=1≤2, but might prefer simple 4'
        },
        {
            'name': 'Memory Case 3: Base=8, Stock=7, Scm=9+1',
            'base': 8, 'stock': 7, 'scm': '9+1',
            'expected_after_stock': 1,  # 8-7=1
            'expected_scheme': '1',  # Too small for any scheme
            'description': 'After 8-7=1, too small for scheme (0.5x would be 4.5+0.5=5, diff=4>2)'
        },
        {
            'name': 'Memory Case 4: Base=10, Stock=5, Scm=10+1',
            'base': 10, 'stock': 5, 'scm': '10+1',
            'expected_after_stock': 5,  # 10-5=5
            'expected_scheme': '5+0.5',  # Need to check if this is valid for 10+1 scheme
            'description': 'After 10-5=5, should find best valid multiplier'
        }
    ]
    
    for case in memory_cases:
        print(f"{case['name']}:")
        print(f"  {case['description']}")
        
        test_data = pd.DataFrame({
            'Name': ['Test Product'],
            'Predicted_Base_Quantity': [case['base']],
            'Stock': [case['stock']],
            'Box': [1],  # No box adjustment
            'Scm': [case['scm']]
        })
        
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=False,  # Disable box adjustment
            apply_scm=True,
            scm_tolerance=2
        )
        
        predicted_order = result['Predicted_Order'].iloc[0]
        
        print(f"  Input: Base={case['base']}, Stock={case['stock']}, Scm={case['scm']}")
        print(f"  After stock: {case['base']}-{case['stock']}={case['expected_after_stock']}")
        print(f"  Result: {predicted_order}")
        
        # Analyze the result
        if '+' in predicted_order:
            try:
                parts = predicted_order.split('+')
                result_base = float(parts[0])
                result_bonus = float(parts[1])
                result_total = result_base + result_bonus
                
                # Check what multiplier was used
                original_parts = case['scm'].split('+')
                original_base = float(original_parts[0])
                original_bonus = float(original_parts[1])
                
                multiplier = result_base / original_base if original_base != 0 else 0
                
                print(f"  Analysis: {result_base}+{result_bonus}={result_total} ({multiplier:.1f}x multiplier)")
                
                # Check if it's a valid multiplier
                valid_multipliers = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
                is_valid = any(abs(multiplier - vm) < 0.01 for vm in valid_multipliers)
                
                if is_valid:
                    print(f"  ✅ Valid multiplier used")
                else:
                    print(f"  ❌ Invalid multiplier: {multiplier:.2f}x")
                    
            except Exception as e:
                print(f"  ❌ Error parsing result: {e}")
        else:
            print(f"  Analysis: Simple number (no scheme applied)")
            try:
                numeric_result = float(predicted_order)
                if numeric_result == case['expected_after_stock']:
                    print(f"  ✅ Correct simple number")
                else:
                    print(f"  ⚠️  Expected {case['expected_after_stock']}, got {numeric_result}")
            except:
                print(f"  ❌ Invalid result format")
        
        print()
    
    print('=== Additional Test: Scheme vs Simple Number Decision ===')
    
    # Test when scheme should vs shouldn't be applied
    decision_cases = [
        {'qty': 1, 'scm': '9+1', 'description': 'Very small quantity, should stay simple'},
        {'qty': 5, 'scm': '9+1', 'description': '0.5x gives 4.5+0.5=5, exact match'},
        {'qty': 10, 'scm': '9+1', 'description': '1x gives 9+1=10, exact match'},
        {'qty': 7, 'scm': '9+1', 'description': 'Between 0.5x(5) and 1x(10), check tolerance'},
    ]
    
    for case in decision_cases:
        print(f"Qty={case['qty']}, Scm={case['scm']}: {case['description']}")
        
        test_data = pd.DataFrame({
            'Name': ['Decision Test'],
            'Predicted_Base_Quantity': [case['qty'] + 1],  # Add 1 for stock subtraction
            'Stock': [1],
            'Box': [1],
            'Scm': [case['scm']]
        })
        
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=False,
            apply_scm=True,
            scm_tolerance=2
        )
        
        print(f"  Result: {result['Predicted_Order'].iloc[0]}")
        print()
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
