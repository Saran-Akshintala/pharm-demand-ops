"""
Test the corrected scheme logic with valid multipliers only.
"""

import pandas as pd
import sys
sys.path.append('.')

try:
    from utils import compute_predicted_order_with_adjustments, _apply_scheme_adjustment_new
    
    print('=== Testing Corrected Scheme Logic ===')
    print('Valid multipliers: 0.5x, 1x, 2x, 3x, 4x, 5x')
    print()
    
    # Test cases based on user examples
    test_cases = [
        {
            'name': 'Case 1: Scm=3+1, should NOT give 7.5+2.5',
            'scm': '3+1',
            'predicted_qty': 10,  # After stock subtraction and box adjustment
            'tolerance': 2,
            'valid_options': ['1.5+0.5 (0.5x)', '3+1 (1x)', '6+2 (2x)', '9+3 (3x)', '12+4 (4x)', '15+5 (5x)'],
            'expected_closest': '9+3',  # Total=12, closest to 10 within tolerance
            'description': 'Should find closest valid multiplier, not arbitrary 7.5+2.5'
        },
        {
            'name': 'Case 2: Scm=9+1, should NOT give 2.5+0.5',
            'scm': '9+1',
            'predicted_qty': 5,  # After stock subtraction and box adjustment
            'tolerance': 2,
            'valid_options': ['4.5+0.5 (0.5x)', '9+1 (1x)', '18+2 (2x)', '27+3 (3x)', '36+4 (4x)', '45+5 (5x)'],
            'expected_closest': '4.5+0.5',  # Total=5, exact match
            'description': 'Should use 0.5x multiplier (4.5+0.5=5), not invalid 2.5+0.5'
        },
        {
            'name': 'Case 3: Scm=5+1, predicted_qty=6',
            'scm': '5+1',
            'predicted_qty': 6,
            'tolerance': 2,
            'valid_options': ['2.5+0.5 (0.5x)', '5+1 (1x)', '10+2 (2x)', '15+3 (3x)', '20+4 (4x)', '25+5 (5x)'],
            'expected_closest': '5+1',  # Total=6, exact match
            'description': 'Should use 1x multiplier (5+1=6), exact match'
        },
        {
            'name': 'Case 4: Outside tolerance, should return simple number',
            'scm': '3+1',
            'predicted_qty': 15,  # Far from any valid multiplier
            'tolerance': 2,
            'valid_options': ['1.5+0.5 (0.5x)', '3+1 (1x)', '6+2 (2x)', '9+3 (3x)', '12+4 (4x)', '15+5 (5x)'],
            'expected_closest': '15+5',  # Total=20, but diff=5 > tolerance=2, so return '15'
            'description': 'Should return simple number when no valid multiplier within tolerance'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"{case['name']}:")
        print(f"  {case['description']}")
        print(f"  Scm: {case['scm']}, Predicted_qty: {case['predicted_qty']}, Tolerance: {case['tolerance']}")
        print(f"  Valid options: {', '.join(case['valid_options'])}")
        
        # Create test data
        df = pd.DataFrame({'Scm': [case['scm']]})
        row = pd.Series({'Scm': case['scm']})
        
        # Test the scheme adjustment function
        result = _apply_scheme_adjustment_new(case['predicted_qty'], row, df, case['tolerance'])
        
        print(f"  Result: {result}")
        
        # Verify it's a valid multiplier or simple number
        if '+' in result:
            try:
                parts = result.split('+')
                result_base = float(parts[0])
                result_bonus = float(parts[1])
                result_total = result_base + result_bonus
                
                # Check if it's a valid multiplier
                original_base = float(case['scm'].split('+')[0])
                original_bonus = float(case['scm'].split('+')[1])
                
                # Calculate what multiplier was used
                multiplier = result_base / original_base if original_base != 0 else 0
                
                if abs(multiplier - 0.5) < 0.01:
                    mult_str = "0.5x"
                elif abs(multiplier - 1.0) < 0.01:
                    mult_str = "1x"
                elif abs(multiplier - 2.0) < 0.01:
                    mult_str = "2x"
                elif abs(multiplier - 3.0) < 0.01:
                    mult_str = "3x"
                elif abs(multiplier - 4.0) < 0.01:
                    mult_str = "4x"
                elif abs(multiplier - 5.0) < 0.01:
                    mult_str = "5x"
                else:
                    mult_str = f"INVALID {multiplier:.2f}x"
                
                print(f"  Analysis: {result_base}+{result_bonus}={result_total} ({mult_str})")
                
                if "INVALID" in mult_str:
                    print(f"  ❌ INVALID multiplier detected!")
                else:
                    print(f"  ✅ Valid multiplier used")
                    
            except:
                print(f"  ❌ Error parsing scheme result")
        else:
            print(f"  ✅ Simple number (no scheme applied)")
        
        print()
    
    print('=== Full Integration Test ===')
    
    # Test full pipeline
    test_data = pd.DataFrame({
        'Name': ['Product A', 'Product B'],
        'Predicted_Base_Quantity': [15, 12],  # Will become 10 and 5 after stock subtraction
        'Stock': [5, 7],
        'Box': [1, 1],  # No box adjustment
        'Scm': ['3+1', '9+1']
    })
    
    print('Input:')
    print(test_data)
    print()
    
    result = compute_predicted_order_with_adjustments(
        test_data,
        apply_box=False,  # Disable box for clarity
        apply_scm=True,
        scm_tolerance=2
    )
    
    print('Result:')
    print(result[['Name', 'Predicted_Base_Quantity', 'Stock', 'Scm', 'Predicted_Order']])
    print()
    
    print('Analysis:')
    print('Product A: 15-5=10, Scm=3+1, closest valid: 9+3 (3x multiplier, total=12, diff=2≤tolerance)')
    print('Product B: 12-7=5, Scm=9+1, closest valid: 4.5+0.5 (0.5x multiplier, total=5, exact match)')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
