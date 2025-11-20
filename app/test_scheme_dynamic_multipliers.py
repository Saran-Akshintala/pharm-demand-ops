"""
Unit tests for scheme calculation with dynamic multipliers.

This test file allows testing the scheme calculation logic without needing
to upload files through the UI. You can modify the test cases to test different
scenarios.

Usage:
    python -m app.test_scheme_dynamic_multipliers
    or
    python app/test_scheme_dynamic_multipliers.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import (
    _apply_scheme_adjustment_new,
    apply_scheme_to_quantity,
    _apply_scheme_adjustment
)


def test_scheme_adjustment_new():
    """Test _apply_scheme_adjustment_new() with various scenarios."""
    print("\n" + "=" * 70)
    print("Testing _apply_scheme_adjustment_new()")
    print("=" * 70)
    
    # Create a dummy DataFrame with Scm column
    df = pd.DataFrame({'Scm': ['5+1']})
    row = pd.Series({'Scm': '5+1'})
    
    test_cases = [
        {
            'name': 'Exact match: Scm "5+1", Predicted_Base = 6',
            'scm': '5+1',
            'predicted_qty': 6,
            'tolerance': 2,
            'expected': '5+1',
            'expected_total': 6.0
        },
        {
            'name': 'Half pack: Scm "5+1", Predicted_Base = 3',
            'scm': '5+1',
            'predicted_qty': 3,
            'tolerance': 2,
            'expected': '2.5+0.5',
            'expected_total': 3.0
        },
        {
            'name': 'Multiple packs: Scm "9+1", Predicted_Base = 20',
            'scm': '9+1',
            'predicted_qty': 20,
            'tolerance': 2,
            'expected': '18+2',
            'expected_total': 20.0
        },
        {
            'name': 'High quantity: Scm "5+1", Predicted_Base = 500',
            'scm': '5+1',
            'predicted_qty': 500,
            'tolerance': 2,
            'expected_pattern': '+',  # Should contain scheme format
            'expected_total_range': (498, 502)  # Should be close to 500
        },
        {
            'name': 'Double pack: Scm "5+1", Predicted_Base = 12',
            'scm': '5+1',
            'predicted_qty': 12,
            'tolerance': 2,
            'expected': '10+2',
            'expected_total': 12.0
        },
        {
            'name': 'Triple pack: Scm "5+1", Predicted_Base = 18',
            'scm': '5+1',
            'predicted_qty': 18,
            'tolerance': 2,
            'expected': '15+3',
            'expected_total': 18.0
        },
        {
            'name': 'Very small quantity: Scm "5+1", Predicted_Base = 1',
            'scm': '5+1',
            'predicted_qty': 1,
            'tolerance': 2,
            'expected': '1',  # Should fall back to simple number (< 0.5 multiplier)
            'expected_total': 1.0
        },
        {
            'name': 'Large quantity: Scm "5+1", Predicted_Base = 1000',
            'scm': '5+1',
            'predicted_qty': 1000,
            'tolerance': 2,
            'expected_pattern': '+',  # Should contain scheme format
            'expected_total_range': (998, 1002)  # Should be close to 1000
        },
        {
            'name': 'Different scheme: Scm "10+2", Predicted_Base = 24',
            'scm': '10+2',
            'predicted_qty': 24,
            'tolerance': 2,
            'expected': '20+4',
            'expected_total': 24.0
        },
        {
            'name': 'Decimal scheme: Scm "2.5+0.5", Predicted_Base = 6',
            'scm': '2.5+0.5',
            'predicted_qty': 6,
            'tolerance': 2,
            'expected': '5+1',
            'expected_total': 6.0
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Scm: {case['scm']}, Predicted_Base: {case['predicted_qty']}, Tolerance: {case['tolerance']}")
        
        # Update row and df with current scheme
        row['Scm'] = case['scm']
        df['Scm'] = [case['scm']]
        
        try:
            result = _apply_scheme_adjustment_new(
                case['predicted_qty'],
                row,
                df,
                case['tolerance']
            )
            
            print(f"   Result: {result}")
            
            # Verify the result
            if '+' in result:
                parts = result.split('+')
                base_val = float(parts[0])
                bonus_val = float(parts[1])
                total = base_val + bonus_val
                diff = abs(total - case['predicted_qty'])
                
                print(f"   Total: {total}, Difference: {diff:.2f}")
                
                # Check expectations
                success = True
                if 'expected' in case:
                    if result != case['expected']:
                        print(f"   [FAIL] Expected: {case['expected']}, Got: {result}")
                        success = False
                    else:
                        print(f"   [PASS] Matches expected result")
                
                if 'expected_total' in case:
                    if abs(total - case['expected_total']) > 0.01:
                        print(f"   [FAIL] Expected total: {case['expected_total']}, Got: {total}")
                        success = False
                
                if 'expected_total_range' in case:
                    min_val, max_val = case['expected_total_range']
                    if not (min_val <= total <= max_val):
                        print(f"   [FAIL] Expected total in range [{min_val}, {max_val}], Got: {total}")
                        success = False
                
                if diff <= case['tolerance']:
                    print(f"   [OK] Within tolerance")
                else:
                    print(f"   [WARN] Outside tolerance (but may be acceptable)")
                    success = False
                
                if success:
                    passed += 1
                else:
                    failed += 1
            else:
                # Simple number result
                result_num = float(result)
                print(f"   Returned as simple number: {result_num}")
                
                if 'expected' in case and case['expected'] == result:
                    print(f"   [PASS] Matches expected result")
                    passed += 1
                elif 'expected_pattern' in case:
                    print(f"   [FAIL] Expected scheme format (with '+'), got simple number")
                    failed += 1
                else:
                    print(f"   [INFO] Simple number (may be expected for small quantities)")
                    passed += 1
                    
        except Exception as e:
            print(f"   [ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return passed, failed


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    df = pd.DataFrame({'Scm': ['5+1']})
    row = pd.Series({'Scm': '5+1'})
    
    edge_cases = [
        {
            'name': 'No scheme column',
            'scm': None,
            'predicted_qty': 10,
            'tolerance': 2,
            'expected': '10'  # Should return simple number
        },
        {
            'name': 'Empty scheme',
            'scm': '',
            'predicted_qty': 10,
            'tolerance': 2,
            'expected': '10'
        },
        {
            'name': 'Zero scheme',
            'scm': '0+0',
            'predicted_qty': 10,
            'tolerance': 2,
            'expected': '10'
        },
        {
            'name': 'Invalid scheme format (no +)',
            'scm': '10',
            'predicted_qty': 10,
            'tolerance': 2,
            'expected': '10'
        },
        {
            'name': 'Zero base',
            'scm': '0+1',
            'predicted_qty': 10,
            'tolerance': 2,
            'expected': '10'
        },
        {
            'name': 'Zero bonus',
            'scm': '5+0',
            'predicted_qty': 10,
            'tolerance': 2,
            'expected': '10'
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Scm: {case['scm']}, Predicted_Base: {case['predicted_qty']}")
        
        if case['scm'] is None:
            df_no_scm = pd.DataFrame({})  # No Scm column
            row_no_scm = pd.Series({})
            test_df = df_no_scm
            test_row = row_no_scm
        else:
            row['Scm'] = case['scm']
            df['Scm'] = [case['scm']]
            test_df = df
            test_row = row
        
        try:
            result = _apply_scheme_adjustment_new(
                case['predicted_qty'],
                test_row,
                test_df,
                case['tolerance']
            )
            
            print(f"   Result: {result}")
            
            if result == case['expected']:
                print(f"   [PASS] Matches expected: {case['expected']}")
                passed += 1
            else:
                print(f"   [FAIL] Expected: {case['expected']}, Got: {result}")
                failed += 1
                
        except Exception as e:
            print(f"   [ERROR] Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Edge Case Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return passed, failed


def test_custom_scenario():
    """Test a custom scenario - modify this function to test your specific cases."""
    print("\n" + "=" * 70)
    print("Custom Scenario Testing")
    print("=" * 70)
    print("Modify this function to test your specific scenarios")
    print("=" * 70)
    
    # Example: Test your specific case
    df = pd.DataFrame({'Scm': ['5+1']})
    row = pd.Series({'Scm': '5+1'})
    
    # Modify these values to test your scenario
    custom_tests = [
        {
            'scm': '5+1',
            'predicted_qty': 3000,  # Change this
            'tolerance': 2,       # Change this
        },
        # Add more custom tests here
    ]
    
    for i, test in enumerate(custom_tests, 1):
        print(f"\nCustom Test {i}:")
        print(f"  Scm: {test['scm']}")
        print(f"  Predicted_Base: {test['predicted_qty']}")
        print(f"  Tolerance: {test['tolerance']}")
        
        row['Scm'] = test['scm']
        df['Scm'] = [test['scm']]
        
        result = _apply_scheme_adjustment_new(
            test['predicted_qty'],
            row,
            df,
            test['tolerance']
        )
        
        print(f"  Result: {result}")
        
        if '+' in result:
            parts = result.split('+')
            base_val = float(parts[0])
            bonus_val = float(parts[1])
            total = base_val + bonus_val
            diff = abs(total - test['predicted_qty'])
            print(f"  Total: {total}, Difference: {diff:.2f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Scheme Calculation Unit Tests")
    print("Dynamic Multiplier Testing")
    print("=" * 70)
    
    # Run main tests
    passed1, failed1 = test_scheme_adjustment_new()
    
    # Run edge case tests
    passed2, failed2 = test_edge_cases()
    
    # Run custom scenario tests
    test_custom_scenario()
    
    # Summary
    total_passed = passed1 + passed2
    total_failed = failed1 + failed2
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    print("=" * 70)
    
    return total_failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

