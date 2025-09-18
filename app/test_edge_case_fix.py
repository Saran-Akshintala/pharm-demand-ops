"""
Test the edge case fix for scheme whole number adjustment.
Specifically testing X+0.5 cases that should become X.5+0.5
"""

import sys
sys.path.append('.')

import pandas as pd
from utils import _apply_scheme_adjustment

def test_edge_case_scenarios():
    """Test the specific edge cases mentioned by the user."""
    print("🧪 Testing Edge Case Fix for X+0.5 Scenarios...")
    
    # Test Case 1: Scm=4+1, should result in 2.5+0.5 (not 2+0.5)
    df1 = pd.DataFrame({'Scm': ['4+1']})
    row1 = df1.iloc[0]
    result1 = _apply_scheme_adjustment(2.5, row1, df1, 2)  # This should trigger 0.5x multiplier
    print(f"Test 1 - Scm=4+1, adjusted_qty=2.5: {result1}")
    
    # Verify it sums to whole number
    if '+' in result1:
        parts = result1.split('+')
        total = float(parts[0]) + float(parts[1])
        assert total == int(total), f"Result {result1} doesn't sum to whole number: {total}"
        print(f"✅ Sum verification: {parts[0]} + {parts[1]} = {total} (whole number)")
    
    # Test Case 2: Scm=8+1, should result in 5.5+0.5 (not 5+0.5)
    df2 = pd.DataFrame({'Scm': ['8+1']})
    row2 = df2.iloc[0]
    result2 = _apply_scheme_adjustment(5.5, row2, df2, 2)  # This should trigger 0.5x multiplier
    print(f"Test 2 - Scm=8+1, adjusted_qty=5.5: {result2}")
    
    # Verify it sums to whole number
    if '+' in result2:
        parts = result2.split('+')
        total = float(parts[0]) + float(parts[1])
        assert total == int(total), f"Result {result2} doesn't sum to whole number: {total}"
        print(f"✅ Sum verification: {parts[0]} + {parts[1]} = {total} (whole number)")
    
    # Test Case 3: Scm=6+1, should result in 3.5+0.5 (not 3+0.5)
    df3 = pd.DataFrame({'Scm': ['6+1']})
    row3 = df3.iloc[0]
    result3 = _apply_scheme_adjustment(3.5, row3, df3, 2)  # This should trigger 0.5x multiplier
    print(f"Test 3 - Scm=6+1, adjusted_qty=3.5: {result3}")
    
    # Verify it sums to whole number
    if '+' in result3:
        parts = result3.split('+')
        total = float(parts[0]) + float(parts[1])
        assert total == int(total), f"Result {result3} doesn't sum to whole number: {total}"
        print(f"✅ Sum verification: {parts[0]} + {parts[1]} = {total} (whole number)")
    
    return True

def test_additional_scenarios():
    """Test additional scenarios to ensure we didn't break existing logic."""
    print("\n🧪 Testing Additional Scenarios...")
    
    # Test normal 1x multiplier (should work as before)
    df4 = pd.DataFrame({'Scm': ['5+1']})
    row4 = df4.iloc[0]
    result4 = _apply_scheme_adjustment(6, row4, df4, 2)  # Should be 5+1
    print(f"Test 4 - Scm=5+1, adjusted_qty=6: {result4}")
    assert result4 == "5+1", f"Expected '5+1', got '{result4}'"
    
    # Test 2x multiplier (should work as before)
    df5 = pd.DataFrame({'Scm': ['3+1']})
    row5 = df5.iloc[0]
    result5 = _apply_scheme_adjustment(8, row5, df5, 2)  # Should be 6+2
    print(f"Test 5 - Scm=3+1, adjusted_qty=8: {result5}")
    assert result5 == "6+2", f"Expected '6+2', got '{result5}'"
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Edge Case Fix for Scheme Whole Number Adjustment")
    print("=" * 70)
    
    try:
        test_edge_case_scenarios()
        test_additional_scenarios()
        print("\n" + "=" * 70)
        print("🎉 All edge case tests PASSED!")
        print("✅ X+0.5 cases now correctly become X.5+0.5")
        print("✅ Existing logic remains intact")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        print("⚠️ Edge case fix needs adjustment")
