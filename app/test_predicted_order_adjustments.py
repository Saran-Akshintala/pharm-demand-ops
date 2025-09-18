"""
Unit tests for predicted order adjustments functionality.

Tests the deterministic logic for computing Predicted_Order from Predicted_Base_Quantity
with configurable box and scheme adjustments.
"""

import unittest
import pandas as pd
import numpy as np
from utils import (
    compute_predicted_order_with_adjustments,
    _apply_box_adjustment,
    _apply_scheme_adjustment,
    _calculate_demand_trend_factor,
    _round_to_half
)


class TestPredictedOrderAdjustments(unittest.TestCase):
    """Test cases for predicted order adjustments."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'Name': ['Product A', 'Product B', 'Product C', 'Product D'],
            'Predicted_Base_Quantity': [14, 7, 11, 6],
            'Stock': [4, 3, 2, 2],
            'Box': [10, 1, 5, 1],
            'Scm': ['3+1', '3+1', '5+1', '5+1'],
            'L7': [26, 10, 15, 5],
            'L15': [34, 8, 12, 4],
            'L30': [45, 6, 10, 3],
            'L45': [63, 5, 8, 2],
            'L60': [74, 4, 6, 2],
            'L75': [97, 3, 5, 1],
            'L90': [114, 2, 4, 1]
        })
    
    def test_compute_predicted_order_basic(self):
        """Test basic functionality with new priority rules."""
        result = compute_predicted_order_with_adjustments(
            self.sample_data.copy(),
            apply_box=True,
            box_tolerance=2,
            apply_scm=True,
            scm_tolerance=2
        )
        
        # Check that Predicted_Order column is created
        self.assertIn('Predicted_Order', result.columns)
        
        # Check that all rows have predicted orders
        self.assertEqual(len(result['Predicted_Order']), len(self.sample_data))
        
        # Check that no values are NaN
        self.assertFalse(result['Predicted_Order'].isna().any())
        
        # Test the priority rules:
        # Row 0: Base=14, Stock=4 -> Predicted_Order should start from 14-4=10
        # Row 1: Base=7, Stock=3 -> Predicted_Order should start from 7-3=4
        print(f"Test results: {result['Predicted_Order'].tolist()}")
    
    def test_box_adjustment_new_logic(self):
        """Test new box adjustment logic after stock subtraction."""
        from utils import _apply_box_adjustment_new
        
        row = pd.Series({'Box': 10})
        
        # Test case: predicted_order_qty=8, box=10, tolerance=2
        # Difference = |8-10| = 2, which is <= tolerance, so should adjust to 10
        result = _apply_box_adjustment_new(8, row, tolerance=2)
        self.assertEqual(result, 10)
        
        # Test case: predicted_order_qty=12, box=10, tolerance=2  
        # Difference = |12-10| = 2, which is <= tolerance, so should adjust to 10
        result = _apply_box_adjustment_new(12, row, tolerance=2)
        self.assertEqual(result, 10)
        
        # Test case: predicted_order_qty=15, box=10, tolerance=2
        # Difference = |15-10| = 5, which is > tolerance, so should keep original
        result = _apply_box_adjustment_new(15, row, tolerance=2)
        self.assertEqual(result, 15)
    
    def test_box_adjustment_edge_cases(self):
        """Test box adjustment edge cases."""
        # Missing box quantity
        row = pd.Series({'Box': None})
        result = _apply_box_adjustment(14, row, tolerance=2)
        self.assertEqual(result, 14)
        
        # Zero box quantity
        row = pd.Series({'Box': 0})
        result = _apply_box_adjustment(14, row, tolerance=2)
        self.assertEqual(result, 14)
        
        # Invalid box quantity
        row = pd.Series({'Box': 'invalid'})
        result = _apply_box_adjustment(14, row, tolerance=2)
        self.assertEqual(result, 14)
    
    def test_scheme_adjustment_whole_numbers(self):
        """Test new scheme adjustment ensuring whole number sums."""
        from utils import _apply_scheme_adjustment_new
        
        df = pd.DataFrame({'Scm': ['5+1']})
        row = pd.Series({'Scm': '5+1'})
        
        # Test case: predicted_order_qty=6, scheme=5+1 (total=6)
        # Should return 5+1 since 6 is within tolerance of 6
        result = _apply_scheme_adjustment_new(6, row, df, tolerance=2)
        self.assertEqual(result, '5+1')
        
        # Test case where result should sum to whole number
        # If we get something like 4+0.5, it should be adjusted to ensure whole sum
        df2 = pd.DataFrame({'Scm': ['9+1']})
        row2 = pd.Series({'Scm': '9+1'})
        result2 = _apply_scheme_adjustment_new(5, row2, df2, tolerance=2)
        
        # Parse the result and check it sums to a whole number
        if '+' in result2:
            parts = result2.split('+')
            total = float(parts[0]) + float(parts[1])
            self.assertEqual(total, int(total), f"Result {result2} should sum to whole number, got {total}")
    
    def test_scheme_adjustment_no_scheme(self):
        """Test scheme adjustment when no scheme is available."""
        df = pd.DataFrame({'Scm': ['0']})
        row = pd.Series({'Scm': '0'})
        
        result = _apply_scheme_adjustment(10, row, df, tolerance=2)
        self.assertEqual(result, '10')
        
        # Test missing scheme column
        df_no_scm = pd.DataFrame({'Other': ['value']})
        row_no_scm = pd.Series({'Other': 'value'})
        
        result = _apply_scheme_adjustment(10, row_no_scm, df_no_scm, tolerance=2)
        self.assertEqual(result, '10')
    
    def test_demand_trend_factor_calculation(self):
        """Test demand trend factor calculation."""
        # Test increasing trend
        row_increasing = pd.Series({
            'L7': 114, 'L15': 97, 'L30': 74, 'L45': 63, 'L60': 45, 'L75': 34, 'L90': 26
        })
        df = pd.DataFrame([row_increasing])
        factor = _calculate_demand_trend_factor(row_increasing, df)
        self.assertGreater(factor, 1.0)  # Should be > 1 for increasing trend
        
        # Test decreasing trend
        row_decreasing = pd.Series({
            'L7': 26, 'L15': 34, 'L30': 45, 'L45': 63, 'L60': 74, 'L75': 97, 'L90': 114
        })
        df = pd.DataFrame([row_decreasing])
        factor = _calculate_demand_trend_factor(row_decreasing, df)
        self.assertLess(factor, 1.0)  # Should be < 1 for decreasing trend
        
        # Test insufficient data
        row_insufficient = pd.Series({'L7': 10, 'L15': 12})
        df = pd.DataFrame([row_insufficient])
        factor = _calculate_demand_trend_factor(row_insufficient, df)
        self.assertEqual(factor, 1.0)  # Should be neutral
    
    def test_round_to_half(self):
        """Test rounding to nearest 0.5."""
        self.assertEqual(_round_to_half(1.2), 1.0)
        self.assertEqual(_round_to_half(1.3), 1.5)
        self.assertEqual(_round_to_half(1.7), 1.5)
        self.assertEqual(_round_to_half(1.8), 2.0)
        self.assertEqual(_round_to_half(2.0), 2.0)
    
    def test_priority_rules_integration(self):
        """Test the complete priority rules: Stock subtraction -> Box -> Scheme."""
        # Case: Base=14, Stock=4, Box=10, Scm=5+1
        # Step 1: 14 - 4 = 10
        # Step 2: 10 is exactly Box=10, so stays 10 (within tolerance)
        # Step 3: 10 vs 5+1=6, difference=4, if tolerance>=4, could become scheme
        
        test_data = pd.DataFrame({
            'Name': ['Test Product'],
            'Predicted_Base_Quantity': [14],
            'Stock': [4],
            'Box': [10],
            'Scm': ['5+1'],
            'L7': [10], 'L15': [12], 'L30': [8], 'L45': [9], 
            'L60': [11], 'L75': [7], 'L90': [6]
        })
        
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=True,
            box_tolerance=2,
            apply_scm=True,
            scm_tolerance=2
        )
        
        predicted_order = result['Predicted_Order'].iloc[0]
        print(f"Priority rules test result: {predicted_order}")
        
        # The result should reflect the priority order
        self.assertIsNotNone(predicted_order)
    
    def test_stock_subtraction_priority(self):
        """Test that stock subtraction happens first in priority."""
        # Case: Base=10, Stock=3 -> Should start with 10-3=7
        
        test_data = pd.DataFrame({
            'Name': ['Stock Test'],
            'Predicted_Base_Quantity': [10],
            'Stock': [3],
            'Box': [1],  # No box effect
            'Scm': ['0'],  # No scheme
        })
        
        result = compute_predicted_order_with_adjustments(
            test_data,
            apply_box=False,
            apply_scm=False
        )
        
        predicted_order = result['Predicted_Order'].iloc[0]
        self.assertEqual(predicted_order, '7')  # 10 - 3 = 7
        
        # Test negative case
        test_data2 = pd.DataFrame({
            'Name': ['Negative Test'],
            'Predicted_Base_Quantity': [5],
            'Stock': [8],  # Stock > Base
            'Box': [1],
            'Scm': ['0'],
        })
        
        result2 = compute_predicted_order_with_adjustments(
            test_data2,
            apply_box=False,
            apply_scm=False
        )
        
        predicted_order2 = result2['Predicted_Order'].iloc[0]
        self.assertEqual(predicted_order2, '0')  # Should be 0 when negative
    
    def test_apply_both_disabled(self):
        """Test when both box and scheme adjustments are disabled."""
        result = compute_predicted_order_with_adjustments(
            self.sample_data.copy(),
            apply_box=False,
            apply_scm=False,
            box_tolerance=2,
            scm_tolerance=2
        )
        
        # Should return values equal to Predicted_Base_Quantity - Stock
        for i, row in result.iterrows():
            base_qty = row['Predicted_Base_Quantity']
            stock_qty = row.get('Stock', 0)
            expected_qty = max(0, base_qty - stock_qty)  # Can't be negative
            expected = str(int(expected_qty)) if expected_qty == int(expected_qty) else f"{expected_qty:.1f}"
            if expected_qty <= 0:
                expected = '0'
            self.assertEqual(row['Predicted_Order'], expected)
    
    def test_error_handling_missing_column(self):
        """Test error handling when required column is missing."""
        invalid_data = pd.DataFrame({
            'Name': ['Product A'],
            'Stock': [4]
            # Missing Predicted_Base_Quantity
        })
        
        with self.assertRaises(ValueError):
            compute_predicted_order_with_adjustments(invalid_data)


class TestIntegrationWithBusinessRules(unittest.TestCase):
    """Integration tests with existing business rules."""
    
    def test_integration_with_business_rules(self):
        """Test that the new logic integrates properly with existing business rules."""
        from utils import apply_predicted_order_business_rules
        
        # Create test data that would trigger business rules
        test_data = pd.DataFrame({
            'Name': ['Product A'],
            'Predicted_Base_Quantity': [10],
            'Predicted_Order': ['10+2'],  # Will be computed by our function
            'Stock': [15],  # High stock should trigger "No Order"
            'Box': [5],
            'Days': [30],  # Normal days
            'No_of_Customer_Last_Month': [5],
            'L7': [10], 'L15': [12], 'L30': [8], 'L45': [9], 
            'L60': [11], 'L75': [7], 'L90': [6]
        })
        
        # First apply our adjustments
        result = compute_predicted_order_with_adjustments(test_data)
        
        # Then apply business rules
        final_result, styling_info = apply_predicted_order_business_rules(result)
        
        # Should have styling information
        self.assertIsInstance(styling_info, dict)
        
        # Should have Predicted_Order column
        self.assertIn('Predicted_Order', final_result.columns)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
