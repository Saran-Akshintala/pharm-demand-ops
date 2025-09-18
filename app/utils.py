"""
Utility functions for pharmacy demand forecasting.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import joblib
import xgboost as xgb
import json
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder


def parse_order_scheme(order_str: str) -> Tuple[int, str]:
    """
    Parse order strings like '9+1', '12', '5+2' into base quantity and scheme info.
    
    Args:
        order_str: Order string from Excel (e.g., '9+1', '12')
    
    Returns:
        Tuple of (base_quantity, scheme_info)
    """
    if pd.isna(order_str) or order_str == '':
        return 0, 'none'
    
    order_str = str(order_str).strip()
    
    # Check for scheme patterns like '9+1', '12+2', etc.
    scheme_pattern = r'(\d+)\+(\d+)'
    match = re.match(scheme_pattern, order_str)
    
    if match:
        base_qty = int(match.group(1))
        bonus_qty = int(match.group(2))
        scheme_info = f"{base_qty}+{bonus_qty}"
        return base_qty, scheme_info
    
    # Try to parse as simple number
    try:
        base_qty = int(float(order_str))
        return base_qty, 'none'
    except (ValueError, TypeError):
        return 0, 'unknown'


def reconstruct_order_prediction(base_qty: int, scheme_info: str = None, scheme_prob: float = 0.3) -> str:
    """
    Reconstruct order string from predicted base quantity using scheme patterns.
    
    Args:
        base_qty: Predicted base quantity
        scheme_info: Scheme pattern from training data (e.g., '11+1', '9+1')
        scheme_prob: Probability of having a scheme (default 0.3)
    
    Returns:
        Order string like '12', '9+1', '5.5+0.5', etc.
    """
    if base_qty <= 0:
        return "0"
    
    # If scheme info is provided, use it to determine pattern
    if scheme_info and scheme_info != 'None' and '+' in str(scheme_info):
        try:
            # Parse scheme pattern like '11+1' to understand ratio
            parts = str(scheme_info).split('+')
            if len(parts) == 2:
                scheme_base = float(parts[0])
                scheme_bonus = float(parts[1])
                
                # Calculate ratio and apply to predicted quantity
                if scheme_base > 0:
                    ratio = scheme_bonus / scheme_base
                    predicted_bonus = base_qty * ratio
                    
                    # Handle fractional bonuses
                    if predicted_bonus < 1:
                        return f"{base_qty}+{predicted_bonus:.1f}"
                    else:
                        return f"{base_qty}+{int(round(predicted_bonus))}"
        except:
            pass
    
    # Fallback to heuristic approach
    if base_qty >= 9 and np.random.random() < scheme_prob:
        # Common schemes: +1, +2, or fractional for smaller quantities
        if base_qty < 10:
            bonus = 0.5
            return f"{base_qty}+{bonus}"
        elif base_qty < 15:
            bonus = 1
            return f"{base_qty}+{bonus}"
        else:
            bonus = 2
            return f"{base_qty}+{bonus}"
    else:
        return str(base_qty)


def preprocess_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess sales features (L7, L15, L30, etc.) for model training.
    
    Args:
        df: DataFrame with sales columns
    
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    # Define sales columns
    sales_cols = [col for col in df.columns if col.startswith('L') and len(col) > 1 and col[1:].isdigit()]
    
    if len(sales_cols) >= 2:
        # Sales trends
        if 'L7' in df.columns and 'L15' in df.columns:
            df_processed['sales_trend_7_15'] = df_processed['L7'] / (df_processed['L15'] + 1)  # +1 to avoid division by zero
        
        if 'L15' in df.columns and 'L30' in df.columns:
            df_processed['sales_trend_15_30'] = df_processed['L15'] / (df_processed['L30'] + 1)
        
        # Average sales velocity
        if sales_cols:
            df_processed['avg_sales_velocity'] = df_processed[sales_cols].mean(axis=1)
        
        # Maximum sales period
        if sales_cols:
            df_processed['max_sales_period'] = df_processed[sales_cols].max(axis=1)
        
        # Sales volatility (standard deviation)
        if len(sales_cols) > 1:
            df_processed['sales_volatility'] = df_processed[sales_cols].std(axis=1)
    
    # Add numerical features
    numerical_features = ['Stock', 'Sales_Qty', 'Days', 'Max_Qty', 'Unit', 'Box', 'Pur_Dis', 'No_of_Customer_Last_Month']
    for feat in numerical_features:
        if feat in df_processed.columns:
            df_processed[f'{feat}_clean'] = pd.to_numeric(df_processed[feat], errors='coerce').fillna(0)
    
    # Create interaction features: Stock × Sales patterns
    if 'Stock_clean' in df_processed.columns and 'L30' in df_processed.columns:
        df_processed['stock_to_sales_ratio'] = df_processed['Stock_clean'] / (df_processed['L30'] + 1)
    
    if 'Stock_clean' in df_processed.columns and 'avg_sales_velocity' in df_processed.columns:
        df_processed['stock_velocity_interaction'] = df_processed['Stock_clean'] * df_processed['avg_sales_velocity']
    
    # Add seasonality features from date
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        df_processed['month'] = df_processed['Date'].dt.month
        df_processed['quarter'] = df_processed['Date'].dt.quarter
        df_processed['day_of_week'] = df_processed['Date'].dt.dayofweek
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for seasonality
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
        df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
    
    # Encode categorical features
    categorical_features = ['Supplier', 'Best_Supplier', 'Company', 'Pack', 'Scm']
    for feat in categorical_features:
        if feat in df_processed.columns:
            # Clean and encode categorical data
            df_processed[feat] = df_processed[feat].astype(str).fillna('Unknown')
            le = LabelEncoder()
            df_processed[f'{feat}_encoded'] = le.fit_transform(df_processed[feat])
    
    # Product categorization from Name patterns
    if 'Name' in df_processed.columns:
        df_processed['Name'] = df_processed['Name'].astype(str).fillna('Unknown')
        
        # Extract medicine type patterns
        df_processed['is_tablet'] = df_processed['Name'].str.contains('TAB', case=False, na=False).astype(int)
        df_processed['is_capsule'] = df_processed['Name'].str.contains('CAP', case=False, na=False).astype(int)
        df_processed['is_syrup'] = df_processed['Name'].str.contains('SYR|SYRUP', case=False, na=False).astype(int)
        df_processed['is_injection'] = df_processed['Name'].str.contains('INJ', case=False, na=False).astype(int)
        df_processed['is_cream'] = df_processed['Name'].str.contains('CREAM|OINT', case=False, na=False).astype(int)
        
        # Extract pack size from Name if not in Pack column
        pack_pattern = r'(\d+(?:\.\d+)?\s*(?:MG|ML|GM|G|L)?)'
        df_processed['extracted_dose'] = df_processed['Name'].str.extract(pack_pattern, expand=False)
        
        # Name length as complexity indicator
        df_processed['name_length'] = df_processed['Name'].str.len()
        
        # Encode product names for similarity
        le_name = LabelEncoder()
        df_processed['name_encoded'] = le_name.fit_transform(df_processed['Name'])
    
    return df_processed


def load_and_merge_excel_files(data_dir: str) -> pd.DataFrame:
    """
    Load and merge multiple Excel files from the data directory.
    
    Args:
        data_dir: Path to directory containing Excel files
    
    Returns:
        Merged DataFrame with Date column
    """
    data_path = Path(data_dir)
    excel_files = list(data_path.glob("*.xlsx"))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")
    
    dfs = []
    
    for file_path in excel_files:
        try:
            # Extract date from filename (assuming format like "0102 A19.xlsx")
            filename = file_path.stem
            date_match = re.search(r'(\d{4})', filename)
            
            if date_match:
                # Parse date from filename
                date_str = date_match.group(1)
                month = int(date_str[:2])
                day = int(date_str[2:])
                
                # Validate month and day
                if month < 1 or month > 12 or day < 1 or day > 31:
                    # Use file modification time as fallback
                    date = pd.to_datetime(file_path.stat().st_mtime, unit='s')
                else:
                    # Assume current year or extract from filename if available
                    year = 2019 if 'A19' in filename else 2023
                    try:
                        date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                    except:
                        date = pd.to_datetime(file_path.stat().st_mtime, unit='s')
            else:
                # Use file modification time as fallback
                date = pd.to_datetime(file_path.stat().st_mtime, unit='s')
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip().str.replace(' ', '_')
            
            # Add metadata
            df['Date'] = date
            df['Source_File'] = file_path.name
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid Excel files could be loaded")
    
    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df


def prepare_features_for_prediction(df: pd.DataFrame, 
                                  feature_columns: list = None) -> pd.DataFrame:
    """
    Prepare features for model prediction.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns to use
    
    Returns:
        DataFrame ready for prediction
    """
    df_features = df.copy()
    
    # Standardize column names - handle spaces and variations
    column_mapping = {}
    for col in df_features.columns:
        if isinstance(col, str):
            # Remove spaces and standardize
            clean_col = col.strip().replace(' ', '_')
            if clean_col != col:
                column_mapping[col] = clean_col
    
    if column_mapping:
        df_features = df_features.rename(columns=column_mapping)
    
    # Define all possible sales columns
    all_sales_cols = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
    
    # Find available sales columns in the data
    available_sales_cols = []
    for col in all_sales_cols:
        if col in df_features.columns:
            available_sales_cols.append(col)
    
    # If no feature columns specified, use available sales columns
    if feature_columns is None:
        feature_columns = available_sales_cols
    
    # Ensure all required feature columns exist with default values
    for col in feature_columns:
        if col not in df_features.columns:
            df_features[col] = 0
    
    # Convert sales columns to numeric
    for col in available_sales_cols:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
    
    # Preprocess sales features to create derived features
    df_features = preprocess_sales_features(df_features)
    
    # Define the exact feature set that should be used for prediction
    # This should match what was used during training
    standard_features = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
    derived_features = ['sales_trend_7_15', 'sales_trend_15_30', 'avg_sales_velocity', 
                       'max_sales_period', 'sales_volatility']
    
    # Add all new feature categories
    numerical_features = ['Stock_clean', 'Sales_Qty_clean', 'Days_clean', 'Max_Qty_clean', 
                         'Unit_clean', 'Box_clean', 'Pur_Dis_clean', 'No_of_Customer_Last_Month_clean']
    interaction_features = ['stock_to_sales_ratio', 'stock_velocity_interaction']
    seasonality_features = ['month', 'quarter', 'day_of_week', 'is_weekend', 
                           'month_sin', 'month_cos', 'day_sin', 'day_cos']
    categorical_encoded = ['Supplier_encoded', 'Best_Supplier_encoded', 'Company_encoded', 
                          'Pack_encoded', 'Scm_encoded']
    product_features = ['is_tablet', 'is_capsule', 'is_syrup', 'is_injection', 'is_cream',
                       'name_length', 'name_encoded']
    
    # Ensure all standard features exist
    for col in standard_features:
        if col not in df_features.columns:
            df_features[col] = 0
    
    # Select features that actually exist and are in the feature_columns list
    final_features = []
    
    # Add sales features
    for col in standard_features:
        if col in df_features.columns:
            final_features.append(col)
    
    # Add derived features if they exist
    for col in derived_features:
        if col in df_features.columns:
            final_features.append(col)
    
    # Add all new feature categories if they exist
    all_new_features = numerical_features + interaction_features + seasonality_features + categorical_encoded + product_features
    for col in all_new_features:
        if col in df_features.columns:
            final_features.append(col)
    
    # If feature_columns is specified, use only those that exist
    if feature_columns is not None:
        final_features = [col for col in feature_columns if col in df_features.columns]
    
    # Ensure we have the exact columns in the right order
    result_df = df_features[final_features].copy()
    
    # Fill any remaining NaN values
    result_df = result_df.fillna(0)
    
    return result_df


def save_model(model, model_path: str):
    """Save trained model to disk."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str):
    """Load trained model from disk with XGBoost compatibility fallback.
    - Tries joblib (pickle) first
    - If XGBoost 'gpu_id' AttributeError occurs (version mismatch), tries JSON booster next to the pkl
    - Raises a helpful RuntimeError with remediation steps if loading still fails
    """
    try:
        return joblib.load(model_path)
    except AttributeError as e:
        msg = str(e).lower()
        if 'gpu_id' in msg or 'xgbmodel' in msg:
            # Attempt JSON fallback (if user exported booster for compatibility)
            json_path = Path(model_path).with_suffix('.json')
            try:
                if json_path.exists():
                    model = xgb.XGBRegressor()
                    model.load_model(str(json_path))
                    return model
            except Exception:
                pass
            raise RuntimeError(
                "XGBoost version mismatch detected while loading the model (gpu_id attribute not found). "
                "Fix options: (1) Use Python 3.10/3.11 and install xgboost==1.7.6, then rerun; "
                "(2) Retrain/resave the model with your current XGBoost version; "
                "(3) From the training environment, export booster to JSON via model.get_booster().save_model('models/order_predictor.json') "
                "and place it next to the .pkl — this app will auto-load the JSON."
            ) from e
        # Unknown AttributeError, bubble up
        raise


def validate_input_data(df: pd.DataFrame) -> dict:
    """
    Validate input data and return validation results.
    
    Args:
        df: Input DataFrame to validate
    
    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    results = {'errors': [], 'warnings': []}
    
    # Check if DataFrame is empty
    if df.empty:
        results['errors'].append("Input DataFrame is empty")
        return results
    
    # Standardize column names for validation
    df_check = df.copy()
    df_check.columns = df_check.columns.astype(str).str.strip().str.replace(' ', '_')
    
    # Check for sales columns with flexible naming
    sales_cols = []
    for col in df_check.columns:
        if isinstance(col, str) and col.startswith('L') and len(col) > 1 and col[1:].isdigit():
            sales_cols.append(col)
    
    if not sales_cols:
        results['warnings'].append("No sales columns (L7, L15, etc.) found - will use default values")
    
    # Check for missing values in sales columns
    if sales_cols:
        missing_pct = df_check[sales_cols].isnull().sum() / len(df_check)
        high_missing = missing_pct[missing_pct > 0.5]
        if not high_missing.empty:
            results['warnings'].append(f"High missing values in columns: {high_missing.index.tolist()}")
    
    # Check data types for sales columns
    for col in sales_cols:
        if col in df_check.columns:
            non_numeric = pd.to_numeric(df_check[col], errors='coerce').isnull().sum()
            if non_numeric > 0:
                results['warnings'].append(f"Non-numeric values found in {col}: {non_numeric} rows")
    
    # Check for Order column - ignore existing values, always predict
    order_cols = [col for col in df_check.columns if 'order' in col.lower()]
    if order_cols:
        order_col = order_cols[0]
        filled_orders = df_check[order_col].notna().sum()
        if filled_orders > 0:
            results['warnings'].append(f"Existing Order values will be ignored - generating fresh predictions for all {len(df_check)} products")
    
    # Check for essential pharmacy columns
    expected_cols = ['Name', 'L7', 'L15', 'L30']
    missing_essential = []
    for col in expected_cols:
        if col not in df_check.columns:
            missing_essential.append(col)
    
    if missing_essential:
        results['warnings'].append(f"Missing recommended columns: {missing_essential}")
    
    return results


def apply_predicted_order_business_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Apply business rules to Predicted_Order column with priority-based highlighting.
    
    Args:
        df: DataFrame with Predicted_Order and other required columns
    
    Returns:
        Tuple of (modified_df, styling_info_dict)
        styling_info_dict: {row_index: {'color': str, 'tooltip': str, 'display_value': str}}
    """
    df_result = df.copy()
    styling_info = {}
    
    # Define color constants
    COLORS = {
        'reddish': '#ffcccc',     # Light red
        'greenish': '#ccffcc',    # Light green  
        'orangish': '#ffe6cc',    # Light orange
        'yellowish': '#ffffcc'    # Light yellow
    }
    
    # Priority order (highest first)
    PRIORITY_ORDER = [
        'days_over_90',       # 1. Days > 90 → Reddish
        'uneven_sales',       # 2. Uneven sales pattern → Reddish
        'negative_after_stock', # 3. Negative after Stock subtraction → Greenish
        'box_adjustment',     # 4. Box adjustment within ±2 → Orangish
        'low_customers'       # 5. No. of Customer Last Month ≤ 2 → Yellowish
    ]
    
    for idx, row in df_result.iterrows():
        conditions = {}
        
        # Get original predicted value (before any modifications)
        original_predicted = row.get('Predicted_Base_Quantity', 0)
        
        # Rule 2: Adjust to nearest Box quantity
        box_qty = row.get('Box', 1)  # Default to 1 if Box column missing
        if pd.notna(box_qty) and box_qty > 0:
            # Convert to numeric to handle string values
            box_qty = pd.to_numeric(box_qty, errors='coerce')
            if pd.notna(box_qty) and box_qty > 0:
                # Round to nearest multiple of box quantity
                # Example: original_predicted=28, box_qty=10 -> adjusted_to_box=30
                adjusted_to_box = round(original_predicted / box_qty) * box_qty
                box_diff = abs(original_predicted - adjusted_to_box)
                
                # Always highlight when adjustment is made (not just within ±2)
                if box_diff > 0:
                    conditions['box_adjustment'] = {
                        'color': COLORS['orangish'],
                        'tooltip': f'Adjusted to nearest Box qty (diff: ±{box_diff:.1f})'
                    }
            else:
                adjusted_to_box = original_predicted
        else:
            adjusted_to_box = original_predicted
        
        # Rule 3: Subtract Stock from Predicted_Order
        stock_qty = row.get('Stock', 0)
        if pd.notna(stock_qty):
            final_predicted = adjusted_to_box - stock_qty
        else:
            final_predicted = adjusted_to_box
        
        # Rule 3b: Check if negative after stock subtraction
        if final_predicted < 0:
            conditions['negative_after_stock'] = {
                'color': COLORS['greenish'],
                'tooltip': f'Negative after Stock subtraction ({final_predicted})'
            }
        
        # Rule 3c: Check if result should be "No Order" - ONLY when stock truly covers the need
        original_predicted = row.get('Predicted_Base_Quantity', 0)
        stock_qty = row.get('Stock', 0)
        
        # "No Order" only applies in these specific cases:
        # 1. Stock >= Predicted_Base_Quantity (stock fully covers need)
        # 2. Predicted_Base_Quantity is 0 or very low (genuinely no demand)
        should_be_no_order = False
        tooltip_reason = ""
        
        if pd.notna(stock_qty) and stock_qty > 0 and stock_qty >= original_predicted and original_predicted > 0:
            # Case 1: Stock fully covers the predicted need
            should_be_no_order = True
            tooltip_reason = f"In-Stock: Current stock ({stock_qty}) covers predicted need ({original_predicted})"
        elif original_predicted <= 0:
            # Case 2: Genuinely no demand predicted
            should_be_no_order = True
            tooltip_reason = f"Low demand: Predicted quantity ({original_predicted}) is very low"
        
        if should_be_no_order:
            display_value = "No Order"
            # Clear all existing conditions and create specific "No Order" tooltip
            conditions.clear()
            conditions['no_order'] = {
                'color': None,  # No background color
                'tooltip': tooltip_reason
            }
        elif final_predicted <= 0:
            # If calculation results in ≤0 but we shouldn't show "No Order", 
            # use the original predicted quantity (before stock subtraction)
            final_predicted = original_predicted
            display_value = None  # Will be set later with scheme logic
        else:
            display_value = None  # Will be set later with scheme logic
        
        # Rule 4: Check uneven sales patterns (repeating 1s or 2s)
        sales_cols = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
        available_sales = [col for col in sales_cols if col in df_result.columns]
        
        if available_sales:
            sales_values = [row.get(col, 0) for col in available_sales]
            # Convert to numeric and handle NaN
            sales_values = [pd.to_numeric(val, errors='coerce') for val in sales_values]
            sales_values = [val for val in sales_values if pd.notna(val)]
            
            if sales_values:
                # Check for repeating patterns of 1 or 2
                ones_count = sales_values.count(1)
                twos_count = sales_values.count(2)
                total_values = len(sales_values)
                
                # If more than 50% are 1s or 2s, consider it uneven
                if (ones_count + twos_count) / total_values > 0.5:
                    conditions['uneven_sales'] = {
                        'color': COLORS['reddish'],
                        'tooltip': f'High uneven sales pattern (1s: {ones_count}, 2s: {twos_count})'
                    }
        
        # Rule 5: Check Days > 90
        days_since_purchase = row.get('Days', 0)
        if pd.notna(days_since_purchase) and days_since_purchase > 90:
            conditions['days_over_90'] = {
                'color': COLORS['reddish'],
                'tooltip': f'Days since last purchase > 90 ({days_since_purchase})'
            }
        
        # Rule 6: Check No. of Customer Last Month ≤ 2
        # Try different possible column name variations
        customer_count = None
        customer_col_variations = [
            'No_of_Customer_Last_Month', 
            'No. of Customer Last Month',
            'No of Customer Last Month',
            'No_of_Customer_Last_Month_clean'
        ]
        
        for col_name in customer_col_variations:
            if col_name in df_result.columns:
                customer_count = row.get(col_name)
                break
        
        # If still not found, try case-insensitive search
        if customer_count is None:
            for col in df_result.columns:
                if 'customer' in str(col).lower() and 'month' in str(col).lower():
                    customer_count = row.get(col)
                    break
        
        # Apply the rule only if we found a valid customer count
        if customer_count is not None and pd.notna(customer_count):
            try:
                customer_count_num = pd.to_numeric(customer_count, errors='coerce')
                if pd.notna(customer_count_num) and customer_count_num <= 2:
                    conditions['low_customers'] = {
                        'color': COLORS['yellowish'],
                        'tooltip': f'Low customer count last month ({customer_count_num})'
                    }
            except:
                pass
        
        # Apply Scm. scheme logic to the final predicted quantity (if not already set as "No Order")
        if display_value is None:
            display_value = apply_scheme_to_quantity(final_predicted, row, df_result)
            
            # Special constraint: If Days > 90, cap the order at Predicted_Base_Quantity
            # This prevents over-ordering of potentially stale/slow-moving items
            days_since_purchase = row.get('Days', 0)
            if pd.notna(days_since_purchase) and days_since_purchase > 90:
                # Parse the display_value to get total quantity
                if '+' in str(display_value):
                    # Handle scheme format like "10+2"
                    try:
                        parts = str(display_value).split('+')
                        if len(parts) == 2:
                            base_part = float(parts[0])
                            bonus_part = float(parts[1])
                            total_scheme_qty = base_part + bonus_part
                            
                            # If scheme total exceeds base prediction, cap it
                            if total_scheme_qty > original_predicted:
                                # Use the original predicted quantity instead
                                display_value = str(int(original_predicted)) if original_predicted == int(original_predicted) else str(original_predicted)
                                
                                # Update the tooltip to reflect this capping
                                if 'days_over_90' in conditions:
                                    conditions['days_over_90']['tooltip'] = f'Days > 90 ({days_since_purchase}): Capped order at base quantity ({original_predicted}) instead of scheme quantity ({total_scheme_qty})'
                    except:
                        pass
                else:
                    # Handle simple numeric format
                    try:
                        numeric_value = float(display_value)
                        if numeric_value > original_predicted:
                            display_value = str(int(original_predicted)) if original_predicted == int(original_predicted) else str(original_predicted)
                            
                            # Update the tooltip to reflect this capping
                            if 'days_over_90' in conditions:
                                conditions['days_over_90']['tooltip'] = f'Days > 90 ({days_since_purchase}): Capped order at base quantity ({original_predicted}) instead of calculated quantity ({numeric_value})'
                    except:
                        pass
        
        # Apply priority-based conflict resolution
        selected_condition = None
        
        # Special handling for "No Order" - always store tooltip but no color
        if 'no_order' in conditions:
            selected_condition = conditions['no_order']
        else:
            # Normal priority-based resolution for other conditions
            for priority_key in PRIORITY_ORDER:
                if priority_key in conditions:
                    condition = conditions[priority_key]
                    # Skip conditions that have color set to None
                    if condition.get('color') is not None:
                        selected_condition = condition
                        break
        
        # Update the dataframe with final values
        df_result.at[idx, 'Predicted_Order'] = display_value
        
        # Store styling information (including "No Order" tooltips)
        if selected_condition:
            styling_info[idx] = {
                'color': selected_condition['color'],
                'tooltip': selected_condition['tooltip'],
                'display_value': display_value
            }
    
    return df_result, styling_info


def apply_scheme_to_quantity(final_quantity: float, row: pd.Series, df: pd.DataFrame) -> str:
    """
    Apply Scm. scheme logic to the final predicted quantity.
    Intelligently rounds up to nearest scheme when beneficial.
    
    Args:
        final_quantity: Final predicted quantity after business rules
        row: Current row data
        df: DataFrame containing the data
    
    Returns:
        Formatted order string following the scheme pattern
    """
    # Check for Scm. column (note the period)
    scm_col = None
    for col in ['Scm', 'Scm.', 'scm', 'scm.']:
        if col in df.columns:
            scm_col = col
            break
    
    if scm_col and pd.notna(row[scm_col]):
        scm_value = str(row[scm_col]).strip()
        
        # Check for no scheme cases - use simple base quantity
        if scm_value in ['0', '0+0', '', 'nan', 'null', 'none']:
            return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
        elif '+' in scm_value:
            # Parse scheme pattern (e.g., "5+1", "2+1")
            try:
                scheme_parts = scm_value.split('+')
                if len(scheme_parts) == 2:
                    scheme_base = float(scheme_parts[0])
                    scheme_bonus = float(scheme_parts[1])
                    
                    # If either part is 0, treat as no scheme
                    if scheme_base == 0 or scheme_bonus == 0:
                        return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
                    else:
                        total_scheme = scheme_base + scheme_bonus
                        
                        # Intelligent scheme application logic:
                        # Apply proportional scheme based on final_quantity relative to scheme total
                        
                        # Case 1: Very small quantities (≤ 1) - return as-is
                        if final_quantity <= 1:
                            return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
                        
                        # Case 2: Find optimal scheme multiplier that gets closest to needed quantity
                        # Calculate which multiple of the scheme gets closest to final_quantity
                        
                        # Helper functions
                        def round_to_half(value):
                            return round(value * 2) / 2
                        
                        def format_value(val):
                            if val == int(val):
                                return str(int(val))
                            else:
                                return f"{val:.1f}"
                        
                        # Analyze demand trend to influence multiplier selection
                        sales_cols = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
                        available_sales = [col for col in sales_cols if col in df.columns]
                        
                        demand_trend_factor = 1.0  # Default neutral
                        
                        if len(available_sales) >= 3:
                            # Get sales values and calculate trend
                            sales_values = []
                            for col in available_sales:
                                val = row.get(col, 0)
                                if pd.notna(val):
                                    sales_values.append(pd.to_numeric(val, errors='coerce'))
                            
                            # Remove NaN values
                            sales_values = [val for val in sales_values if pd.notna(val)]
                            
                            if len(sales_values) >= 3:
                                # Enhanced trend analysis: compare recent vs older sales AND check overall growth
                                recent_avg = sum(sales_values[:2]) / 2 if len(sales_values) >= 2 else sales_values[0]
                                older_avg = sum(sales_values[-2:]) / 2 if len(sales_values) >= 2 else sales_values[-1]
                                
                                # Also check overall growth pattern (first vs last)
                                overall_growth = sales_values[0] / sales_values[-1] if sales_values[-1] > 0 else 1
                                
                                # Check for consistent growth across multiple periods
                                growth_consistency = 0
                                for i in range(len(sales_values) - 1):
                                    if sales_values[i] > sales_values[i + 1]:
                                        growth_consistency += 1
                                
                                consistency_ratio = growth_consistency / (len(sales_values) - 1) if len(sales_values) > 1 else 0
                                
                                if older_avg > 0:
                                    trend_ratio = recent_avg / older_avg
                                    
                                    # Enhanced trend factor calculation
                                    if trend_ratio < 0.7:  # Decreasing trend
                                        demand_trend_factor = 0.8
                                    elif trend_ratio > 2.0 and consistency_ratio > 0.6:  # Strong consistent growth
                                        demand_trend_factor = 1.8  # Very aggressive for strong trends
                                    elif trend_ratio > 1.5 and consistency_ratio > 0.5:  # Good growth
                                        demand_trend_factor = 1.5  # More aggressive
                                    elif trend_ratio > 1.3:  # Moderate growth
                                        demand_trend_factor = 1.2  # Standard increase
                        
                        # Test valid scheme multipliers to find the best fit
                        # Valid multipliers: 0.5x, 1x, 2x, 3x, 4x, 5x (maintaining base:bonus ratio)
                        valid_multipliers = [0.5, 1, 2, 3, 4, 5]
                        
                        best_multiplier = 1
                        best_score = float('inf')
                        
                        # Test each valid multiplier with trend consideration
                        for test_mult in valid_multipliers:
                            test_total = total_scheme * test_mult
                            difference = abs(final_quantity - test_total)
                            
                            # Apply enhanced trend factor: more aggressive for strong growth
                            if demand_trend_factor >= 1.8 and test_total > final_quantity:
                                # Very strong bonus for higher quantities with strong growth
                                adjusted_difference = difference * 0.4
                            elif demand_trend_factor >= 1.5 and test_total > final_quantity:
                                # Strong bonus for higher quantities with good growth
                                adjusted_difference = difference * 0.5
                            elif demand_trend_factor > 1.0 and test_total > final_quantity:
                                # Standard bonus for higher quantities when demand is increasing
                                adjusted_difference = difference * 0.7
                            elif demand_trend_factor < 1.0 and test_total < final_quantity:
                                # Bonus for lower quantities when demand is decreasing
                                adjusted_difference = difference * 0.7
                            else:
                                adjusted_difference = difference
                            
                            if adjusted_difference < best_score:
                                best_score = adjusted_difference
                                best_multiplier = test_mult
                        
                        # Apply the best multiplier
                        new_base = round_to_half(scheme_base * best_multiplier)
                        new_bonus = round_to_half(scheme_bonus * best_multiplier)
                        
                        return f"{format_value(new_base)}+{format_value(new_bonus)}"
                else:
                    return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
            except Exception:
                return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
        else:
            # Single number or other format
            return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
    else:
        # No Scm column - fallback to original logic using Order column
        order_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['order', 'ord', 'oreder']:
                order_col = col
                break
        
        if order_col and pd.notna(row[order_col]):
            try:
                _, scheme_info = parse_order_scheme(str(row[order_col]))
                return reconstruct_order_prediction(int(final_quantity), scheme_info)
            except:
                return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)
        else:
            return str(int(final_quantity)) if final_quantity == int(final_quantity) else str(final_quantity)


def detect_uneven_sales_pattern(sales_values: list) -> bool:
    """
    Detect if sales pattern shows high frequency of repeating low numbers (1, 2).
    
    Args:
        sales_values: List of sales values from L7, L15, L30, etc.
    
    Returns:
        True if pattern is considered uneven
    """
    if not sales_values or len(sales_values) < 3:
        return False
    
    # Count occurrences of 1 and 2
    ones_count = sales_values.count(1)
    twos_count = sales_values.count(2)
    total_count = len(sales_values)
    
    # If more than 50% of values are 1 or 2, consider it uneven
    uneven_ratio = (ones_count + twos_count) / total_count
    return uneven_ratio > 0.5


def compute_predicted_order_with_adjustments(
    df: pd.DataFrame, 
    apply_box: bool = True, 
    box_tolerance: int = 2,
    apply_scm: bool = True, 
    scm_tolerance: int = 2
) -> pd.DataFrame:
    """
    Compute Predicted_Order from Predicted_Base_Quantity with specific priority rules.
    
    Priority Order:
    1. Predicted_Order = Predicted_Base_Quantity - Stock
    2. Box adjustment (±tolerance to fit box quantity)
    3. Scheme adjustment (±tolerance to fit scheme, ensuring whole numbers)
    
    Args:
        df: DataFrame with Predicted_Base_Quantity and other required columns
        apply_box: Whether to apply box quantity adjustments
        box_tolerance: Tolerance for box quantity adjustments (±tolerance)
        apply_scm: Whether to apply scheme adjustments
        scm_tolerance: Tolerance for scheme adjustments (±tolerance)
    
    Returns:
        DataFrame with updated Predicted_Order column
    """
    df_result = df.copy()
    
    # Ensure Predicted_Base_Quantity exists
    if 'Predicted_Base_Quantity' not in df_result.columns:
        raise ValueError("Predicted_Base_Quantity column is required")
    
    predicted_orders = []
    
    for idx, row in df_result.iterrows():
        base_qty = row.get('Predicted_Base_Quantity', 0)
        stock_qty = row.get('Stock', 0)
        
        # Ensure values are numeric
        if pd.isna(base_qty):
            base_qty = 0
        else:
            base_qty = float(base_qty)
            
        if pd.isna(stock_qty):
            stock_qty = 0
        else:
            stock_qty = float(stock_qty)
        
        # Step 1: Predicted_Order = Predicted_Base_Quantity - Stock
        predicted_order_qty = base_qty - stock_qty
        
        # If result is negative or zero, handle appropriately
        if predicted_order_qty <= 0:
            predicted_orders.append("0")
            continue
        
        # Step 2: Apply box quantity adjustment if enabled
        if apply_box:
            predicted_order_qty = _apply_box_adjustment_new(predicted_order_qty, row, box_tolerance)
        
        # Step 3: Apply scheme adjustment if enabled
        if apply_scm:
            final_order = _apply_scheme_adjustment_new(predicted_order_qty, row, df_result, scm_tolerance)
        else:
            # Format as integer if whole number, otherwise as decimal
            if predicted_order_qty == int(predicted_order_qty):
                final_order = str(int(predicted_order_qty))
            else:
                final_order = f"{predicted_order_qty:.1f}"
        
        predicted_orders.append(final_order)
    
    df_result['Predicted_Order'] = predicted_orders
    return df_result


def _apply_box_adjustment_new(predicted_order_qty: float, row: pd.Series, tolerance: int) -> float:
    """
    Apply box quantity adjustment to predicted order quantity after stock subtraction.
    
    Rule: If predicted order quantity is within ±tolerance of box quantity, 
    adjust to box quantity to fit into box.
    
    Args:
        predicted_order_qty: Quantity after stock subtraction
        row: Current row data
        tolerance: Tolerance for box adjustments (±tolerance)
    
    Returns:
        Adjusted quantity based on box size
    """
    box_qty = row.get('Box', None)
    
    # Handle missing or invalid box quantities
    if pd.isna(box_qty) or box_qty is None:
        return predicted_order_qty
    
    try:
        box_qty = float(box_qty)
        if box_qty <= 0:
            return predicted_order_qty
    except (ValueError, TypeError):
        return predicted_order_qty
    
    # Check if predicted order quantity is within tolerance of box quantity
    difference = abs(predicted_order_qty - box_qty)
    
    if difference <= tolerance:
        # Adjust to box quantity
        return box_qty
    else:
        # Keep original quantity if not within tolerance
        return predicted_order_qty


def _apply_box_adjustment(base_qty: float, row: pd.Series, tolerance: int) -> float:
    """
    Legacy box adjustment function - kept for backward compatibility.
    
    Args:
        base_qty: Base predicted quantity
        row: Current row data
        tolerance: Tolerance for box adjustments (±tolerance)
    
    Returns:
        Adjusted quantity based on box size
    """
    box_qty = row.get('Box', 1)
    
    # Handle missing or invalid box quantities
    if pd.isna(box_qty) or box_qty <= 0:
        return base_qty
    
    try:
        box_qty = float(box_qty)
    except (ValueError, TypeError):
        return base_qty
    
    # Round to nearest multiple of box quantity
    adjusted_qty = round(base_qty / box_qty) * box_qty
    
    # Check if adjustment is within tolerance
    difference = abs(base_qty - adjusted_qty)
    if difference <= tolerance:
        return adjusted_qty
    else:
        # If difference exceeds tolerance, return original quantity
        return base_qty


def _apply_scheme_adjustment_new(predicted_order_qty: float, row: pd.Series, df: pd.DataFrame, tolerance: int) -> str:
    """
    Apply scheme adjustment to predicted order quantity ensuring whole number results.
    
    Rule: If predicted order quantity is within ±tolerance of a scheme total,
    adjust to that scheme format. The scheme result must always sum to a whole number.
    
    Args:
        predicted_order_qty: Quantity after box adjustment
        row: Current row data
        df: Full DataFrame for context
        tolerance: Tolerance for scheme adjustments (±tolerance)
    
    Returns:
        Formatted order string with scheme pattern (ensuring whole number sum)
    """
    # Check for Scm column
    scm_col = None
    for col in ['Scm', 'Scm.', 'scm', 'scm.']:
        if col in df.columns:
            scm_col = col
            break
    
    if not scm_col or pd.isna(row[scm_col]):
        # No scheme available, return as simple number
        if predicted_order_qty == int(predicted_order_qty):
            return str(int(predicted_order_qty))
        else:
            return f"{predicted_order_qty:.1f}"
    
    scm_value = str(row[scm_col]).strip()
    
    # Handle no scheme cases
    if scm_value in ['0', '0+0', '', 'nan', 'null', 'none']:
        if predicted_order_qty == int(predicted_order_qty):
            return str(int(predicted_order_qty))
        else:
            return f"{predicted_order_qty:.1f}"
    
    # Parse scheme pattern (e.g., "5+1", "9+1")
    if '+' not in scm_value:
        # Not a scheme pattern, return as simple number
        if predicted_order_qty == int(predicted_order_qty):
            return str(int(predicted_order_qty))
        else:
            return f"{predicted_order_qty:.1f}"
    
    try:
        scheme_parts = scm_value.split('+')
        if len(scheme_parts) != 2:
            raise ValueError("Invalid scheme format")
        
        scheme_base = float(scheme_parts[0])
        scheme_bonus = float(scheme_parts[1])
        
        # If either part is 0, treat as no scheme
        if scheme_base == 0 or scheme_bonus == 0:
            if predicted_order_qty == int(predicted_order_qty):
                return str(int(predicted_order_qty))
            else:
                return f"{predicted_order_qty:.1f}"
        
        # Calculate original scheme total
        original_scheme_total = scheme_base + scheme_bonus
        
        # Check if predicted order quantity is within tolerance of the scheme total
        difference = abs(predicted_order_qty - original_scheme_total)
        
        if difference <= tolerance:
            # Use the original scheme
            base_str = str(int(scheme_base)) if scheme_base == int(scheme_base) else f"{scheme_base:.1f}"
            bonus_str = str(int(scheme_bonus)) if scheme_bonus == int(scheme_bonus) else f"{scheme_bonus:.1f}"
            return f"{base_str}+{bonus_str}"
        
        # Try to find a proportional scheme that results in a whole number
        # and is within tolerance of predicted_order_qty
        
        # Test different multipliers to find one that gives a whole number total
        # and is close to predicted_order_qty
        best_scheme = None
        best_difference = float('inf')
        
        # Test multipliers from 0.1 to 5.0 in increments of 0.1
        for multiplier in [i/10.0 for i in range(1, 51)]:
            test_base = scheme_base * multiplier
            test_bonus = scheme_bonus * multiplier
            test_total = test_base + test_bonus
            
            # Check if total is a whole number (or very close to one)
            if abs(test_total - round(test_total)) < 0.001:
                test_total_rounded = round(test_total)
                diff = abs(predicted_order_qty - test_total_rounded)
                
                if diff <= tolerance and diff < best_difference:
                    best_difference = diff
                    # Adjust base and bonus to ensure they sum to the whole number
                    # Keep the original ratio but adjust to sum to whole number
                    ratio = scheme_base / (scheme_base + scheme_bonus)
                    adjusted_base = test_total_rounded * ratio
                    adjusted_bonus = test_total_rounded * (1 - ratio)
                    
                    # Round to reasonable precision
                    adjusted_base = round(adjusted_base * 2) / 2  # Round to nearest 0.5
                    adjusted_bonus = round(adjusted_bonus * 2) / 2  # Round to nearest 0.5
                    
                    # Ensure they still sum to whole number
                    if abs((adjusted_base + adjusted_bonus) - test_total_rounded) > 0.1:
                        # Adjust bonus to make sum exact
                        adjusted_bonus = test_total_rounded - adjusted_base
                    
                    best_scheme = (adjusted_base, adjusted_bonus)
        
        if best_scheme:
            base_val, bonus_val = best_scheme
            base_str = str(int(base_val)) if base_val == int(base_val) else f"{base_val:.1f}"
            bonus_str = str(int(bonus_val)) if bonus_val == int(bonus_val) else f"{bonus_val:.1f}"
            return f"{base_str}+{bonus_str}"
        
        # If no suitable scheme found, return as simple number
        if predicted_order_qty == int(predicted_order_qty):
            return str(int(predicted_order_qty))
        else:
            return f"{predicted_order_qty:.1f}"
        
    except (ValueError, TypeError, IndexError):
        # Fallback to simple number if scheme parsing fails
        if predicted_order_qty == int(predicted_order_qty):
            return str(int(predicted_order_qty))
        else:
            return f"{predicted_order_qty:.1f}"


def _apply_scheme_adjustment(adjusted_qty: float, row: pd.Series, df: pd.DataFrame, tolerance: int) -> str:
    """
    Apply scheme adjustment to quantity with intelligent multiplier selection.
    
    Args:
        adjusted_qty: Quantity after box adjustment
        row: Current row data
        df: Full DataFrame for context
        tolerance: Tolerance for scheme adjustments (±tolerance)
    
    Returns:
        Formatted order string with scheme pattern
    """
    # Check for Scm column
    scm_col = None
    for col in ['Scm', 'Scm.', 'scm', 'scm.']:
        if col in df.columns:
            scm_col = col
            break
    
    if not scm_col or pd.isna(row[scm_col]):
        # No scheme available, return as simple number
        if adjusted_qty == int(adjusted_qty):
            return str(int(adjusted_qty))
        else:
            return f"{adjusted_qty:.1f}"
    
    scm_value = str(row[scm_col]).strip()
    
    # Handle no scheme cases
    if scm_value in ['0', '0+0', '', 'nan', 'null', 'none']:
        if adjusted_qty == int(adjusted_qty):
            return str(int(adjusted_qty))
        else:
            return f"{adjusted_qty:.1f}"
    
    # Parse scheme pattern (e.g., "5+1", "3+1")
    if '+' not in scm_value:
        # Not a scheme pattern, return as simple number
        if adjusted_qty == int(adjusted_qty):
            return str(int(adjusted_qty))
        else:
            return f"{adjusted_qty:.1f}"
    
    try:
        scheme_parts = scm_value.split('+')
        if len(scheme_parts) != 2:
            raise ValueError("Invalid scheme format")
        
        scheme_base = float(scheme_parts[0])
        scheme_bonus = float(scheme_parts[1])
        
        # If either part is 0, treat as no scheme
        if scheme_base == 0 or scheme_bonus == 0:
            if adjusted_qty == int(adjusted_qty):
                return str(int(adjusted_qty))
            else:
                return f"{adjusted_qty:.1f}"
        
        # Find optimal scheme multiplier
        total_scheme = scheme_base + scheme_bonus
        
        # For very small quantities, return as-is
        if adjusted_qty <= 1:
            if adjusted_qty == int(adjusted_qty):
                return str(int(adjusted_qty))
            else:
                return f"{adjusted_qty:.1f}"
        
        # Test valid multipliers: 0.5x, 1x, 2x, 3x, 4x, 5x
        valid_multipliers = [0.5, 1, 2, 3, 4, 5]
        best_multiplier = 1
        best_difference = float('inf')
        
        # Enhanced logic considering demand trends from memories
        demand_trend_factor = _calculate_demand_trend_factor(row, df)
        
        for multiplier in valid_multipliers:
            test_total = total_scheme * multiplier
            difference = abs(adjusted_qty - test_total)
            
            # Apply trend-based scoring
            if demand_trend_factor > 1.2 and test_total > adjusted_qty:
                # Prefer higher quantities for increasing demand
                adjusted_difference = difference * 0.7
            elif demand_trend_factor < 0.8 and test_total < adjusted_qty:
                # Prefer lower quantities for decreasing demand
                adjusted_difference = difference * 0.7
            else:
                adjusted_difference = difference
            
            # Check if within tolerance
            if adjusted_difference <= tolerance and adjusted_difference < best_difference:
                best_difference = adjusted_difference
                best_multiplier = multiplier
        
        # Apply the best multiplier
        new_base = _round_to_half(scheme_base * best_multiplier)
        new_bonus = _round_to_half(scheme_bonus * best_multiplier)
        
        # Format values
        base_str = str(int(new_base)) if new_base == int(new_base) else f"{new_base:.1f}"
        bonus_str = str(int(new_bonus)) if new_bonus == int(new_bonus) else f"{new_bonus:.1f}"
        
        return f"{base_str}+{bonus_str}"
        
    except (ValueError, TypeError, IndexError):
        # Fallback to simple number if scheme parsing fails
        if adjusted_qty == int(adjusted_qty):
            return str(int(adjusted_qty))
        else:
            return f"{adjusted_qty:.1f}"


def _calculate_demand_trend_factor(row: pd.Series, df: pd.DataFrame) -> float:
    """
    Calculate demand trend factor based on sales history.
    
    Args:
        row: Current row data
        df: Full DataFrame for context
    
    Returns:
        Trend factor (>1.0 for increasing, <1.0 for decreasing, 1.0 for stable)
    """
    sales_cols = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
    available_sales = [col for col in sales_cols if col in df.columns]
    
    if len(available_sales) < 3:
        return 1.0  # Neutral if insufficient data
    
    # Get sales values
    sales_values = []
    for col in available_sales:
        val = row.get(col, 0)
        if pd.notna(val):
            try:
                sales_values.append(float(val))
            except (ValueError, TypeError):
                sales_values.append(0)
        else:
            sales_values.append(0)
    
    if len(sales_values) < 3:
        return 1.0
    
    # Enhanced trend analysis from memory
    recent_avg = sum(sales_values[:2]) / 2 if len(sales_values) >= 2 else sales_values[0]
    older_avg = sum(sales_values[-2:]) / 2 if len(sales_values) >= 2 else sales_values[-1]
    
    # Check overall growth pattern
    overall_growth = sales_values[0] / sales_values[-1] if sales_values[-1] > 0 else 1
    
    # Check for consistent growth
    growth_consistency = 0
    for i in range(len(sales_values) - 1):
        if sales_values[i] > sales_values[i + 1]:
            growth_consistency += 1
    
    consistency_ratio = growth_consistency / (len(sales_values) - 1) if len(sales_values) > 1 else 0
    
    if older_avg > 0:
        trend_ratio = recent_avg / older_avg
        
        # Enhanced trend factor calculation based on memory
        if trend_ratio < 0.7:  # Decreasing trend
            return 0.8
        elif trend_ratio > 2.0 and consistency_ratio > 0.6:  # Strong consistent growth
            return 1.8  # Very aggressive for strong trends
        elif trend_ratio > 1.5 and consistency_ratio > 0.5:  # Good growth
            return 1.5  # More aggressive
        elif trend_ratio > 1.3:  # Moderate growth
            return 1.2  # Standard increase
    
    return 1.0  # Neutral


def _round_to_half(value: float) -> float:
    """
    Round value to nearest 0.5.
    
    Args:
        value: Value to round
    
    Returns:
        Rounded value
    """
    return round(value * 2) / 2
