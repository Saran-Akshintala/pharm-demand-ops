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
from typing import Tuple, Optional
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
