"""
Enhanced grid implementation using st-aggrid with color highlighting and tooltips.
This provides a true Excel preview experience with advanced formatting.
"""

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
from st_aggrid.shared import GridUpdateMode, DataReturnMode

def create_enhanced_grid(df_data, styling_info=None, expiry_styling=None, original_data=None, key="enhanced_grid"):
    """
    Create an enhanced editable grid with color highlighting and tooltips.
    
    Args:
        df_data: DataFrame to display
        styling_info: Dictionary with row styling information for Predicted_Order
        expiry_styling: Dictionary with row styling information for Expiry columns
        original_data: Original DataFrame to track changes
        key: Unique key for the grid component
    
    Returns:
        Dictionary with grid response data
    """
    
    # Prepare data with tooltip columns
    grid_data = df_data.copy()
    
    # Add original values for change tracking using product key mapping
    if original_data is not None and 'Predicted_Order' in original_data.columns:
        # Create mapping of original values using product keys
        original_values_map = {}
        for idx, row in original_data.iterrows():
            # Create product key
            key_fields = []
            if 'Name' in row:
                key_fields.append(str(row['Name']))
            if 'Supplier' in row:
                key_fields.append(str(row['Supplier']))
            if 'Stock' in row:
                key_fields.append(str(row['Stock']))
            
            if key_fields:
                product_key = '|'.join(key_fields)
                original_values_map[product_key] = row['Predicted_Order']
        
        # Map original values to current filtered data using product keys
        original_values = []
        for idx, row in grid_data.iterrows():
            # Create product key for current row
            key_fields = []
            if 'Name' in row:
                key_fields.append(str(row['Name']))
            if 'Supplier' in row:
                key_fields.append(str(row['Supplier']))
            if 'Stock' in row:
                key_fields.append(str(row['Stock']))
            
            if key_fields:
                product_key = '|'.join(key_fields)
                original_val = original_values_map.get(product_key, row['Predicted_Order'])
            else:
                original_val = row['Predicted_Order']  # Fallback to current value
            
            original_values.append(original_val)
        
        grid_data['_original_Predicted_Order'] = original_values
    
    # Add tooltip column for Predicted_Order
    if 'Predicted_Order' in grid_data.columns:
        tooltip_col = []
        for idx, row in grid_data.iterrows():
            tooltip = ""
            if styling_info and idx in styling_info:
                tooltip = styling_info[idx].get('tooltip', '')
            tooltip_col.append(tooltip)
        
        grid_data['Predicted_Order_Tooltip'] = tooltip_col
    
    # Add tooltip column for Expiry if exists
    expiry_col = None
    for col in grid_data.columns:
        if 'expiry' in col.lower():
            expiry_col = col
            break
    
    if expiry_col:
        expiry_tooltip_col = []
        for idx, row in grid_data.iterrows():
            tooltip = ""
            if expiry_styling and idx in expiry_styling:
                tooltip = expiry_styling[idx].get('tooltip', '')
            expiry_tooltip_col.append(tooltip)
        
        grid_data[f'{expiry_col}_Tooltip'] = expiry_tooltip_col
    
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(grid_data)
    
    # Configure Predicted_Order column
    if 'Predicted_Order' in grid_data.columns:
        gb.configure_column(
            "Predicted_Order",
            editable=True,
            tooltipField="Predicted_Order_Tooltip",
            cellStyle=create_cell_style_js(styling_info, is_predicted_order=True),
            width=150,
            headerTooltip="Edit predicted order values. Hover for business rule explanations."
        )
    
    # Configure Expiry column if exists
    if expiry_col:
        gb.configure_column(
            expiry_col,
            editable=False,
            tooltipField=f"{expiry_col}_Tooltip",
            cellStyle=create_cell_style_js(expiry_styling, is_predicted_order=False),
            width=120,
            headerTooltip="Expiry information with color-coded warnings"
        )
    
    # Hide tooltip columns and original value column
    gb.configure_column("Predicted_Order_Tooltip", hide=True)
    if expiry_col:
        gb.configure_column(f"{expiry_col}_Tooltip", hide=True)
    if '_original_Predicted_Order' in grid_data.columns:
        gb.configure_column("_original_Predicted_Order", hide=True)
    
    # Configure other columns as read-only
    for col in grid_data.columns:
        if col not in ['Predicted_Order', 'Predicted_Order_Tooltip', f'{expiry_col}_Tooltip'] and col != expiry_col:
            gb.configure_column(col, editable=False)
    
    # General grid configuration
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        width=100
    )
    
    gb.configure_grid_options(
        enableRangeSelection=True,
        enableCellTextSelection=True,
        tooltipShowDelay=500,
        tooltipHideDelay=2000
    )
    
    # Build grid options
    grid_options = gb.build()
    
    # Display the grid
    grid_response = AgGrid(
        grid_data,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        height=400,
        width='100%',
        reload_data=False,
        allow_unsafe_jscode=True,  # Required for JsCode objects
        key=key
    )
    
    return grid_response

def create_cell_style_js(styling_info, is_predicted_order=True):
    """
    Create JavaScript function for cell styling based on business rules.
    """
    if not styling_info:
        return None
    
    # Create color mapping for rows
    color_map = {}
    for row_idx, style_info in styling_info.items():
        if style_info.get('color'):
            color_map[row_idx] = style_info['color']
    
    if not color_map:
        return None
    
    # Create JavaScript function with change detection for Predicted_Order
    if is_predicted_order:
        js_code = f"""
        function(params) {{
            const rowIndex = params.node.rowIndex;
            const colorMap = {color_map};
            const currentValue = params.value;
            const originalValue = params.data._original_Predicted_Order;
            
            // Check if value has been changed
            const isChanged = originalValue !== undefined && currentValue !== originalValue;
            
            // Priority: Change highlighting > Business rule colors
            if (isChanged) {{
                return {{
                    'background-color': '#d4edda',  // Light green for changes
                    'color': 'black',
                    'font-weight': 'bold',
                    'border': '2px solid #28a745'
                }};
            }} else if (colorMap[rowIndex]) {{
                return {{
                    'background-color': colorMap[rowIndex],
                    'color': 'black',
                    'font-weight': 'bold'
                }};
            }}
            return null;
        }}
        """
    else:
        # For other columns (like Expiry), just apply business rule colors
        js_code = f"""
        function(params) {{
            const rowIndex = params.node.rowIndex;
            const colorMap = {color_map};
            
            if (colorMap[rowIndex]) {{
                return {{
                    'background-color': colorMap[rowIndex],
                    'color': 'black',
                    'font-weight': 'bold'
                }};
            }}
            return null;
        }}
        """
    
    return JsCode(js_code)

def create_expiry_cell_style_js(expiry_styling):
    """
    Create JavaScript function for expiry cell styling.
    """
    if not expiry_styling:
        return None
    
    # Create color mapping for rows
    color_map = {}
    for row_idx, style_info in expiry_styling.items():
        if style_info.get('color'):
            color_map[row_idx] = style_info['color']
    
    if not color_map:
        return None
    
    # Create JavaScript function
    js_code = f"""
    function(params) {{
        const rowIndex = params.node.rowIndex;
        const colorMap = {color_map};
        
        if (colorMap[rowIndex]) {{
            return {{
                'background-color': colorMap[rowIndex],
                'color': 'black',
                'font-weight': 'bold'
            }};
        }}
        return null;
    }}
    """
    
    return JsCode(js_code)

def process_grid_changes(original_df, grid_response, product_key_columns=['Name', 'Supplier', 'Stock']):
    """
    Process changes from the enhanced grid and return updated DataFrame.
    
    Args:
        original_df: Original DataFrame
        grid_response: Response from AgGrid
        product_key_columns: Columns to use for product identification
    
    Returns:
        Updated DataFrame with changes applied
    """
    if not grid_response or 'data' not in grid_response:
        return original_df
    
    # Get the updated data
    updated_data = pd.DataFrame(grid_response['data'])
    
    # Remove tooltip columns and internal columns
    internal_cols = [col for col in updated_data.columns if col.endswith('_Tooltip') or col.startswith('_original_')]
    updated_data = updated_data.drop(columns=internal_cols, errors='ignore')
    
    # Create a copy of original data to modify
    result_df = original_df.copy()
    
    # Apply changes using product key mapping (same logic as before)
    try:
        # Create mapping of changes
        changes_map = {}
        for idx, row in updated_data.iterrows():
            # Create product key
            key_fields = []
            for col in product_key_columns:
                if col in row:
                    key_fields.append(str(row[col]))
            
            if key_fields:
                product_key = '|'.join(key_fields)
                if 'Predicted_Order' in row:
                    changes_map[product_key] = row['Predicted_Order']
        
        # Apply changes to result DataFrame
        for idx, row in result_df.iterrows():
            key_fields = []
            for col in product_key_columns:
                if col in row:
                    key_fields.append(str(row[col]))
            
            if key_fields:
                product_key = '|'.join(key_fields)
                if product_key in changes_map:
                    result_df.iloc[idx, result_df.columns.get_loc('Predicted_Order')] = changes_map[product_key]
    
    except Exception as e:
        st.warning(f"Error processing grid changes: {str(e)}")
        return original_df
    
    return result_df

def show_grid_changes_summary(original_df, updated_df):
    """
    Show a summary of changes made in the grid.
    """
    changes = []
    
    if len(original_df) != len(updated_df):
        st.warning("DataFrame structure changed - cannot compare changes")
        return
    
    for idx in range(len(original_df)):
        if idx < len(updated_df):
            original_val = original_df.iloc[idx].get('Predicted_Order', '')
            updated_val = updated_df.iloc[idx].get('Predicted_Order', '')
            
            if str(original_val) != str(updated_val):
                product_name = original_df.iloc[idx].get('Name', f'Row {idx+1}')
                changes.append({
                    'Product': product_name,
                    'Original': original_val,
                    'Updated': updated_val
                })
    
    if changes:
        st.success(f"✅ {len(changes)} changes detected! These will be included in the Excel export.")
        
        with st.expander("📝 View Changes", expanded=False):
            for change in changes:
                st.write(f"**{change['Product']}**: {change['Original']} → {change['Updated']}")
    
    return len(changes) > 0
