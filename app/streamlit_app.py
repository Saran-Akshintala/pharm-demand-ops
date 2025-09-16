"""
Streamlit app for pharmacy demand forecasting.
Upload Excel files, get predictions, and download results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
from io import BytesIO

# Add current directory to path for imports
sys.path.append('.')
from utils import (
    preprocess_sales_features, 
    prepare_features_for_prediction,
    load_model,
    parse_order_scheme,
    reconstruct_order_prediction,
    validate_input_data,
    apply_predicted_order_business_rules
)

# Load custom CSS (deprecated) - default Streamlit styling only
def load_css():
    """Deprecated: No custom CSS injection. Using Streamlit defaults."""
    pass

def inject_pandas_table_css():
    """Deprecated: No CSS injection. Using Streamlit defaults."""
    pass

def render_dataframe(df: pd.DataFrame):
    """Render a dataframe using Streamlit's default styling and viewer."""
    st.dataframe(df, use_container_width=True)

# Page configuration
st.set_page_config(
    page_title="Pharmacy Demand Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default Streamlit styling only (no CSS injection)

def process_uploaded_file(uploaded_file):
    """Process uploaded Excel file and return predictions"""
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        # Validate input data
        validation_results = validate_input_data(df)
        
        if validation_results['errors']:
            st.error("❌ Data validation failed:")
            for error in validation_results['errors']:
                st.error(f"• {error}")
            return None, None, None
        
        if validation_results['warnings']:
            st.warning("⚠️ Data validation warnings:")
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
        
        # Load model and make predictions
        script_dir = Path(__file__).parent.absolute()
        
        # Always use absolute path from script location
        model_path = script_dir.parent / "models" / "order_predictor.pkl"
        
        
        if not model_path.exists():
            st.error("❌ Model not found. Please train the model first.")
            return None, None, None
        
        model = load_model(str(model_path))
        
        # Load model info for feature columns
        model_info_path = script_dir.parent / "models" / "model_info.json"
            
        if model_info_path.exists():
            with open(model_info_path) as f:
                model_info = json.load(f)
            feature_columns = model_info['feature_columns']
        else:
            # Fallback feature columns
            feature_columns = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
        
        # Preprocess features
        df_processed = preprocess_sales_features(df)
        
        # Prepare features for prediction
        X = prepare_features_for_prediction(df_processed, feature_columns)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to original dataframe
        df_result = df.copy()
        df_result['Predicted_Base_Quantity'] = predictions.round().astype(int)
        
        # Reconstruct order predictions using Scm column for scheme-based predictions
        predicted_orders = []
        for i, row in df_result.iterrows():
            base_qty = int(row['Predicted_Base_Quantity'])
            
            # Use Scm column to determine order format
            predicted_order = str(base_qty)  # Default fallback
            
            # Check for Scm. column (note the period)
            scm_col = None
            for col in ['Scm', 'Scm.', 'scm', 'scm.']:
                if col in df.columns:
                    scm_col = col
                    break
            
            if scm_col and pd.notna(row[scm_col]):
                scm_value = str(row[scm_col]).strip()
                
                # Debug: Check what we're getting
                print(f"Debug: Scm value = '{scm_value}', Base qty = {base_qty}")
                
                # Check for no scheme cases - use simple base quantity
                if scm_value in ['0', '0+0', '', 'nan', 'null', 'none']:
                    predicted_order = str(base_qty)
                    print(f"Debug: No scheme case, order = {predicted_order}")
                elif '+' in scm_value:
                    # Parse scheme pattern (e.g., "2+1", "5+1")
                    try:
                        scheme_parts = scm_value.split('+')
                        if len(scheme_parts) == 2:
                            scheme_base = float(scheme_parts[0])
                            scheme_bonus = float(scheme_parts[1])
                            
                            print(f"Debug: Scheme parts - base: {scheme_base}, bonus: {scheme_bonus}")
                            
                            # If either part is 0, treat as no scheme
                            if scheme_base == 0 or scheme_bonus == 0:
                                predicted_order = str(base_qty)
                                print(f"Debug: Zero scheme detected, order = {predicted_order}")
                            else:
                                # Calculate proportional scheme: maintain the ratio
                                total_scheme = scheme_base + scheme_bonus
                                base_ratio = scheme_base / total_scheme
                                bonus_ratio = scheme_bonus / total_scheme
                                
                                new_base = base_qty * base_ratio
                                new_bonus = base_qty * bonus_ratio
                                
                                print(f"Debug: Calculated - new_base: {new_base}, new_bonus: {new_bonus}")
                                
                                # Round to nearest 0.5 for fractional values
                                def round_to_half(value):
                                    return round(value * 2) / 2
                                
                                new_base_rounded = round_to_half(new_base)
                                new_bonus_rounded = round_to_half(new_bonus)
                                
                                print(f"Debug: Rounded - new_base: {new_base_rounded}, new_bonus: {new_bonus_rounded}")
                                
                                # Format as integer if whole numbers, otherwise as decimal
                                if new_base_rounded == int(new_base_rounded) and new_bonus_rounded == int(new_bonus_rounded):
                                    predicted_order = f"{int(new_base_rounded)}+{int(new_bonus_rounded)}"
                                else:
                                    predicted_order = f"{new_base_rounded:.1f}+{new_bonus_rounded:.1f}"
                                
                                print(f"Debug: Final scheme order = {predicted_order}")
                        else:
                            predicted_order = str(base_qty)
                    except Exception as e:
                        print(f"Debug: Exception in scheme parsing: {e}")
                        predicted_order = str(base_qty)
                else:
                    # Single number or other format
                    predicted_order = str(base_qty)
                    print(f"Debug: Single number case, order = {predicted_order}")
            else:
                # No Scm column - fallback to original logic
                print(f"Debug: No Scm column, using fallback")
                order_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in ['order', 'ord', 'oreder']:
                        order_col = col
                        break
                
                if order_col and pd.notna(row[order_col]):
                    try:
                        _, scheme_info = parse_order_scheme(str(row[order_col]))
                        predicted_order = reconstruct_order_prediction(base_qty, scheme_info)
                    except:
                        predicted_order = str(base_qty)
                else:
                    predicted_order = str(base_qty)
            
            predicted_orders.append(predicted_order)
        
        df_result['Predicted_Order'] = predicted_orders
        
        # Apply business rules to Predicted_Order column
        df_result, styling_info = apply_predicted_order_business_rules(df_result)
        
        # Reorder columns to place predictions next to Order column (fix from memory)
        order_col = None
        # Check for various order column name variations
        for col in df_result.columns:
            col_lower = col.lower()
            if col_lower in ['order', 'ord', 'oreder']:
                order_col = col
                break
        
        if order_col:
            # Use df_result.columns (after adding predictions) instead of df.columns
            cols = list(df_result.columns)
            
            # Remove prediction columns from their current positions
            cols = [col for col in cols if col not in ['Predicted_Order', 'Predicted_Base_Quantity']]
            
            # Find the position of the Order column
            order_idx = cols.index(order_col)
            
            # Insert prediction columns right after Order column
            cols.insert(order_idx + 1, 'Predicted_Order')
            cols.insert(order_idx + 2, 'Predicted_Base_Quantity')
            
            # Reorder the dataframe
            df_result = df_result.reindex(columns=cols)
        
        return df_result, df_processed, styling_info
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None, None, None

def create_visualizations(df_result, df_processed):
    """Create visualizations for the predictions"""
    
    # Prediction distribution
    fig_dist = px.histogram(
        df_result, 
        x='Predicted_Base_Quantity',
        title="📊 Distribution of Predicted Order Quantities",
        nbins=30,
        color_discrete_sequence=['#667eea']
    )
    fig_dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top products by predicted quantity
    if len(df_result) > 0:
        top_products = df_result.nlargest(10, 'Predicted_Base_Quantity')[['Name', 'Predicted_Base_Quantity', 'Predicted_Order']]
        
        fig_top = px.bar(
            top_products,
            x='Predicted_Base_Quantity',
            y='Name',
            orientation='h',
            title="🔝 Top 10 Products by Predicted Quantity",
            color='Predicted_Base_Quantity',
            color_continuous_scale='viridis'
        )
        fig_top.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        st.plotly_chart(fig_top, use_container_width=True)

def main():
    """Main application function"""
    
    # App header with modern styling
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 💊 Pharmacy Demand Forecasting")
    st.markdown("### AI-Powered Order Prediction System")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("## 🤖 Model Information")
        
        # Try to load model info
        try:
            script_dir = Path(__file__).parent.absolute()
            model_info_path = script_dir.parent / "models" / "model_info.json"
                
            if model_info_path.exists():
                with open(model_info_path) as f:
                    model_info = json.load(f)
                
                st.markdown("### 📊 Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{model_info['performance']['rmse']:.3f}")
                    st.metric("R²", f"{model_info['performance']['r2']:.3f}")
                with col2:
                    st.metric("MAE", f"{model_info['performance']['mae']:.3f}")
                    st.metric("Features", len(model_info['feature_columns']))
                
                st.markdown("### 🎯 Model Type")
                st.info(f"**{model_info['model_type']}**")
                
                st.markdown("### 📈 Training Data")
                st.write(f"Training samples: {model_info['training_data_size']:,}")
                st.write(f"Test samples: {model_info['test_data_size']:,}")
            else:
                st.warning("Model info not found. Please train the model first.")
        except Exception as e:
            st.error(f"Error loading model info: {e}")
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("## 📁 Upload Pharmacy Data")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your pharmacy order data in Excel format"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        
        # Process the file
        with st.spinner("🔄 Processing your data..."):
            result = process_uploaded_file(uploaded_file)
            if result[0] is not None:
                df_result, df_processed, styling_info = result
            else:
                df_result, df_processed, styling_info = None, None, None
        
        if df_result is not None:
            st.success("✅ Predictions generated successfully!")
            
            # Display summary statistics
            st.markdown("## 📈 Prediction Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Products", len(df_result))
            with col2:
                avg_prediction = df_result['Predicted_Base_Quantity'].mean()
                st.metric("Avg Predicted Qty", f"{avg_prediction:.1f}")
            with col3:
                max_prediction = df_result['Predicted_Base_Quantity'].max()
                st.metric("Max Predicted Qty", f"{max_prediction}")
            with col4:
                total_prediction = df_result['Predicted_Base_Quantity'].sum()
                st.metric("Total Predicted Qty", f"{total_prediction:,}")
            
            # Add section divider
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("## 📊 Prediction Analytics")
            create_visualizations(df_result, df_processed)
            
            # Add section divider
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Preview table with business rule styling
            st.markdown("## 📋 Detailed Results (All Rows)")
            
            # Apply styling based on business rules
            def apply_business_rule_styling(df, styling_info):
                styler = df.style
                
                # Apply business rule colors to Predicted_Order column
                if 'Predicted_Order' in df.columns:
                    for row_idx, style_info in styling_info.items():
                        if row_idx < len(df):
                            # Only apply background color if color is not None (skip "No Order" cases)
                            if style_info.get('color') is not None:
                                styler = styler.set_properties(
                                    subset=pd.IndexSlice[row_idx, 'Predicted_Order'],
                                    **{"background-color": style_info['color'], "color": "black"}
                                )
                
                # Highlight key columns with light background
                cols_lower_map = {c.lower(): c for c in df.columns}
                target_cols = ["name", "order", "predicted_order", "predicted_base_quantity"]
                highlight_cols = [cols_lower_map[c] for c in target_cols if c in cols_lower_map]
                
                if highlight_cols:
                    # Apply light background to key columns (but don't override business rule colors)
                    for col in highlight_cols:
                        if col != 'Predicted_Order':  # Don't override business rule styling
                            styler = styler.set_properties(
                                subset=[col],
                                **{"background-color": "#fce5cd", "color": "black"}
                            )
                
                return styler
            
            if styling_info:
                styled_df = apply_business_rule_styling(df_result, styling_info)
                st.dataframe(styled_df, use_container_width=True)
                
                # Display tooltip information as expandable section
                with st.expander("📝 View Highlighting Reasons", expanded=False):
                    st.markdown("**Rows with business rule highlighting:**")
                    tooltip_data = []
                    for row_idx, style_info in styling_info.items():
                        if row_idx < len(df_result):
                            product_name = df_result.iloc[row_idx].get('Name', f'Row {row_idx + 1}')
                            predicted_order = df_result.iloc[row_idx].get('Predicted_Order', 'N/A')
                            tooltip_data.append({
                                'Product': str(product_name)[:50] + '...' if len(str(product_name)) > 50 else str(product_name),
                                'Predicted_Order': predicted_order,
                                'Reason': style_info['tooltip']
                            })
                    
                    if tooltip_data:
                        tooltip_df = pd.DataFrame(tooltip_data)
                        st.dataframe(tooltip_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No business rules triggered for this dataset.")
                
                # Add legend for colors
                st.markdown("### 🎨 Color Legend")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div style="background-color: #ffcccc; padding: 5px; border-radius: 3px; text-align: center; color: black;">🔴 Reddish: Days > 90 / Uneven Sales</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div style="background-color: #ccffcc; padding: 5px; border-radius: 3px; text-align: center; color: black;">🟢 Greenish: Negative after Stock</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div style="background-color: #ffe6cc; padding: 5px; border-radius: 3px; text-align: center; color: black;">🟠 Orangish: Box Adjustment ±2</div>', unsafe_allow_html=True)
                with col4:
                    st.markdown('<div style="background-color: #ffffcc; padding: 5px; border-radius: 3px; text-align: center; color: black;">🟡 Yellowish: Low Customers ≤2</div>', unsafe_allow_html=True)
            else:
                st.dataframe(df_result, use_container_width=True)

            # Download button
            st.markdown("### 💾 Download Results")
            
            # Convert to Excel for download with styling
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Predictions')
                
                # Apply styling to Excel file
                if styling_info:
                    from openpyxl.styles import PatternFill
                    from openpyxl.comments import Comment
                    
                    worksheet = writer.sheets['Predictions']
                    
                    # Find Predicted_Order column
                    predicted_order_col = None
                    for col_idx, col_name in enumerate(df_result.columns, 1):
                        if col_name == 'Predicted_Order':
                            predicted_order_col = col_idx
                            break
                    
                    if predicted_order_col:
                        # Apply colors and comments
                        for row_idx, style_info in styling_info.items():
                            excel_row = row_idx + 2  # +2 because Excel is 1-indexed and has header
                            cell = worksheet.cell(row=excel_row, column=predicted_order_col)
                            
                            # Only apply background color if color is not None (skip "No Order" cases)
                            if style_info.get('color') is not None:
                                color_hex = style_info['color'].replace('#', '')
                                fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type='solid')
                                cell.fill = fill
                            
                            # Always add comment with tooltip (including "No Order" explanations)
                            comment = Comment(style_info['tooltip'], 'Business Rules')
                            cell.comment = comment
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Excel with Predictions",
                data=excel_data,
                file_name=f"pharmacy_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
        <p>🚀 Powered by Advanced Machine Learning | Built with Streamlit</p>
        <p>📊 Enhanced with XGBoost, Random Forest & Feature Engineering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
