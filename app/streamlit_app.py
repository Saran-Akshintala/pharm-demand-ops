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
    apply_predicted_order_business_rules,
    compute_predicted_order_with_adjustments
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

def process_uploaded_file(uploaded_file, apply_box=True, box_tolerance=2, apply_scm=True, scm_tolerance=2):
    """Process uploaded Excel file and return predictions with configurable adjustments"""
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
        
        # Use new deterministic logic to compute Predicted_Order from Predicted_Base_Quantity
        df_result = compute_predicted_order_with_adjustments(
            df_result, 
            apply_box=apply_box, 
            box_tolerance=box_tolerance,
            apply_scm=apply_scm, 
            scm_tolerance=scm_tolerance
        )
        
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
            
            # Add UI controls for box and scheme adjustments
            st.markdown("### ⚙️ Order Adjustment Settings")
            
            # Add explanation of priority rules
            with st.expander("📝 Priority Rules Explanation", expanded=False):
                st.markdown("""
                **Predicted_Order Computation Priority:**
                
                1. **Stock Subtraction**: `Predicted_Order = Predicted_Base_Quantity - Stock`
                2. **Box Adjustment**: If result is within ±tolerance of Box quantity, adjust to Box quantity
                3. **Scheme Adjustment**: If result is within ±tolerance of scheme total, apply scheme format
                
                **Examples:**
                - Base=14, Stock=4 → 14-4=10
                - If Box=10 and tolerance=2: 10 is exactly Box, so stays 10
                - If Scm=5+1 (total=6) and tolerance=2: |10-6|=4 > 2, so stays 10
                
                **Note**: Scheme results always sum to whole numbers (e.g., 4.5+0.5=5, not 4+0.5=4.5)
                """)
            
            # Create columns for compact layout
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 1])
            
            with col1:
                apply_box_adj = st.checkbox("Apply Box Quantity", value=True, key="apply_box_checkbox")
            
            with col2:
                box_tolerance_val = st.number_input("Box tolerance", min_value=0, max_value=10, value=2, step=1, key="box_tolerance_input")
            
            with col3:
                apply_scm_adj = st.checkbox("Apply Scm", value=True, key="apply_scm_checkbox")
            
            with col4:
                scm_tolerance_val = st.number_input("Scm tolerance", min_value=0, max_value=10, value=2, step=1, key="scm_tolerance_input")
            
            with col5:
                generate_button = st.button("Generate", type="primary", key="generate_button", help="Regenerate Predicted_Order with current settings")
            
            # Initialize session state for settings
            if 'current_settings' not in st.session_state:
                st.session_state.current_settings = {
                    'apply_box': True,
                    'box_tolerance': 2,
                    'apply_scm': True,
                    'scm_tolerance': 2
                }
            
            # Check if settings changed or Generate button clicked
            settings_changed = (
                st.session_state.current_settings['apply_box'] != apply_box_adj or
                st.session_state.current_settings['box_tolerance'] != box_tolerance_val or
                st.session_state.current_settings['apply_scm'] != apply_scm_adj or
                st.session_state.current_settings['scm_tolerance'] != scm_tolerance_val
            )
            
            # Regenerate if Generate button clicked or if this is initial load
            if generate_button or settings_changed or 'df_result_adjusted' not in st.session_state:
                # Update current settings
                st.session_state.current_settings = {
                    'apply_box': apply_box_adj,
                    'box_tolerance': box_tolerance_val,
                    'apply_scm': apply_scm_adj,
                    'scm_tolerance': scm_tolerance_val
                }
                
                # Reprocess with new settings
                with st.spinner("🔄 Regenerating predictions with new settings..."):
                    result_adjusted = process_uploaded_file(
                        uploaded_file, 
                        apply_box=apply_box_adj, 
                        box_tolerance=box_tolerance_val,
                        apply_scm=apply_scm_adj, 
                        scm_tolerance=scm_tolerance_val
                    )
                    if result_adjusted[0] is not None:
                        st.session_state.df_result_adjusted, st.session_state.df_processed_adjusted, st.session_state.styling_info_adjusted = result_adjusted
                        st.success("✅ Predictions regenerated successfully!")
                    else:
                        st.error("❌ Error regenerating predictions")
                        st.session_state.df_result_adjusted, st.session_state.df_processed_adjusted, st.session_state.styling_info_adjusted = None, None, None
            
            # Use adjusted results if available, otherwise fall back to original
            if 'df_result_adjusted' in st.session_state and st.session_state.df_result_adjusted is not None:
                df_result = st.session_state.df_result_adjusted
                df_processed = st.session_state.df_processed_adjusted
                styling_info = st.session_state.styling_info_adjusted
            
            st.markdown("---")  # Add separator line
            
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
