# 💊 Pharmacy Demand Forecasting System

A comprehensive machine learning solution for predicting pharmacy order quantities based on historical sales data. This system provides end-to-end workflow from data analysis to deployment with both web UI and API interfaces.

## 🎯 Project Overview

This project helps pharmacists predict optimal order quantities by analyzing historical sales patterns and generating intelligent recommendations. The system handles complex order schemes (like "9+1" promotions) and provides confidence scores for predictions.

### Key Features

- **📊 Comprehensive EDA**: Analyze sales patterns, trends, and correlations
- **🤖 ML Pipeline**: Train and evaluate multiple models (Linear Regression, Random Forest, XGBoost)
- **🌐 Enhanced Web Interface**: Advanced Streamlit app with interactive grid features
- **📱 Interactive Data Grid**: Professional st-aggrid integration with Excel preview
- **🎨 Visual Feedback**: Color-coded business rules and change tracking
- **🔍 Advanced Filtering**: Real-time filtering with "Ignore No Order" and supplier exclusion
- **✏️ In-Grid Editing**: Direct editing of predicted values with persistence
- **🔌 REST API**: FastAPI service for enterprise integration
- **📈 Scheme Handling**: Parse and predict promotional schemes (e.g., "9+1", "12+2")
- **📋 Excel Integration**: Direct Excel file processing and output with full styling

## 📂 Project Structure

```
pharm-demand-ops/
├── app/                          # Application code
│   ├── streamlit_app.py         # Enhanced Streamlit web interface
│   ├── enhanced_grid.py         # Advanced st-aggrid implementation
│   ├── api.py                   # FastAPI REST service
│   └── utils.py                 # Utility functions
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_model_training.ipynb  # Model training and evaluation
├── data/                        # Data directory (gitignored)
│   └── orders-data/            # Raw Excel files
├── models/                      # Trained models (gitignored)
│   ├── order_predictor.pkl     # Best trained model
│   ├── scaler.pkl              # Feature scaler (if needed)
│   └── model_info.json         # Model metadata
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🚀 Quick Start

### TL;DR (Minimal Setup)

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py

# OR run API
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Saran-Akshintala/pharm-demand-ops.git
cd pharm-demand-ops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your Excel files in the `data/orders-data/` directory. Expected format:

| Product_Code | Product_Name | L7 | L15 | L30 | L60 
|--------------|--------------|----|----|----|----|
| P001         | Medicine A   | 15 | 32 | 68 | 125|
| P002         | Medicine B   | 8  | 18 | 35 | 72 |

**Column Descriptions:**
- `L7`, `L15`, `L30`, `L60`: Sales in last 7, 15, 30, 60 days

### 3. Run Analysis and Training

```bash
# Start Jupyter notebook server
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_eda.ipynb - Exploratory Data Analysis
# 2. notebooks/02_model_training.ipynb - Model Training
```

### 4. Launch Applications

#### Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
Access at: http://localhost:8501

#### FastAPI Service
```bash
uvicorn app.api:app --reload
```
Access at: http://localhost:8000
API Documentation: http://localhost:8000/docs

## 📊 Usage Examples

### Enhanced Web Interface (Streamlit)

1. **Upload Excel File**: Upload your pharmacy data with sales history
2. **Review Data**: Check the uploaded data preview and validation warnings
3. **Interactive Grid**: View predictions in professional Excel-preview grid with:
   - **Color-coded Business Rules**: Visual indicators for different scenarios
   - **Interactive Tooltips**: Hover over cells for detailed explanations
   - **In-Grid Editing**: Edit Predicted_Order values directly in the grid
   - **Change Tracking**: Green highlighting shows your modifications
4. **Advanced Filtering**: Use filters to focus on specific data:
   - **"Ignore No Order"**: Hide rows with no predicted orders
   - **Supplier Exclusion**: Filter out specific suppliers
5. **Real-time Updates**: All changes and filters update immediately
6. **One-Click Download**: Generate and download Excel with all styling and edits

### API Usage (FastAPI)

#### Single Product Prediction
```python
import requests

# Predict for single product
data = {
    "products": [
        {
            "product_code": "P001",
            "product_name": "Medicine A",
            "L7": 15,
            "L15": 32,
            "L30": 68,
            "L60": 125
        }
    ]
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

#### File Upload Prediction
```python
# Upload Excel file for batch prediction
files = {"file": open("pharmacy_data.xlsx", "rb")}
response = requests.post("http://localhost:8000/predict/file", files=files)
print(response.json())
```

#### cURL Examples
```bash
# JSON payload prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "products": [
          {"product_code": "P001", "product_name": "Medicine A", "L7": 15, "L15": 32, "L30": 68, "L60": 125}
        ]
      }'

# File upload prediction
curl -X POST "http://localhost:8000/predict/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pharmacy_data.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Health check
curl http://localhost:8000/health
```

#### Health Check
```python
response = requests.get("http://localhost:8000/health")
print(response.json())
```

## 🚀 Enhanced Features

### Interactive Data Grid (st-aggrid)

The application features a professional, Excel-like data grid powered by st-aggrid:

#### **Visual Features:**
- **Color-coded Business Rules**: Automatic highlighting based on:
  - 🔴 **Red**: Days > 90 or uneven sales patterns
  - 🟠 **Orange**: Box quantity adjustments (±2)
  - 🟡 **Yellow**: Low customer count (≤2)
  - 🟢 **Green**: User-edited values
  - **Expiry Colors**: Red (≤1 month), Orange (≤3 months), Yellow (≤5 months)

#### **Interactive Features:**
- **Hover Tooltips**: Detailed explanations for business rule decisions
- **In-Grid Editing**: Direct editing of Predicted_Order values
- **Change Tracking**: Visual indicators for modified cells
- **Excel Preview**: True WYSIWYG experience

#### **Advanced Filtering:**
- **"Ignore No Order"**: Hide rows with no predicted orders
- **Supplier Exclusion**: Multi-select dropdown to exclude specific suppliers
- **Real-time Updates**: Immediate grid refresh on filter changes
- **Persistent Edits**: Changes maintained across filter operations

#### **Export Integration:**
- **One-Click Download**: Single button for Excel generation and download
- **Styled Export**: All colors, tooltips, and formatting preserved in Excel
- **Filtered Export**: Only visible data included in download
- **Automatic Server Backup**: Server copy saved automatically

### Business Rule Engine

Comprehensive business logic with visual feedback:

1. **Box Quantity Adjustment**: Rounds to nearest box size with tolerance
2. **Stock Subtraction**: Accounts for current inventory levels
3. **Demand Trend Analysis**: Considers sales velocity and patterns
4. **Expiry Management**: MM/YY format support with tiered warnings
5. **Scheme Logic**: Intelligent promotional scheme handling
6. **Customer Analysis**: Low customer count detection

## 🔧 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**:
   - Handle missing values and inconsistent column names
   - Parse order schemes ("9+1" → base quantity + scheme info)
   - Create derived features (sales trends, volatility)

2. **Feature Engineering**:
   - Sales velocity calculations
   - Trend analysis (L7/L15, L15/L30 ratios)
   - Statistical features (mean, max, volatility)

3. **Model Training**:
   - Compare Linear Regression, Random Forest, XGBoost
   - Cross-validation and hyperparameter tuning
   - Save best model with metadata

4. **Prediction**:
   - Generate base quantity predictions
   - Apply scheme logic based on quantity and patterns
   - Provide confidence scores

### Enhanced Grid Architecture

#### **st-aggrid Integration:**
```python
# Enhanced grid with professional features
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Configure grid with tooltips and styling
gb = GridOptionsBuilder.from_dataframe(data)
gb.configure_column(
    "Predicted_Order",
    editable=True,
    tooltipField="Predicted_Order_Tooltip",
    cellStyle=create_cell_style_js(styling_info)
)

# JavaScript-based cell styling
def create_cell_style_js(styling_info):
    return JsCode(f"""
    function(params) {{
        const colorMap = {styling_info};
        if (colorMap[params.node.rowIndex]) {{
            return {{
                'background-color': colorMap[params.node.rowIndex],
                'color': 'black',
                'font-weight': 'bold'
            }};
        }}
        return null;
    }}
    """)
```

#### **Product Key Mapping:**
```python
# Persistent edit tracking across filter operations
def create_product_key(row):
    return '|'.join([
        str(row['Name']),
        str(row['Supplier']),
        str(row['Stock'])
    ])

# Apply changes back to full dataset
for product_key, new_value in changes_map.items():
    matching_rows = df[df.apply(lambda r: create_product_key(r) == product_key, axis=1)]
    if not matching_rows.empty:
        df.loc[matching_rows.index, 'Predicted_Order'] = new_value
```

### API Endpoints

- `GET /health` - Health check and model status
- `POST /predict` - Predict orders for JSON payload
- `POST /predict/file` - Upload Excel/CSV for batch prediction
- `GET /model/info` - Get model information and performance metrics
- `GET /` - API information and available endpoints

#### **Enhanced API Response Format:**
```json
{
  "predictions": [
    {
      "product_code": "P001",
      "product_name": "Medicine A",
      "predicted_order": "12+2",
      "predicted_base_quantity": 14,
      "confidence_score": 0.87,
      "business_rules": {
        "box_adjustment": true,
        "scheme_applied": "12+2",
        "expiry_warning": "moderate",
        "styling": {
          "color": "#ffe6cc",
          "tooltip": "Box quantity adjustment applied"
        }
      }
    }
  ],
  "summary": {
    "total_products": 1,
    "predictions_generated": 1,
    "model_version": "1.0.0",
    "processing_time_ms": 45
  }
}
```

### Supported Order Schemes

The system recognizes and predicts various promotional schemes:
- **Simple orders**: "12", "24", "6"
- **Bonus schemes**: "9+1", "12+2", "24+4"
- **Complex patterns**: Automatically detected from historical data

## 📈 Model Performance

The system evaluates models using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Typical performance metrics:
- RMSE: ~2-5 units (depending on data quality)
- MAE: ~1-3 units
- R²: 0.7-0.9 (70-90% variance explained)

## 🧪 Model Evaluation and Selection

This section summarizes the models we evaluated and why we selected XGBoost for production. Metrics were computed on our internal train/test split and validated via cross‑validation. Your results may vary depending on data quality and feature configuration.

### Models Evaluated
- Linear Regression (with/without scaling)
- Random Forest Regressor
- XGBoost Regressor

### Evaluation Protocol
- Hold‑out split and k‑fold cross‑validation on the training set
- Error metrics: RMSE (lower is better), MAE (lower is better), R² (higher is better)
- Same feature set across all models for a fair comparison

### Results (representative previous training run)

| Model              | RMSE | MAE  | R²   |
|--------------------|-----:|-----:|-----:|
| Linear Regression  | 3.45 | 2.31 | 0.78 |
| Random Forest      | 2.95 | 2.06 | 0.84 |
| XGBoost            | 2.62 | 1.88 | 0.87 |

Notes:
- Figures above are illustrative of prior runs on our internal dataset; expect some variation across datasets and seeds.
- The Streamlit app sidebar reads metrics from `models/model_info.json` populated during training, so you can verify live results in the UI.

### Comparison
- Error reduction: XGBoost achieves the lowest RMSE/MAE among the candidates, indicating the most accurate base‑quantity predictions.
- Generalization: XGBoost consistently delivered higher R² on validation folds, suggesting better fit without excessive overfitting.
- Robustness: Compared to Linear Regression, the tree‑based models naturally capture non‑linear interactions and handle skewed/heterogeneous tabular features.

### Why XGBoost was chosen
- Performance: Best overall accuracy (lowest RMSE/MAE) in our experiments.
- Stability: More consistent fold‑to‑fold metrics and less sensitivity to outliers than Linear Regression.
- Tabular suitability: Gradient‑boosted trees excel on structured/tabular data and can model non‑linear relationships and interactions out‑of‑the‑box.
- Interpretability: Offers feature importance; can be paired with SHAP for local/global explanations when needed.
- Practicality: Efficient training/inference and robust defaults; integrates smoothly with our existing preprocessing pipeline.

### Reproducing and Updating Metrics
1. Open and run `notebooks/02_model_training.ipynb` to retrain and evaluate models.
2. The training notebook writes `models/model_info.json` with the final model type, feature columns, and performance metrics.
3. The Streamlit app (`app/streamlit_app.py`) reads and displays these metrics under “Model Information”.

## 🛠️ Development

### Adding New Features

1. **New Preprocessing Logic**: Add functions to `app/utils.py`
2. **Model Improvements**: Modify `notebooks/02_model_training.ipynb`
3. **API Endpoints**: Extend `app/api.py`
4. **UI Features**: Update `app/streamlit_app.py`

### Testing

```bash
# Test API endpoints
python -m pytest tests/  # (if tests are added)

# Manual testing
python app/api.py  # Start API server
python -c "from app.utils import *; test_functions()"  # Test utilities
```

### Deployment

#### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Production Considerations
- Use environment variables for configuration
- Implement proper logging and monitoring
- Add authentication for API endpoints
- Set up model versioning and A/B testing
- Configure load balancing for high traffic

#### Streamlit Community Cloud (UI)
1. Push this repository to GitHub.
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), create a new app and connect your repo.
3. Set the main file to `app/streamlit_app.py`.
4. Set Python version to 3.10 and point to `requirements.txt`.
5. Deploy. The app will be available at a public URL.

#### Render (API)
1. Create a new Web Service on [Render](https://render.com/).
2. Runtime: Python 3.10.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
5. Add model files under `models/` as environment-provisioned files or persistent storage.
6. Once deployed, test with `https://<your-service>.onrender.com/health`.

### Cleanup Notes
- This repository is kept lightweight for easy cloning and running.
- Training notebooks and ad-hoc debug scripts can be removed for production deployments.
- Ensure model artifacts exist under `models/` before running the API or app.

## 📋 Requirements

### System Requirements
- Python 3.10+
- 4GB+ RAM (for large datasets)
- 1GB+ disk space

### Key Dependencies
- **Data Science**: pandas, numpy, scikit-learn, xgboost
- **Visualization**: plotly
- **Web Frameworks**: streamlit, streamlit-aggrid, fastapi, uvicorn
- **File Handling**: openpyxl, xlrd
- **Enhanced UI**: st-aggrid for professional data grids

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

1. **Model not loading**: Ensure you've run the training notebook first
2. **Excel file errors**: Check column names and data format
3. **API connection issues**: Verify the server is running on correct port
4. **Memory errors**: Reduce batch size or upgrade system RAM

### Getting Help

- Check the Jupyter notebooks for detailed examples
- Review API documentation at `/docs` endpoint
- Examine log files for error details
- Open GitHub issues for bugs or feature requests

## 🔄 Changelog

### Version 2.0.0 (Latest)
- 🎆 **Enhanced Interactive Grid**: Professional st-aggrid integration
- 🎨 **Visual Business Rules**: Color-coded cells with hover tooltips
- ✏️ **In-Grid Editing**: Direct editing with change tracking
- 🔍 **Advanced Filtering**: Real-time filters with persistence
- 📥 **One-Click Export**: Simplified download with automatic server backup
- 🟢 **Change Highlighting**: Green background for user modifications
- 📅 **Expiry Management**: MM/YY format support with tiered warnings
- 🔄 **Filter Persistence**: Edits maintained across filter operations
- 🏢 **Supplier Management**: Multi-select supplier exclusion
- 📊 **Excel Preview**: True WYSIWYG grid experience

### Version 1.0.0
- Initial release with complete ML pipeline
- Basic Streamlit web interface
- FastAPI REST service
- Comprehensive documentation
- Support for Excel file processing
- Order scheme handling (9+1, 12+2, etc.)

---

**Built with ❤️ for pharmacy demand forecasting**
