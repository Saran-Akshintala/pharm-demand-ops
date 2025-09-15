# 💊 Pharmacy Demand Forecasting System

A comprehensive machine learning solution for predicting pharmacy order quantities based on historical sales data. This system provides end-to-end workflow from data analysis to deployment with both web UI and API interfaces.

## 🎯 Project Overview

This project helps pharmacists predict optimal order quantities by analyzing historical sales patterns and generating intelligent recommendations. The system handles complex order schemes (like "9+1" promotions) and provides confidence scores for predictions.

### Key Features

- **📊 Comprehensive EDA**: Analyze sales patterns, trends, and correlations
- **🤖 ML Pipeline**: Train and evaluate multiple models (Linear Regression, Random Forest, XGBoost)
- **🌐 Web Interface**: User-friendly Streamlit app for Excel upload/download
- **🔌 REST API**: FastAPI service for enterprise integration
- **📈 Scheme Handling**: Parse and predict promotional schemes (e.g., "9+1", "12+2")
- **📋 Excel Integration**: Direct Excel file processing and output

## 📂 Project Structure

```
pharm-demand-ops/
├── app/                          # Application code
│   ├── streamlit_app.py         # Streamlit web interface
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

### Web Interface (Streamlit)

1. **Upload Excel File**: Upload your pharmacy data with sales history
2. **Review Data**: Check the uploaded data preview and validation warnings
3. **Get Predictions**: Click to generate order predictions with confidence scores
4. **Download Results**: Download Excel file with predicted order quantities

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

### API Endpoints

- `GET /health` - Health check and model status
- `POST /predict` - Predict orders for JSON payload
- `POST /predict/file` - Upload Excel/CSV for batch prediction
- `GET /model/info` - Get model information and performance metrics
- `GET /` - API information and available endpoints

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
- **Web Frameworks**: streamlit, fastapi, uvicorn
- **File Handling**: openpyxl, xlrd

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

### Version 1.0.0
- Initial release with complete ML pipeline
- Streamlit web interface
- FastAPI REST service
- Comprehensive documentation
- Support for Excel file processing
- Order scheme handling (9+1, 12+2, etc.)

---

**Built with ❤️ for pharmacy demand forecasting**
