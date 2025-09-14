"""
FastAPI service for pharmacy demand forecasting.
Provides REST API endpoints for order prediction.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from io import BytesIO

# Import utility functions
from .utils import (
    preprocess_sales_features,
    prepare_features_for_prediction,
    load_model,
    reconstruct_order_prediction,
    validate_input_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pharmacy Demand Forecasting API",
    description="API for predicting pharmacy order quantities based on sales history",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model = None
model_info = None
scaler = None

# Pydantic models for request/response
class ProductFeatures(BaseModel):
    """Product features for prediction."""
    product_code: Optional[str] = Field(None, description="Product identifier")
    product_name: Optional[str] = Field(None, description="Product name")
    L7: Optional[float] = Field(0, description="Sales in last 7 days")
    L15: Optional[float] = Field(0, description="Sales in last 15 days")
    L30: Optional[float] = Field(0, description="Sales in last 30 days")
    L60: Optional[float] = Field(0, description="Sales in last 60 days")
    L90: Optional[float] = Field(0, description="Sales in last 90 days")
    L120: Optional[float] = Field(0, description="Sales in last 120 days")

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    products: List[ProductFeatures] = Field(..., description="List of products to predict")

class PredictionResult(BaseModel):
    """Individual prediction result."""
    product_code: Optional[str]
    product_name: Optional[str]
    predicted_order: str = Field(..., description="Predicted order string (e.g., '12', '9+1')")
    predicted_base_quantity: int = Field(..., description="Predicted base quantity")
    confidence_score: float = Field(..., description="Prediction confidence (0-1)")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    message: str
    predictions: List[PredictionResult]
    model_info: Dict

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    version: str

def load_trained_model():
    """Load the trained model and associated files."""
    global model, model_info, scaler
    
    try:
        model_path = Path("models/order_predictor.pkl")
        model_info_path = Path("models/model_info.json")
        scaler_path = Path("models/scaler.pkl")
        
        if not model_path.exists():
            logger.error("Model file not found")
            return False
        
        # Load model
        model = load_model(str(model_path))
        logger.info("Model loaded successfully")
        
        # Load model info
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            logger.info("Model info loaded")
        else:
            model_info = {"model_type": "Unknown", "feature_columns": []}
        
        # Load scaler if exists
        if scaler_path.exists():
            import joblib
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = load_trained_model()
    if not success:
        logger.warning("Failed to load model on startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_type=model_info.get("model_type") if model_info else None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_orders(request: PredictionRequest):
    """Predict order quantities for given products."""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check model files."
        )
    
    try:
        # Convert request to DataFrame
        products_data = []
        for product in request.products:
            product_dict = product.dict()
            products_data.append(product_dict)
        
        df = pd.DataFrame(products_data)
        
        # Validate input data
        validation_results = validate_input_data(df)
        if validation_results['errors']:
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {validation_results['errors']}"
            )
        
        # Preprocess features
        df_processed = preprocess_sales_features(df)
        
        # Get feature columns
        feature_cols = model_info.get('feature_columns', [])
        if not feature_cols:
            # Fallback to common feature columns
            feature_cols = [col for col in df_processed.columns 
                          if col.startswith('L') and col[1:].isdigit()]
        
        # Prepare features for prediction
        X = prepare_features_for_prediction(df_processed, feature_cols)
        
        # Apply scaling if needed (for Linear Regression)
        if scaler is not None and model_info.get('model_type') == 'Linear Regression':
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        # Calculate confidence scores (simple approach based on prediction variance)
        # For more sophisticated confidence, you could use prediction intervals
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions) if len(predictions) > 1 else 1.0
        confidence_scores = []
        
        for pred in predictions:
            # Simple confidence based on how close prediction is to mean
            if std_pred > 0:
                z_score = abs(pred - mean_pred) / std_pred
                confidence = max(0.1, 1.0 - min(z_score / 3.0, 0.9))  # Scale to 0.1-1.0
            else:
                confidence = 0.8  # Default confidence
            confidence_scores.append(confidence)
        
        # Create prediction results
        results = []
        for i, (product, pred, conf) in enumerate(zip(request.products, predictions, confidence_scores)):
            # Round to nearest integer and ensure positive
            base_qty = max(0, int(round(pred)))
            
            # Generate order string with scheme logic
            scheme_prob = conf * 0.4 if base_qty >= 9 else conf * 0.1
            order_str = reconstruct_order_prediction(base_qty, scheme_prob)
            
            result = PredictionResult(
                product_code=product.product_code,
                product_name=product.product_name,
                predicted_order=order_str,
                predicted_base_quantity=base_qty,
                confidence_score=round(conf, 3)
            )
            results.append(result)
        
        return PredictionResponse(
            success=True,
            message=f"Successfully predicted orders for {len(results)} products",
            predictions=results,
            model_info={
                "model_type": model_info.get("model_type", "Unknown"),
                "features_used": len(feature_cols),
                "performance": model_info.get("performance", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict orders from uploaded Excel/CSV file."""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check model files."
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(BytesIO(contents))
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload Excel (.xlsx, .xls) or CSV files."
            )
        
        # Validate input data
        validation_results = validate_input_data(df)
        if validation_results['errors']:
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {validation_results['errors']}"
            )
        
        # Preprocess and predict
        df_processed = preprocess_sales_features(df)
        feature_cols = model_info.get('feature_columns', [])
        if not feature_cols:
            feature_cols = [col for col in df_processed.columns 
                          if col.startswith('L') and col[1:].isdigit()]
        
        X = prepare_features_for_prediction(df_processed, feature_cols)
        
        # Apply scaling if needed
        if scaler is not None and model_info.get('model_type') == 'Linear Regression':
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        # Add predictions to dataframe with proper column ordering
        df_result = df.copy()
        predicted_orders = []
        predicted_quantities = []
        
        for i, pred in enumerate(predictions):
            base_qty = max(0, int(round(pred)))
            
            # Get scheme info from original data if available
            scheme_info = None
            if 'Scm' in df.columns and i < len(df):
                scheme_info = df.iloc[i]['Scm'] if pd.notna(df.iloc[i]['Scm']) else None
            
            # Use enhanced scheme prediction
            scheme_prob = 0.3 if base_qty >= 9 else 0.1
            order_str = reconstruct_order_prediction(base_qty, scheme_info, scheme_prob)
            
            predicted_orders.append(order_str)
            predicted_quantities.append(base_qty)
        
        # Find Order column position (case insensitive) and insert predictions next to it
        order_col = None
        for col in df_result.columns:
            if col.lower() in ('order', 'ord', 'oreder'):
                order_col = col
                break

        # Add the prediction data first
        df_result['Predicted_Order'] = predicted_orders
        df_result['Predicted_Base_Quantity'] = predicted_quantities

        # If Order column exists, reorder to place predictions right after it using df_result columns
        if order_col is not None:
            cols = list(df_result.columns)
            # Remove if already present to avoid duplicates
            cols = [c for c in cols if c not in ['Predicted_Order', 'Predicted_Base_Quantity']]
            order_idx = cols.index(order_col)
            cols.insert(order_idx + 1, 'Predicted_Order')
            cols.insert(order_idx + 2, 'Predicted_Base_Quantity')
            df_result = df_result.reindex(columns=cols)
        
        # Convert to JSON for response
        result_json = df_result.to_dict('records')
        
        return {
            "success": True,
            "message": f"Successfully processed {len(df_result)} records",
            "data": result_json,
            "summary": {
                "total_records": len(df_result),
                "total_predicted_quantity": sum(predicted_quantities),
                "average_order_size": np.mean(predicted_quantities),
                "validation_warnings": validation_results.get('warnings', [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_loaded": True,
        "model_info": model_info,
        "feature_columns": model_info.get("feature_columns", []),
        "model_type": model_info.get("model_type", "Unknown"),
        "performance": model_info.get("performance", {}),
        "scaler_loaded": scaler is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Pharmacy Demand Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_file": "/predict/file",
            "model_info": "/model/info",
            "docs": "/docs"
        },
        "model_status": "loaded" if model is not None else "not_loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
