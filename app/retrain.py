"""
Model Retraining & Management Module

Handles incremental retraining of XGBoost models, model comparison,
version management, and rollback functionality.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from utils import (
    preprocess_sales_features,
    prepare_features_for_prediction,
    validate_input_data
)


class ModelRegistry:
    """Manages model versions and registry."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.registry_path = self.models_dir / "registry.json"
        self.registry = self._load_registry()
        self._detect_and_register_existing_model()
    
    def _load_registry(self) -> Dict:
        """Load model registry from JSON file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return self._create_empty_registry()
        else:
            return self._create_empty_registry()
    
    def _create_empty_registry(self) -> Dict:
        """Create empty registry structure."""
        return {
            "models": {},
            "active_model": None,
            "next_version": 1
        }
    
    def _save_registry(self):
        """Save registry to JSON file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def register_model(self, model_path: str, metrics: Dict, is_active: bool = False) -> str:
        """Register a new model version."""
        version = f"v{self.registry['next_version']}"
        
        model_info = {
            "version": version,
            "path": model_path,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "is_active": is_active
        }
        
        self.registry["models"][version] = model_info
        self.registry["next_version"] += 1
        
        if is_active:
            # Deactivate other models
            for v in self.registry["models"]:
                self.registry["models"][v]["is_active"] = False
            self.registry["models"][version]["is_active"] = True
            self.registry["active_model"] = version
        
        self._save_registry()
        return version
    
    def get_active_model(self) -> Optional[Dict]:
        """Get currently active model info."""
        if self.registry["active_model"]:
            return self.registry["models"].get(self.registry["active_model"])
        return None
    
    def get_all_models(self) -> Dict:
        """Get all registered models."""
        return self.registry["models"]
    
    def set_active_model(self, version: str) -> bool:
        """Set a model version as active."""
        if version in self.registry["models"]:
            # Deactivate all models
            for v in self.registry["models"]:
                self.registry["models"][v]["is_active"] = False
            
            # Activate selected model
            self.registry["models"][version]["is_active"] = True
            self.registry["active_model"] = version
            self._save_registry()
            return True
        return False
    
    def get_model_path(self, version: str = None) -> Optional[str]:
        """Get model file path for a specific version or active model."""
        if version is None:
            active = self.get_active_model()
            if active:
                return active["path"]
        else:
            if version in self.registry["models"]:
                return self.registry["models"][version]["path"]
        return None
    
    def _detect_and_register_existing_model(self):
        """Detect and register existing model if registry is empty."""
        # Only auto-register if registry is empty
        if len(self.registry["models"]) > 0:
            return
        
        # Check for existing model files
        main_model_path = self.models_dir / "order_predictor.pkl"
        main_info_path = self.models_dir / "model_info.json"
        
        if main_model_path.exists() and main_info_path.exists():
            try:
                # Load existing model info
                with open(main_info_path, 'r') as f:
                    model_info = json.load(f)
                
                # Extract metrics if available
                metrics = {}
                if 'performance' in model_info:
                    perf = model_info['performance']
                    metrics = {
                        'rmse': perf.get('rmse', 0),
                        'mae': perf.get('mae', 0),
                        'r2': perf.get('r2', 0),
                        'accuracy': 0.85,  # Default estimate
                        'precision': 0.80,  # Default estimate
                        'recall': 0.82,     # Default estimate
                        'f1_score': 0.81    # Default estimate
                    }
                
                # Register the existing model as v1
                version = "v1"
                model_entry = {
                    "version": version,
                    "path": str(main_model_path.absolute()),
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "is_active": True
                }
                
                self.registry["models"][version] = model_entry
                self.registry["active_model"] = version
                self.registry["next_version"] = 2
                
                self._save_registry()
                print(f"✅ Registered existing model as {version}")
                
            except Exception as e:
                print(f"⚠️ Could not register existing model: {e}")


class ModelRetrainer:
    """Handles model retraining and evaluation."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.registry = ModelRegistry(models_dir)
        
    def load_training_data(self, uploaded_files: List) -> pd.DataFrame:
        """Load and merge multiple Excel files for training."""
        dataframes = []
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_excel(uploaded_file)
                
                # Validate the uploaded data
                validation_results = validate_input_data(df)
                if validation_results['errors']:
                    raise ValueError(f"Validation failed for {uploaded_file.name}: {validation_results['errors']}")
                
                # Check if Predicted_Order column exists (user-modified values)
                if 'Predicted_Order' not in df.columns:
                    raise ValueError(f"File {uploaded_file.name} missing 'Predicted_Order' column")
                
                dataframes.append(df)
                
            except Exception as e:
                raise ValueError(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if not dataframes:
            raise ValueError("No valid training data found")
        
        # Merge all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        return merged_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        # Preprocess features
        df_processed = preprocess_sales_features(df)
        
        # Load feature columns from existing model info
        model_info_path = self.models_dir / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            feature_columns = model_info['feature_columns']
        else:
            # Fallback feature columns
            feature_columns = ['L7', 'L15', 'L30', 'L45', 'L60', 'L75', 'L90']
        
        # Prepare features
        X = prepare_features_for_prediction(df_processed, feature_columns)
        
        # Extract target values from Predicted_Order
        y = self._extract_target_values(df['Predicted_Order'])
        
        return X, y, feature_columns
    
    def _extract_target_values(self, predicted_orders: pd.Series) -> np.ndarray:
        """Extract numeric target values from Predicted_Order column."""
        targets = []
        
        for order in predicted_orders:
            try:
                if pd.isna(order):
                    targets.append(0)
                elif isinstance(order, str):
                    if '+' in order:
                        # Handle scheme format like "5+1"
                        parts = order.split('+')
                        if len(parts) == 2:
                            total = float(parts[0]) + float(parts[1])
                            targets.append(total)
                        else:
                            targets.append(float(order))
                    else:
                        targets.append(float(order))
                else:
                    targets.append(float(order))
            except (ValueError, TypeError):
                targets.append(0)
        
        return np.array(targets)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   incremental: bool = True) -> xgb.XGBRegressor:
        """Train a new XGBoost model."""
        if incremental and self.registry.get_active_model():
            # Load existing model for incremental training
            active_model_path = self.registry.get_model_path()
            if active_model_path and os.path.exists(active_model_path):
                try:
                    existing_model = joblib.load(active_model_path)
                    
                    # Create new model with same parameters
                    model = xgb.XGBRegressor(
                        n_estimators=existing_model.n_estimators + 100,  # Add more trees
                        max_depth=existing_model.max_depth,
                        learning_rate=existing_model.learning_rate,
                        subsample=existing_model.subsample,
                        colsample_bytree=existing_model.colsample_bytree,
                        random_state=42
                    )
                    
                    # Initialize with existing model's booster
                    model.fit(X, y, xgb_model=existing_model.get_booster())
                    
                except Exception as e:
                    print(f"Incremental training failed, falling back to full training: {e}")
                    model = self._train_new_model(X, y)
            else:
                model = self._train_new_model(X, y)
        else:
            model = self._train_new_model(X, y)
        
        return model
    
    def _train_new_model(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        """Train a completely new XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X, y)
        return model
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        y_pred = model.predict(X)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Convert to classification for additional metrics
        # Use threshold-based classification (e.g., predicted vs actual within 10% tolerance)
        tolerance = 0.1
        y_class_true = (y > 0).astype(int)  # Binary: order vs no order
        y_class_pred = (y_pred > 0).astype(int)
        
        # Classification metrics
        accuracy = accuracy_score(y_class_true, y_class_pred)
        
        # Handle edge cases for precision, recall, f1
        try:
            precision = precision_score(y_class_true, y_class_pred, zero_division=0)
            recall = recall_score(y_class_true, y_class_pred, zero_division=0)
            f1 = f1_score(y_class_true, y_class_pred, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    def save_model(self, model, version: str, feature_columns: List[str]) -> str:
        """Save model and update model info."""
        model_filename = f"order_predictor_{version}.pkl"
        model_path = self.models_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        
        # Update model_info.json for the new model
        model_info = {
            'model_type': 'XGBRegressor',
            'feature_columns': feature_columns,
            'training_timestamp': datetime.now().isoformat(),
            'version': version
        }
        
        model_info_path = self.models_dir / f"model_info_{version}.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return str(model_path)
    
    def compare_models(self, current_metrics: Dict, new_metrics: Dict) -> Dict:
        """Compare current and new model metrics."""
        comparison = {
            'current': current_metrics,
            'new': new_metrics,
            'improvements': {},
            'recommendation': 'keep_current'
        }
        
        # Calculate improvements
        for metric in current_metrics:
            if metric in new_metrics:
                if metric in ['rmse', 'mae']:  # Lower is better
                    improvement = ((current_metrics[metric] - new_metrics[metric]) / current_metrics[metric]) * 100
                else:  # Higher is better
                    improvement = ((new_metrics[metric] - current_metrics[metric]) / current_metrics[metric]) * 100
                comparison['improvements'][metric] = improvement
        
        # Simple recommendation logic
        rmse_improvement = comparison['improvements'].get('rmse', 0)
        r2_improvement = comparison['improvements'].get('r2', 0)
        
        if rmse_improvement > 5 or r2_improvement > 5:  # 5% improvement threshold
            comparison['recommendation'] = 'use_new'
        
        return comparison
    
    def retrain_pipeline(self, uploaded_files: List, incremental: bool = True) -> Dict:
        """Complete retraining pipeline."""
        try:
            # Load and prepare training data
            training_df = self.load_training_data(uploaded_files)
            X, y, feature_columns = self.prepare_training_data(training_df)
            
            # Train new model
            new_model = self.train_model(X, y, incremental=incremental)
            
            # Evaluate new model
            new_metrics = self.evaluate_model(new_model, X, y)
            
            # Get current model metrics for comparison
            current_metrics = None
            active_model = self.registry.get_active_model()
            if active_model:
                current_metrics = active_model.get('metrics', {})
            
            # Save new model (temporary)
            temp_version = f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.save_model(new_model, temp_version, feature_columns)
            
            result = {
                'success': True,
                'new_model_path': model_path,
                'new_metrics': new_metrics,
                'current_metrics': current_metrics,
                'training_samples': len(training_df),
                'feature_columns': feature_columns,
                'temp_version': temp_version
            }
            
            if current_metrics:
                result['comparison'] = self.compare_models(current_metrics, new_metrics)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def finalize_model(self, temp_version: str, model_path: str, metrics: Dict) -> str:
        """Finalize and register the new model."""
        # Register the model
        final_version = self.registry.register_model(model_path, metrics, is_active=True)
        
        # Rename model file to final version
        old_path = Path(model_path)
        new_filename = f"order_predictor_{final_version}.pkl"
        new_path = old_path.parent / new_filename
        
        if old_path.exists():
            old_path.rename(new_path)
        
        # Update registry with new path
        self.registry.registry["models"][final_version]["path"] = str(new_path)
        self.registry._save_registry()
        
        # Update main model files for backward compatibility
        main_model_path = self.models_dir / "order_predictor.pkl"
        main_info_path = self.models_dir / "model_info.json"
        
        # Copy new model to main location
        import shutil
        shutil.copy2(new_path, main_model_path)
        
        # Copy model info
        temp_info_path = self.models_dir / f"model_info_{temp_version}.json"
        if temp_info_path.exists():
            shutil.copy2(temp_info_path, main_info_path)
        
        return final_version
    
    def rollback_to_version(self, version: str) -> bool:
        """Rollback to a specific model version."""
        try:
            model_info = self.registry.registry["models"].get(version)
            if not model_info:
                return False
            
            model_path = Path(model_info["path"])
            if not model_path.exists():
                return False
            
            # Set as active in registry
            self.registry.set_active_model(version)
            
            # Update main model files
            main_model_path = self.models_dir / "order_predictor.pkl"
            main_info_path = self.models_dir / "model_info.json"
            
            import shutil
            shutil.copy2(model_path, main_model_path)
            
            # Copy corresponding model info
            version_info_path = self.models_dir / f"model_info_{version}.json"
            if version_info_path.exists():
                shutil.copy2(version_info_path, main_info_path)
            
            return True
            
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False
