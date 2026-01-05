#started 1/4/2026

import pandas as pd
import joblib
import dill
import numpy as np

class MVPService:
    """
    Pure prediction + explanation service.
    - Loads artifacts once
    - Accepts feature dictionaries
    - Returns JSON-serializable outputs
    """

    def __init__(self):
        import sys
        print("[MVPService] Loading scalers...", file=sys.stderr)
        self.scalers = joblib.load("scalers.pkl")
        print("[MVPService] Loading xgb_model...", file=sys.stderr)
        self.xgb_model = joblib.load("xgb_model.pkl")
        print("[MVPService] Loading LIME explainer...", file=sys.stderr)
        with open("lime_explainer.dill", "rb") as f:
            bundle = dill.load(f)
        print("[MVPService] LIME explainer loaded.", file=sys.stderr)
        self.explainer = bundle["explainer"]
        # We'll create our own predict function instead of using the problematic one
        print("[MVPService] Creating model predict function...", file=sys.stderr)

        # Setup scalers
        self.std_scaler = self.scalers["standard_scaler"]
        self.pwr_scaler = self.scalers["power_scaler"]

        self.standard_cols = [
            "temp_max_F",
            "humidity_pct",
            "windspeed_mph",
            "ndvi",
            "slope"
        ]

        self.power_cols = [
            "precip_in",
            "pop_density"
        ]

        # Canonical feature order (CRITICAL for model correctness)
        self.feature_order = self.xgb_model.get_booster().feature_names
        print("[MVPService] Initialization complete.", file=sys.stderr)
    
    def _model_predict_fn(self, X):
        """
        A proper predict function for LIME that works with the XGBoost model.
        X is expected to be a numpy array of shape (n_samples, n_features)
        """
        import pandas as pd
        # Convert to DataFrame with proper feature names
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_order)
        else:
            df = X
        # Return predictions
        return self.xgb_model.predict(df)


    def _prepare_input(self, features: dict):
        import sys
        print("[MVPService] _prepare_input called with:", features, file=sys.stderr)
        df = pd.DataFrame([features])[self.feature_order]
        df_nonscaled = df.copy()
        print("[MVPService] DataFrame created:", df, file=sys.stderr)
        # Apply scalers
        df[self.standard_cols] = self.std_scaler.transform(df[self.standard_cols])
        df[self.power_cols] = self.pwr_scaler.transform(df[self.power_cols])
        print("[MVPService] DataFrame after scaling:", df, file=sys.stderr)
        x_instance = df.iloc[0].to_numpy()
        print("[MVPService] x_instance:", x_instance, file=sys.stderr)
        return df, df_nonscaled, x_instance

    def predict(self, features: dict):
        import sys
        print("[MVPService] predict called", file=sys.stderr)
        df, _, _ = self._prepare_input(features)
        pred_log = float(self.xgb_model.predict(df)[0])
        pred_acres = float(10 ** pred_log)
        print(f"[MVPService] pred_log: {pred_log}, pred_acres: {pred_acres}", file=sys.stderr)
        if np.isnan(pred_log) or np.isnan(pred_acres) or np.isinf(pred_log) or np.isinf(pred_acres):
            print("[MVPService] ERROR: Model returned NaN or inf!", file=sys.stderr)
            return {"error": "Model returned invalid prediction (NaN or inf). Check input ranges."}
        return {
            "prediction_log": pred_log,
            "prediction_acres": pred_acres
        }

    def explain(self, features: dict, top_k=10):
        import sys
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        print("[MVPService] explain called (LIME ENABLED)", file=sys.stderr)
        df, df_nonscaled, x_instance = self._prepare_input(features)
        pred_log = float(self.xgb_model.predict(df)[0])
        pred_acres = float(10 ** pred_log)
        print(f"[MVPService] pred_log: {pred_log}, pred_acres: {pred_acres}", file=sys.stderr)
        
        def run_lime_explanation():
            print("[MVPService] Starting LIME explanation...", file=sys.stderr)
            start_time = time.time()
            
            # Use the model predict function instead of the problematic one
            explanation = self.explainer.explain_instance(
                data_row=x_instance,
                predict_fn=self._model_predict_fn,
                num_features=top_k
            )
            
            elapsed_time = time.time() - start_time
            print(f"[MVPService] LIME explanation completed in {elapsed_time:.2f} seconds", file=sys.stderr)
            
            # Convert LIME explanation to the expected format
            lime_data = []
            for feature, weight in explanation.as_list():
                lime_data.append({
                    "feature": feature,
                    "weight": float(weight)
                })
            
            return lime_data
        
        try:
            # Run LIME explanation with timeout using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_lime_explanation)
                lime_data = future.result(timeout=30)  # 30 second timeout
            
            return {
                "prediction_log": pred_log,
                "prediction_acres": pred_acres,
                "lime_explanation": lime_data,
                "input_features": df_nonscaled.iloc[0].to_dict()
            }
            
        except FutureTimeoutError:
            print("[MVPService] LIME explanation timed out after 30 seconds", file=sys.stderr)
            # Fallback to dummy explanation
            lime_data = [
                {"feature": f, "weight": 0.0} for f in self.feature_order[:top_k]
            ]
            return {
                "prediction_log": pred_log,
                "prediction_acres": pred_acres,
                "lime_explanation": lime_data,
                "input_features": df_nonscaled.iloc[0].to_dict(),
                "error": "LIME explanation timed out"
            }
            
        except Exception as e:
            print(f"[MVPService] LIME explanation error: {e}", file=sys.stderr)
            # Fallback to dummy explanation
            lime_data = [
                {"feature": f, "weight": 0.0} for f in self.feature_order[:top_k]
            ]
            return {
                "prediction_log": pred_log,
                "prediction_acres": pred_acres,
                "lime_explanation": lime_data,
                "input_features": df_nonscaled.iloc[0].to_dict(),
                "error": f"LIME explanation failed: {str(e)}"
            }
    
