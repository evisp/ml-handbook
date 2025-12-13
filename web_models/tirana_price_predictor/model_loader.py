import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class TiranaPriceModel:
    def __init__(self):
        self.model = joblib.load('models/tirana_price_model.joblib')
        self.feature_names = joblib.load('models/feature_names.joblib')
        self.centers = joblib.load('models/tirana_centers.joblib')
        
    def apply_defaults(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """Convert user input + defaults to 19-feature vector."""
        
        # Default values (model medians)
        defaults = {
            'floor': 2.0,
            'living_rooms': 1.0,
            'kitchens': 1.0,
            'balconies': 0.0,
            'dist_to_city_center_km': 2.0,
            'dist_to_artificial_lake_km': 2.0,
            'dist_to_blloku_km': 2.0,
            'has_elevator': 0,
            'has_terrace': 0,
            'has_garage': 0,
            'has_parking': 0,
            'has_carport': 0,
            'has_garden': 0,
            'furnishing_unfurnished': 1,
            'furnishing_fully_furnished': 0,
            'furnishing_partially_furnished': 0
        }
        
        # User overrides
        input_data = {**defaults, **user_input}
        
        # Ensure all 19 features
        df = pd.DataFrame([input_data])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = defaults.get(col, 0)
        
        return df[self.feature_names].fillna(0)
    
    def predict(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Full prediction pipeline."""
        X = self.apply_defaults(user_input)
        log_price = self.model.predict(X)[0]
        price_eur = np.expm1(log_price)
        
        return {
            'predicted_price_eur': int(price_eur),
            'log_price': float(log_price),
            'confidence_interval': [int(price_eur * 0.75), int(price_eur * 1.25)]
        }

# Global instance
model = TiranaPriceModel()
