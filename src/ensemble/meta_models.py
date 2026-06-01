"""
Meta-learner models for ensemble stacking.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def get_meta_model(model_type='xgboost'):
    """
    Get a meta-learner model.
    
    Args:
        model_type: 'logistic', 'random_forest', or 'xgboost'
    
    Returns:
        Meta-learner instance
    """
    if model_type == 'logistic':
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    elif model_type == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    else:
        raise ValueError(f"Unknown meta-learner type: {model_type}")


class MetaLearner:
    """Wrapper for meta-learning models."""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize meta-learner.
        
        Args:
            model_type: Type of meta-learner
        """
        self.model_type = model_type
        self.model = get_meta_model(model_type)
    
    def fit(self, X, y):
        """Train meta-learner."""
        print(f"Training {self.model_type} meta-learner...")
        self.model.fit(X, y)
        print("Meta-learner training completed!")
    
    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_type} does not support predict_proba")
    
    def save_model(self, path):
        """Save meta-learner."""
        import joblib
        joblib.dump(self.model, path)
        print(f"Meta-learner saved to {path}")
    
    def load_model(self, path):
        """Load meta-learner."""
        import joblib
        self.model = joblib.load(path)
        print(f"Meta-learner loaded from {path}")
