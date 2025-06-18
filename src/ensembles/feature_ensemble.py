from src.models.cnn_model import extract_cnn_features
import numpy as np

class FeatureEnsemble:
    def __init__(self, cnn_model, svm_model_for_features):
        self.cnn = cnn_model
        self.svm = svm_model_for_features

    def predict_proba(self, X_text, X_features):
        cnn_features = extract_cnn_features(self.cnn, X_text)
        combined_features = np.concatenate([cnn_features, X_features], axis=1)
        return self.svm.predict_proba(combined_features)[:, 1]
