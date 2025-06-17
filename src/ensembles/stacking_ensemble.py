from sklearn.linear_model import LogisticRegression
import numpy as np

class StackingEnsemble:
    def __init__(self, cnn_model, svm_model):
        self.cnn = cnn_model
        self.svm = svm_model
        self.meta_classifier = LogisticRegression()

    def fit(self, X_text_val, X_features_val, y_val):
        p_cnn_val = self.cnn.predict(X_text_val)
        p_svm_val = self.svm.predict_proba(X_features_val)[:, 1]
        meta_features = np.column_stack((p_cnn_val, p_svm_val))
        self.meta_classifier.fit(meta_features, y_val)

    def predict_proba(self, X_text, X_features):
        p_cnn = self.cnn.predict(X_text)
        p_svm = self.svm.predict_proba(X_features)[:, 1]
        meta_features = np.column_stack((p_cnn, p_svm))
        return self.meta_classifier.predict_proba(meta_features)[:, 1]
