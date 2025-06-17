import numpy as np

class VotingEnsemble:
    def __init__(self, cnn_model, svm_model, weight_cnn=0.5):
        self.cnn = cnn_model
        self.svm = svm_model
        self.w_cnn = weight_cnn
        self.w_svm = 1 - weight_cnn

    def predict_proba(self, X_text, X_features):
        p_cnn = self.cnn.predict(X_text)
        p_svm = self.svm.predict_proba(X_features)[:, 1]
        p_cnn = np.array(p_cnn).ravel()
        p_svm = np.array(p_svm).ravel()
        avg_probs = self.w_cnn * p_cnn + self.w_svm * p_svm
        return avg_probs
