import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.cnn_model import train_cnn
from models.svm_model import train_svm
from feature_ensemble import FeatureEnsemble

vocab_size = 50
embedding_dim = 8
max_length = 6
X_text = np.random.randint(0, vocab_size, size=(10, max_length))
X_features = np.random.randn(10, 4)
y = np.random.randint(0, 2, size=(10, ))

cnn = train_cnn(X_text, y, vocab_size, embedding_dim, max_length, epochs=1, batch_size=2)
# Trích xuất đặc trưng CNN
from models.cnn_model import extract_cnn_features
cnn_feats = extract_cnn_features(cnn, X_text)
combined = np.concatenate([cnn_feats, X_features], axis=1)
svm = train_svm(combined, y)

ensemble = FeatureEnsemble(cnn, svm)
proba = ensemble.predict_proba(X_text, X_features)
print("FeatureEnsemble predict_proba:", proba)