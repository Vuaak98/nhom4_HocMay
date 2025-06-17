import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.cnn_model import train_cnn
from models.svm_model import train_svm
from stacking_ensemble import StackingEnsemble

vocab_size = 50
embedding_dim = 8
max_length = 6
X_text = np.random.randint(0, vocab_size, size=(20, max_length))
X_features = np.random.randn(20, 4)
y = np.random.randint(0, 2, size=(20, ))

cnn = train_cnn(X_text, y, vocab_size, embedding_dim, max_length, epochs=1, batch_size=2)
svm = train_svm(X_features, y)

# Chia validation và test
X_text_val, X_text_test = X_text[:10], X_text[10:]
X_features_val, X_features_test = X_features[:10], X_features[10:]
y_val, y_test = y[:10], y[10:]

ensemble = StackingEnsemble(cnn, svm)
ensemble.fit(X_text_val, X_features_val, y_val)
proba = ensemble.predict_proba(X_text_test, X_features_test)
print("StackingEnsemble predict_proba:", proba)