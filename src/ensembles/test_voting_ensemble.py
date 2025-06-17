import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.cnn_model import build_cnn_model, train_cnn
from models.svm_model import train_svm
from voting_ensemble import VotingEnsemble

# Dữ liệu giả
vocab_size = 50
embedding_dim = 8
max_length = 6
X_text = np.random.randint(0, vocab_size, size=(10, max_length))
X_features = np.random.randn(10, 4)
y = np.random.randint(0, 2, size=(10, ))

cnn = train_cnn(X_text, y, vocab_size, embedding_dim, max_length, epochs=1, batch_size=2)
svm = train_svm(X_features, y)

ensemble = VotingEnsemble(cnn, svm)
proba = ensemble.predict_proba(X_text, X_features)
print("VotingEnsemble predict_proba:", proba)