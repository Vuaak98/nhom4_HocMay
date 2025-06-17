import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from cnn_model import build_cnn_model, train_cnn, extract_cnn_features

# Tham số giả lập
vocab_size = 100
embedding_dim = 16
max_length = 10
num_samples = 20

# Dữ liệu giả
X_train = np.random.randint(0, vocab_size, size=(num_samples, max_length))
y_train = np.random.randint(0, 2, size=(num_samples, ))

# Test build và train
print(">>> Test build_cnn_model và train_cnn")
model = train_cnn(X_train, y_train, vocab_size, embedding_dim, max_length, epochs=1, batch_size=4)
print("Model summary:")
model.summary()

# Test extract features
print(">>> Test extract_cnn_features")
features = extract_cnn_features(model, X_train)
print("Features shape:", features.shape)