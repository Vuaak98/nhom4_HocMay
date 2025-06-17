import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from svm_model import train_svm

# Dữ liệu giả
X_train = np.random.randn(20, 8)
y_train = np.random.randint(0, 2, size=(20, ))

print(">>> Test train_svm")
svm = train_svm(X_train, y_train)
print("SVM pipeline:", svm)
print("Predict proba:", svm.predict_proba(X_train)[:5])