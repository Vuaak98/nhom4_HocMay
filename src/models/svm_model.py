from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_svm(X_train_features, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'))
    ])
    pipeline.fit(X_train_features, y_train)
    return pipeline
