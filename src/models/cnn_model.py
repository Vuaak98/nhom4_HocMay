import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def build_cnn_model(vocab_size, embedding_dim, max_length):
    inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, embedding_dim)(inputs)
    conv_layer = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.001))(embedding_layer)
    dropout1 = Dropout(0.5)(conv_layer)
    pooling_layer = GlobalMaxPooling1D()(dropout1)
    feature_extractor_layer = Dense(64, activation='relu', name='feature_extractor', kernel_regularizer=l2(0.001))(pooling_layer)
    dropout2 = Dropout(0.5)(feature_extractor_layer)
    outputs = Dense(1, activation='sigmoid')(dropout2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(X_train_text, y_train, vocab_size, embedding_dim, max_length, epochs=5, batch_size=32):
    model = build_cnn_model(vocab_size, embedding_dim, max_length)
    # Tính toán trọng số lớp
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Sử dụng trọng số lớp cho CNN: {class_weight_dict}")
    model.fit(X_train_text, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict)
    return model

def extract_cnn_features(model, X_text):
    feature_extractor_model = Model(
        inputs=model.input,
        outputs=model.get_layer('feature_extractor').output
    )
    features = feature_extractor_model.predict(X_text)
    return features
