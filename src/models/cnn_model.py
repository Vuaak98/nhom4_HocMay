import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense

def build_cnn_model(vocab_size, embedding_dim, max_length):
    inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, embedding_dim)(inputs)
    conv_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
    pooling_layer = GlobalMaxPooling1D()(conv_layer)
    feature_extractor_layer = Dense(64, activation='relu', name='feature_extractor')(pooling_layer)
    outputs = Dense(1, activation='sigmoid')(feature_extractor_layer)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(X_train_text, y_train, vocab_size, embedding_dim, max_length, epochs=5, batch_size=32):
    model = build_cnn_model(vocab_size, embedding_dim, max_length)
    model.fit(X_train_text, y_train, epochs=epochs, batch_size=batch_size)
    return model

def extract_cnn_features(model, X_text):
    feature_extractor_model = Model(
        inputs=model.input,
        outputs=model.get_layer('feature_extractor').output
    )
    features = feature_extractor_model.predict(X_text)
    return features
