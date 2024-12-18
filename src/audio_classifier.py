import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class AudioClassifier:
    def __init__(self, data_path="../data/features/audio_features.csv", model_dir="../data/models"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None

        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        self.data = pd.read_csv(self.data_path, sep=',')

        self.data['mfccs'] = (self.data['mfccs'].astype(str)
                              .str.replace('[', '', regex=False)
                              .str.replace(']', '', regex=False)
                              .apply(self.convert_to_list))
        self.data['chroma'] = (self.data['chroma'].astype(str)
                               .str.replace('[', '', regex=False)
                               .str.replace(']', '', regex=False)
                               .apply(self.convert_to_list))
        self.data['tempo'] = (self.data['tempo'].astype(str)
                              .str.replace('[', '', regex=False)
                              .str.replace(']', '', regex=False))

    def preprocess_data(self):
        if self.data is None:
            self.load_data()

        X = self.data[['mfccs', 'chroma', 'tempo', 'zcr']].values
        y = self.data['genre'].values

        y = self.label_encoder.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = np.array([np.concatenate((row[0], row[1], [row[2]], [row[3]])) for row in self.X_train])
        self.X_test = np.array([np.concatenate((row[0], row[1], [row[2]], [row[3]])) for row in self.X_test])

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

    def build_model(self, lstm_units=128, dropout_rate=0.2, dense_units=32, num_dense_layers=2):
        self.model = Sequential()
        self.model.add(LSTM(lstm_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(dropout_rate))

        for _ in range(num_dense_layers):
            self.model.add(Dense(dense_units, activation='relu'))
            self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(len(np.unique(self.y_train)), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, epochs=50, batch_size=32):
        if self.model is None:
            self.build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.X_test, self.y_test), callbacks=[early_stopping])

    def evaluate_model(self):
        if self.model is None:
            print("Model not trained yet. Please call train_model() first.")
            return

        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, accuracy

    def convert_to_list(self, feature_string):
        return [float(value) for value in feature_string.split(';')]

    def save_model(self, model_filename="audio_classifier_model.h5"):
        if self.model is None:
            print("Aucun modèle à enregistrer. Veuillez d'abord entraîner un modèle.")
            return

        try:
            model_path = os.path.join(self.model_dir, model_filename)
            self.model.save(model_path)
            print(f"Modèle enregistré avec succès dans {model_path}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement du modèle: {e}")