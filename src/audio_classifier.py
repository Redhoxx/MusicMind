import os
import pandas as pd
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

import pickle

class AudioClassifier:
    def __init__(self, model_dir="../data/models"):
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

    def load_data(self, audio_segments):  # Modifier la méthode load_data pour accepter les segments audio

        print(f"Je passe dans load_data avec {len(audio_segments)} segments")

        data = []  # Créer une liste pour stocker les données
        for segment, sr, file_name in audio_segments:
            features = self.extract_features_from_audio(segment, sr)  # Extraire les caractéristiques du segment
            features['genre'] = file_name.split('_')[0]  # Extraire le genre du nom du fichier
            data.append(features)  # Ajouter les caractéristiques à la liste

        self.data = pd.DataFrame(data)  # Créer un DataFrame à partir de la liste

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
            print("Error: Data not loaded. Please call load_data() first.")
            return

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

    @tf.function
    def train_model(self, epochs=100, batch_size=32):
        if self.model is None:
            self.build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Convertir le dataset en une liste
        data_list = list(tf.data.Dataset)  # Remplacez 'dataset' par le nom de votre objet tf.data.Dataset

        # Convertir la liste en tableaux NumPy
        X_train = np.array([item[0] for item in data_list])
        y_train = np.array([item[1] for item in data_list])

        # Entraîner le modèle avec les tableaux NumPy
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.X_test, self.y_test), callbacks=[early_stopping])

    def evaluate_model(self):
        if self.model is None:
            print("Model not trained yet. Please call train_model() first.")
            return

        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, accuracy

    def convert_to_list(self, feature_string):
        return [float(value) for value in feature_string.split(';')]

    def save_scaler(self, scaler_filename="audio_scaler.pkl"):
        import pickle
        try:
            scaler_path = os.path.join(self.model_dir, scaler_filename)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler enregistré avec succès dans {scaler_path}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement du scaler: {e}")

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

    def save_label_encoder(self, encoder_filename="label_encoder.pkl"):
        import pickle
        try:
            encoder_path = os.path.join(self.model_dir, encoder_filename)
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"LabelEncoder enregistré avec succès dans {encoder_path}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement du LabelEncoder: {e}")

    def extract_features_from_audio(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        features = {
            'mfccs': "[" + ';'.join(map(str, mfccs.flatten())) + "]",
            'chroma': "[" + ';'.join(map(str, chroma.flatten())) + "]",
            'tempo': tempo,
            'zcr': np.mean(zcr)
        }
        return features

    def load_pretrained_model(self, model_filename="audio_classifier_model.h5",
                              scaler_filename="audio_scaler.pkl",
                              encoder_filename="label_encoder.pkl"):
        try:
            # Construire les chemins absolus si nécessaire
            if os.environ.get('GITHUB_WORKSPACE'):
                model_path = os.path.join(os.environ.get('GITHUB_WORKSPACE'), self.model_dir, model_filename)
                scaler_path = os.path.join(os.environ.get('GITHUB_WORKSPACE'), self.model_dir, scaler_filename)
                encoder_path = os.path.join(os.environ.get('GITHUB_WORKSPACE'), self.model_dir, encoder_filename)
            else:
                model_path = os.path.join(self.model_dir, model_filename)
                scaler_path = os.path.join(self.model_dir, scaler_filename)
                encoder_path = os.path.join(self.model_dir, encoder_filename)

            self.model = load_model(model_path)
            print(f"Modèle chargé avec succès depuis {model_path}")

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler chargé avec succès depuis {scaler_path}")

            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"LabelEncoder chargé avec succès depuis {encoder_path}")

        except Exception as e:
            print(f"Erreur lors du chargement du modèle, du scaler ou du LabelEncoder: {e}")