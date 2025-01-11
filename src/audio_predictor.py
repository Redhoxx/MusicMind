import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from src.audio_feature_extractor import AudioFeatureExtractor
from src.audio_classifier import AudioClassifier


class AudioPredictor:
    def __init__(self,
                 ml_model_path="../data/models/audio_classifier_model.keras",
                 model_dir="../data/models",
                 raw_audio_dir="../data/raw_to_predict"):

        self.ml_model_path = ml_model_path
        self.model_dir = model_dir
        self.raw_audio_dir = raw_audio_dir
        self.model = None
        self.feature_extractor = AudioFeatureExtractor()
        self.classifier = AudioClassifier()
        self.load_model()
        self.load_scaler()
        self.load_label_encoder()

    def load_model(self):
        try:
            self.model = load_model(self.ml_model_path)
            print(f"Modèle chargé avec succès depuis {self.ml_model_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")

    def load_scaler(self, scaler_filename="audio_scaler.pkl"):
        import pickle
        try:
            scaler_path = os.path.join(self.model_dir, scaler_filename)
            with open(scaler_path, 'rb') as f:
                self.classifier.scaler = pickle.load(f)
            print(f"Scaler chargé avec succès depuis {scaler_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du scaler: {e}")

    def load_label_encoder(self, encoder_filename="label_encoder.pkl"):
        import pickle
        try:
            encoder_path = os.path.join(self.model_dir, encoder_filename)
            with open(encoder_path, 'rb') as f:
                self.classifier.label_encoder = pickle.load(f)
            print(f"LabelEncoder chargé avec succès depuis {encoder_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du LabelEncoder: {e}")

    def predict_genre(self):
        if self.model is None:
            self.load_model()

        audio_file = self.find_first_audio_file(self.raw_audio_dir)
        if audio_file is None:
            print("Aucun fichier audio trouvé dans le répertoire.")
            return

        features = self.extract_features(audio_file)
        if features is None:
            print("Erreur lors de l'extraction des caractéristiques audio.")
            return

        input_data = self.prepare_input_data(features)

        prediction = self.model.predict(input_data)
        predicted_class = np.argmax(prediction)

        predicted_genre = self.classifier.label_encoder.inverse_transform([predicted_class])[0]

        print(f"Genre prédit pour {audio_file}: {predicted_genre}")
        return predicted_genre

    def find_first_audio_file(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith((".mp3", ".wav", ".flac")):
                return os.path.join(directory, filename)
        return None

    def extract_features(self, audio_file):
        y, sr = self.feature_extractor.load_audio_file(audio_file)
        if y is None:
            return None
        features = self.feature_extractor.extract_features_from_audio(y, sr)
        return features

    def prepare_input_data(self, features):
        df = pd.DataFrame([features])

        df['mfccs'] = (df['mfccs'].astype(str)
                       .str.replace('[', '', regex=False)
                       .str.replace(']', '', regex=False)
                       .apply(self.classifier.convert_to_list))
        df['chroma'] = (df['chroma'].astype(str)
                        .str.replace('[', '', regex=False)
                        .str.replace(']', '', regex=False)
                        .apply(self.classifier.convert_to_list))
        df['tempo'] = (df['tempo'].astype(str)
                       .str.replace('[', '', regex=False)
                       .str.replace(']', '', regex=False))

        X = df[['mfccs', 'chroma', 'tempo', 'zcr']].values
        X = np.array([np.concatenate((row[0], row[1], [row[2]], [row[3]])) for row in X])

        X = self.classifier.scaler.transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        return X

    def predict_long_audio(self, audio_file):
        y, sr = self.feature_extractor.load_audio_file(audio_file)
        if y is None:
            return None

        segment_samples = int(10 * sr)
        predictions = []
        for i in range(0, len(y) - segment_samples, segment_samples):
            segment = y[i:i + segment_samples]
            features = self.feature_extractor.extract_features_from_audio(segment, sr)
            input_data = self.prepare_input_data(features)
            prediction = self.model.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_genre = self.classifier.label_encoder.inverse_transform([predicted_class])[0]
            predictions.append(predicted_genre)

        from collections import Counter
        genre_counts = Counter(predictions)
        predicted_genre = genre_counts.most_common(1)[0][0]

        print(f"Genre prédit pour {audio_file}: {predicted_genre}")
        return predicted_genre

    def predict_long_first_audio_in_dir(self, audio_file=None):
        if audio_file is None:
            audio_file = self.find_first_audio_file(self.raw_audio_dir)
        if audio_file:
          return self.predict_long_audio(audio_file)
        else:
          return None

if __name__ == "__main__":
    AP = AudioPredictor()
    AP.predict_long_first_audio_in_dir()

