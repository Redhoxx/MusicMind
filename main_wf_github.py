import sys

from src.audio_predictor import AudioPredictor

audio_predictor = AudioPredictor(
                 ml_model_path="data/models/audio_classifier_model.keras",
                 model_dir="data/models",
                 raw_audio_dir="data/raw_to_predict")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        predicted_genre = audio_predictor.predict_long_first_audio_in_dir(audio_file)
    else:
        print("Erreur : Veuillez fournir le chemin du fichier audio en argument.")