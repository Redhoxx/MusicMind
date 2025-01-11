import os
import argparse
from audio_feature_extractor import AudioFeatureExtractor
from audio_classifier import AudioClassifier

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

def train(github_actions=False):

    extractor = AudioFeatureExtractor()
    raw_dir = "data/raw"
    if github_actions:
        raw_dir = os.path.join(os.environ.get('GITHUB_WORKSPACE'), raw_dir)
    else:
        raw_dir = os.path.join("..", "data", "raw")
    audio_segments = extractor.split_audio_for_training(raw_dir=raw_dir)
    print(f"Nombre de segments audio chargés : {len(audio_segments)}")

    classifier = AudioClassifier(model_dir="../data/models")
    classifier.load_data(audio_segments)
    classifier.preprocess_data()
    classifier.train_model()

    loss, accuracy = classifier.evaluate_model()
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    classifier.save_model()
    classifier.save_scaler()
    classifier.save_label_encoder()

    # --- Gestion des fichiers musicaux ---
    for filename in os.listdir(raw_dir):
        if filename.endswith((".mp3", ".flac", ".wav")) and filename != "MusicMind - musics used for training.xlsx":
            file_path = os.path.join(raw_dir, filename)
            os.remove(file_path)
    # --- Fin de la gestion des fichiers musicaux ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le modèle de classification audio.")
    parser.add_argument("--github_actions", action="store_true", help="Indique si le script est exécuté dans GitHub Actions.")
    args = parser.parse_args()

    train(github_actions=args.github_actions)