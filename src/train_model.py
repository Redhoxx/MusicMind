import os
import argparse
import pandas as pd
from audio_feature_extractor import AudioFeatureExtractor
from audio_classifier import AudioClassifier

def train(retrain=False, github_actions=False):
    # Initialiser l'extracteur de caractéristiques audio
    extractor = AudioFeatureExtractor()

    # Déterminer le chemin du répertoire raw en fonction de l'environnement
    raw_dir = "../data/raw"  # Chemin par défaut pour l'exécution locale
    if github_actions:
        raw_dir = os.path.join(os.environ.get('GITHUB_WORKSPACE'), raw_dir)

    # Charger les segments audio en mémoire
    audio_segments = extractor.split_audio_for_training(raw_dir=raw_dir)

    # Afficher le nombre de segments audio chargés
    print(f"Nombre de segments audio chargés : {len(audio_segments)}")

    # Initialiser le classificateur audio
    classifier = AudioClassifier(model_dir="../data/models")

    if retrain:
        # Charger le modèle pré-entraîné, le scaler et le label encoder
        classifier.load_pretrained_model()

    # Charger les données à partir des segments audio
    classifier.load_data(audio_segments)

    # Prétraiter les données
    classifier.preprocess_data()

    # Construire et entraîner le modèle (ou réentraîner si retrain=True)
    classifier.train_model()

    # Évaluer le modèle
    loss, accuracy = classifier.evaluate_model()
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Enregistrer le modèle, le scaler et le label encoder
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
    parser = argparse.ArgumentParser(description="Entraîner ou réentraîner le modèle de classification audio.")
    parser.add_argument("--retrain", action="store_true", help="Réentraîner le modèle existant.")
    parser.add_argument("--github_actions", action="store_true", help="Indique si le script est exécuté dans GitHub Actions.")
    args = parser.parse_args()

    train(retrain=args.retrain, github_actions=args.github_actions)