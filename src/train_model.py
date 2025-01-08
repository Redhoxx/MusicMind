from audio_feature_extractor import AudioFeatureExtractor
from audio_classifier import AudioClassifier
import argparse

def train(retrain=False):
    # Initialiser l'extracteur de caractéristiques audio
    extractor = AudioFeatureExtractor()

    # Charger les segments audio en mémoire
    audio_segments = extractor.split_audio_for_training(raw_dir="../data/raw")

    # Initialiser le classificateur audio
    classifier = AudioClassifier()

    if retrain:
        # Charger le modèle pré-entraîné, le scaler et le label encoder
        classifier.load_pretrained_model()
    else:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner ou réentraîner le modèle de classification audio.")
    parser.add_argument("--retrain", action="store_true", help="Réentraîner le modèle existant.")
    args = parser.parse_args()

    train(retrain=args.retrain)