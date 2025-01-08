import os
import pandas as pd
from audio_feature_extractor import AudioFeatureExtractor
from audio_classifier import AudioClassifier
import argparse

def train(retrain=False):
    extractor = AudioFeatureExtractor()

    audio_segments = extractor.split_audio_for_training(raw_dir="../data/raw")

    classifier = AudioClassifier()

    if retrain:
        classifier.load_pretrained_model()
    else:
        classifier.load_data(audio_segments)
        classifier.preprocess_data()

    classifier.train_model()

    loss, accuracy = classifier.evaluate_model()
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    classifier.save_model()
    classifier.save_scaler()
    classifier.save_label_encoder()

    excel_file = "../data/raw/MusicMind - musics used for training.xlsx"
    df = pd.read_excel(excel_file)

    for _, _, filename in audio_segments:
        source_path = os.path.join("../data/raw", filename + ".mp3")  # Assurez-vous que l'extension est correcte
        os.remove(source_path)
        new_row = {'Nom du fichier': filename + ".mp3"}  # Assurez-vous que l'extension est correcte
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(excel_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner ou réentraîner le modèle de classification audio.")
    parser.add_argument("--retrain", action="store_true", help="Réentraîner le modèle existant.")
    args = parser.parse_args()

    train(retrain=args.retrain)