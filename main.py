import gradio as gr
import os
import shutil

from src.audio_predictor import AudioPredictor

audio_predictor = AudioPredictor(
                 ml_model_path="data/models/audio_classifier_model.keras",
                 model_dir="data/models",
                 raw_audio_dir="data/raw_to_predict")

os.makedirs(audio_predictor.raw_audio_dir, exist_ok=True)

def handle_audio_upload(audio_file):
    try:
        if not audio_file.endswith((".mp3", ".wav", ".flac")):
            return "Erreur: Veuillez télécharger un fichier audio valide (mp3, wav, flac)."

        destination_path = os.path.join(audio_predictor.raw_audio_dir, os.path.basename(audio_file))

        shutil.copy(audio_file, destination_path)

        predicted_genre = audio_predictor.predict_long_first_audio_in_dir()

        if predicted_genre:
            os.remove(destination_path)
            return f"Genre prédit: {predicted_genre}"
        else:
            return "Erreur lors de la prédiction."

    except Exception as e:
        return f"Erreur: {e}"

interface = gr.Interface(
    fn=handle_audio_upload,
    inputs=gr.Audio(type="filepath", label="Télécharger un fichier audio (mp3, wav, flac, m4a)"),
    outputs=gr.Textbox(label="Résultat de la prédiction"),
    title="Prédiction du genre musical",
    description="Télécharger un fichier audio pour prédire son genre."
)

interface.launch()