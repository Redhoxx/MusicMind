import gradio as gr
import os
import shutil
from src.audio_predictor import AudioPredictor

# Initialiser l'AudioPredictor
audio_predictor = AudioPredictor()

# S'assurer que le répertoire existe
os.makedirs(audio_predictor.raw_audio_dir, exist_ok=True)

def handle_audio_upload(audio_file):
    # Définir le chemin de destination
    destination_path = os.path.join(audio_predictor.raw_audio_dir, os.path.basename(audio_file))
    
    # Copier le fichier audio dans le répertoire de destination
    shutil.copy(audio_file, destination_path)
    
    # Appeler la fonction pour prédire le genre
    predicted_genre = audio_predictor.predict_long_first_audio_in_dir()
    
    if predicted_genre:
        return f"Genre prédit: {predicted_genre}"
    else:
        return "Aucun fichier audio trouvé ou erreur lors de la prédiction."

# Créer une interface Gradio
interface = gr.Interface(
    fn=handle_audio_upload,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(label="Résultat de la prédiction")
)

interface.launch()