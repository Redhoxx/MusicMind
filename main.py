import gradio as gr

# Fonction de traitement fictive
def process_audio(audio_file):
    # Ici, vous ajouterez votre logique de traitement
    # Pour l'instant, nous retournons un message fictif
    return "Prédiction du modèle ici"

# Créer une interface Gradio
interface = gr.Interface(
    fn=process_audio,  # La fonction de traitement
    inputs=gr.Audio(type="filepath"),  # Boîte de dépôt pour l'audio
    outputs=gr.Textbox(label="Prédiction du modèle")  # Espace pour afficher la prédiction
)

# Lancer l'interface
interface.launch()