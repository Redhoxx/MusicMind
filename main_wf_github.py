import sys
from pytube import YouTube
from src.audio_predictor import AudioPredictor

audio_predictor = AudioPredictor(
                 ml_model_path="data/models/audio_classifier_model.keras",
                 model_dir="data/models",
                 raw_audio_dir="data/raw_to_predict")

def download_audio_from_youtube(video_url):
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(filename="youtube_audio.mp3")
        return audio_file
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'audio YouTube: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_url = sys.argv[1]

        audio_file = download_audio_from_youtube(video_url)

        if audio_file:
            predicted_genre = audio_predictor.predict_long_audio(audio_file)
            print(f"Genre prédit: {predicted_genre}")
        else:
            print("Erreur: Impossible de télécharger l'audio de la vidéo YouTube.")
    else:
        print("Erreur: Veuillez fournir l'URL de la vidéo YouTube en argument.")