import librosa
import os
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, segment_length=10):
        self.segment_length = segment_length

    def split_audio_for_training(self, raw_dir):
        audio_segments = []
        for filename in tqdm(os.listdir(raw_dir), desc="Splitting audios for training"):
            if filename.endswith((".mp3", ".wav", ".flac", ".m4a")):
                file_path = os.path.join(raw_dir, filename)
                try:
                    temp_wav_file = ""
                    if filename.endswith(".m4a"):
                        audio = AudioSegment.from_file(file_path, format="m4a")
                        temp_wav_file = "temp.wav"  # Enregistrer dans le répertoire courant
                        audio.export(temp_wav_file, format="wav")
                        file_path = temp_wav_file

                    try:
                        y, sr = librosa.load(file_path)
                    except (librosa.util.exceptions.ParameterError, Exception) as e:
                        print("C'EST LE LOAD DE LIBROSA QUI PARLE")
                        print(f"Erreur lors du traitement du fichier {filename}: {e}")
                        continue  # Passer au fichier suivant

                    if filename.endswith(".m4a"):
                        os.remove(temp_wav_file)

                    segment_samples = int(self.segment_length * sr)
                    file_name = os.path.splitext(filename)[0]

                    for i in range(0, len(y) - segment_samples, segment_samples):
                        segment = y[i:i + segment_samples]
                        audio_segments.append((segment, sr, file_name))  # Stocker le segment, sr et le nom du fichier

                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {filename}: {e}")
        return audio_segments  # Retourner la liste des segments audio

    def load_audio_file(self, file_path):
        try:
            y, sr = librosa.load(file_path)
            return y, sr
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")

    def extract_features_from_audio(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        features = {
            'mfccs': "[" + ';'.join(map(str, mfccs.flatten())) + "]",
            'chroma': "[" + ';'.join(map(str, chroma.flatten())) + "]",
            'tempo': tempo,
            'zcr': np.mean(zcr)
        }
        return features

    def dataframe_to_csv(self, df, output_folderpathname="../data/features/", output_filename="features.csv"):
        try:
            df.to_csv(output_folderpathname + output_filename, index=False)
            print(f"DataFrame exporté avec succès vers {output_filename}")
        except Exception as e:
            print(f"Erreur lors de l'exportation du DataFrame: {e}")