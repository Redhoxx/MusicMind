import librosa
import soundfile as sf
import os
from pydub import AudioSegment
import numpy as np
import pandas as pd
from tqdm import tqdm


class AudioFeatureExtractor:
    def __init__(self, raw_dir, processed_dir, segment_length=10):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.segment_length = segment_length

        os.makedirs(self.processed_dir, exist_ok=True)

        self.split_audio()
        self.audio_data = self.load_audio_files()

    def split_audio(self):
        for filename in tqdm(os.listdir(self.raw_dir), desc="Splitting audios"):
            if filename.endswith((".mp3", ".wav", ".flac", ".m4a")):
                file_path = os.path.join(self.raw_dir, filename)

                temp_wav_file = ""
                try:
                    if filename.endswith(".m4a"):
                        audio = AudioSegment.from_file(file_path, format="m4a")
                        temp_wav_file = os.path.join(self.processed_dir, "temp.wav")
                        audio.export(temp_wav_file, format="wav")
                        file_path = temp_wav_file

                    y, sr = librosa.load(file_path)

                    if filename.endswith(".m4a"):
                        os.remove(temp_wav_file)

                    segment_samples = int(self.segment_length * sr)
                    file_name = os.path.splitext(filename)[0]

                    for i in range(0, len(y) - segment_samples, segment_samples):
                        segment = y[i:i + segment_samples]
                        segment_file = os.path.join(self.processed_dir, f"{file_name}_{i // segment_samples}.wav")
                        sf.write(segment_file, segment, sr)

                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {filename}: {e}")

    def load_audio_files(self):
        audio_data = []
        for filename in tqdm(os.listdir(self.processed_dir), desc="Loading processed audios"):
            if filename.endswith((".wav")):
                file_path = os.path.join(self.processed_dir, filename)
                try:
                    y, sr = librosa.load(file_path)
                    audio_data.append((os.path.splitext(filename)[0], y, sr))
                except Exception as e:
                    print(f"Erreur lors du chargement de {filename}: {e}")
        return audio_data

    def extract_features(self):
        all_features = []
        for filename, y, sr in tqdm(self.audio_data, desc="Extracting features"):
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)

            genre = filename[:3]
            music_name = filename[4:]

            data = {
                'mfccs': mfccs.flatten(),
                'chroma': chroma.flatten(),
                'tempo': tempo,
                'zcr': np.mean(zcr),
                'genre': genre,
                'music_name': music_name
            }
            all_features.append(data)

        return pd.DataFrame(all_features)

    def dataframe_to_excel(self, df, output_filename="features.xlsx"):
        try:
            df.to_excel("../data/features/" + output_filename, index=False)
            print(f"DataFrame exporté avec succès vers {output_filename}")
        except Exception as e:
            print(f"Erreur lors de l'exportation du DataFrame: {e}")