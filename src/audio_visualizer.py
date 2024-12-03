import matplotlib.pyplot as plt
import librosa

class AudioVisualizer:
    def __init__(self, audio_data):
        self.audio_data = audio_data

    def plot_waveform(self, index=0):
        try:
            music_name = self.audio_data[index][0]
            y = self.audio_data[index][1]
            sr = self.audio_data[index][2]

            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Forme d\'onde audio de {music_name} (index {index})')
            plt.xlabel('Temps (s)')
            plt.ylabel('Amplitude')
            plt.show()
        except IndexError:
            print(f"Erreur: Index {index} invalide. La liste audio_data contient {len(self.audio_data)} éléments.")