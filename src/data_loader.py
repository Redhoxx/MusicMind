import librosa
import soundfile as sf
import os

def split_audio(file_path, output_dir, segment_length=10):

      y, sr = librosa.load(file_path)
      segment_samples = int(segment_length * sr)

      file_name = os.path.splitext(os.path.basename(file_path))[0]

      for i in range(0, len(y) - segment_samples, segment_samples):
          segment = y[i:i + segment_samples]
          segment_file = os.path.join(output_dir, f"{file_name}_{i//segment_samples}.wav")
          sf.write(segment_file, segment, sr)

if __name__ == "__main__":
      raw_dir = "../data/raw"
      processed_dir = "../data/processed"
      audio_file = os.path.join(raw_dir, "rock_highoctane.mp3")

      os.makedirs(processed_dir, exist_ok=True)

      split_audio(audio_file, processed_dir)