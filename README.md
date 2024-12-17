AI project 2024 

Bruce L'horset, Lucas Jendzio-Verdasca, Guillaume Cappe de Baillon, Thibaud Barberon

This project aims to recognize a type of music with a music file. This is possible by creating an AI that will analyze the audio signal of the song to get some indicators of the song using the librosa library with custom functions to compute the following features:

MFCCs (Mel Frequency Cepstral Coefficients)
Chroma features
Tempo (beat estimation)
Zero Crossing Rate (ZCR)

Each feature serves a specific purpose in describing the audio content. 

- MFCCs are widely used in audio signal processing, especially for speech and music analysis. They capture the spectral characteristics of an audio signal on the Mel scale, which approximates how humans perceive pitch and frequency.
- Chroma features represent the 12 semitones (chromatic scale) of music. They are useful for capturing the harmonic and tonal content of the audio signal.
- Tempo extraction identifies the beats per minute (BPM) of the audio, which is particularly important for rhythmic analysis in music.
- ZCR measures the rate at which the audio signal crosses the zero amplitude axis. It is a simple yet effective feature for distinguishing between voiced and unvoiced sounds, which is helpful in speech recognition.

These are the inputs that our AI will take for every song that we will compute in order to dertimine its type of music. 

We then have 4 inputs and 1 output. 
