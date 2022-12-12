import os
import sys
import librosa
import soundfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

def gain_normalize(waveform):
    amplitude = np.iinfo(np.int16).max
    waveform = np.int16(0.8 * amplitude * waveform / np.max(np.abs(waveform)))

    return waveform

def normalize_audios(audio_dir, save_dir):
    sr = 16000
    audio_dir = Path(audio_dir).expanduser().absolute()
    save_dir = Path(save_dir).expanduser().absolute()
    audio_file_paths = librosa.util.find_files(audio_dir.as_posix(), ext="wav")

    for audio_file_path in tqdm(audio_file_paths):
        basename = os.path.basename(audio_file_path)
        audio_wav, _ = librosa.load(audio_file_path, sr=sr, mono=False)
        normalized_wav = gain_normalize(audio_wav)
        # print(audio_wav[:10])
        # print(normalized_wav[:10])
        # break
        soundfile.write((save_dir / basename).as_posix(), normalized_wav.T, samplerate=sr)
        
if __name__ == "__main__":
    audio_dir = sys.argv[1]
    save_dir = sys.argv[2]
    normalize_audios(audio_dir, save_dir)