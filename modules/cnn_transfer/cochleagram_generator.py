import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal
import multiprocessing
from pycochleagram import cochleagram as cgram
from PIL import Image

input_folder = "cnnMusicSeg/music1_Seg"
output_folder = "cnnMusicSeg/pngMusic1_Seg"

os.makedirs(output_folder, exist_ok=True)

def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.Resampling.LANCZOS)
    return np.array(resized_image)

def plot_cochleagram(cochleagram, output_path):
    plt.figure(figsize=(6, 3))
    plt.imshow(cochleagram, origin='lower', cmap=plt.cm.Blues, aspect='auto')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_cochleagram(wav_f, sr):
    n, low_lim, hi_lim = 50, 20, 8000
    sample_factor, pad_factor = 4, 2
    downsample = 250 if sr % 250 == 0 else 200
    nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
  
    num_samples = int(np.ceil(len(wav_f) / (sr / downsample)) * (sr / downsample))
    wav_f = librosa.util.fix_length(wav_f, size=num_samples)

    if sr % downsample != 0:
        wav_f = scipy.signal.resample_poly(wav_f, downsample, sr)
        sr = downsample

    c_gram = cgram.cochleagram(wav_f, sr, n, low_lim, hi_lim, 
                               sample_factor, pad_factor, downsample,
                               nonlinearity, fft_mode, ret_mode, strict=True)

    c_gram_rescaled = 255 * (1 - ((np.max(c_gram) - c_gram) / np.ptp(c_gram)))

    c_gram_reshape = resample(c_gram_rescaled, (256, 256))

    return c_gram_reshape

def process_single_wav(input_wav_path, output_image_path):
    try:
        wav_f, sr = librosa.load(input_wav_path, sr=16000)
        cochleagram_img = generate_cochleagram(wav_f, sr)
        plot_cochleagram(cochleagram_img, output_image_path)
        print(f"finished: {os.path.basename(input_wav_path)} -> {output_image_path}")
    except Exception as e:
        print(f"failed: {os.path.basename(input_wav_path)} - {e}")

def process_wav_folder_parallel(input_folder, output_folder, num_workers=1):
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    print(f" {len(wav_files)}  WAV files，process using {num_workers} cores...")

    file_pairs = [(os.path.join(input_folder, f), os.path.join(output_folder, f.replace(".wav", ".png")))
                  for f in wav_files]

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(process_single_wav, file_pairs)

    print("All done！")

if __name__ == "__main__":
    process_wav_folder_parallel(input_folder, output_folder, num_workers=5)
