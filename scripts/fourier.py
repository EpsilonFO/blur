import numpy as np
from scipy.io import wavfile

from config import band_size, shift_amount

def shift_frequencies(spectrum):
    """Décale les fréquences dans le spectre FFT."""
    shifted = np.roll(spectrum, shift_amount)
    if shift_amount > 0:
        shifted[:shift_amount] = 0
    else:
        shifted[shift_amount:] = 0
    return shifted

def permute_frequency_bands(spectrum):
    """Permute les bandes de fréquences dans le spectre FFT."""
    permuted = spectrum.copy()
    num_bands = len(spectrum) // band_size
    for i in range(0, num_bands - 1, 2):
        start1 = i * band_size
        start2 = (i + 1) * band_size
        vect = np.array((spectrum[start2:start2 + band_size] + spectrum[start1:start2]) / 2)
        vect = np.concatenate((vect, vect))
        permuted[start1:start1 + band_size], permuted[start2:start2 + band_size] = vect[:band_size], vect[band_size:]
    return permuted

def fourier_transform(video, extracted_audio_path, output_audio_path):
    """Extrait l'audio d'une vidéo, applique une transformation FFT et sauvegarde le résultat."""
    # Charger la vidéo et extraire l'audio
    video.audio.write_audiofile(extracted_audio_path)

    # Charger l'audio
    rate, data = wavfile.read(extracted_audio_path)

    # Si stéréo, ne garder qu'un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Appliquer la FFT
    fft_data = np.fft.fft(data)

    # Décalage fréquentiel
    fft_shifted = shift_frequencies(fft_data)

    # Permutation de bandes
    fft_modified = permute_frequency_bands(fft_shifted)

    # Reconstruction avec IFFT
    modified_data = np.fft.ifft(fft_modified).real

    # Normalisation et conversion
    modified_data = np.int16(modified_data / np.max(np.abs(modified_data)) * 32767)

    # Sauvegarde
    wavfile.write(output_audio_path, rate, modified_data)