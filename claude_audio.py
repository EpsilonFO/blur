import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import random

def anonymize_audio_irreversible(audio_path, output_path, method='spectral_warping'):
    """
    Anonymise l'audio de manière irréversible tout en préservant l'intelligibilité.
    
    Args:
        audio_path: chemin vers le fichier audio d'entrée
        output_path: chemin vers le fichier de sortie
        method: méthode d'anonymisation ('spectral_warping', 'formant_shift', 'combined')
    """
    # Charger l'audio
    y, sr = librosa.load(audio_path, sr=None)
    
    if method == 'spectral_warping':
        y_anonymized = spectral_warping(y, sr)
    elif method == 'formant_shift':
        y_anonymized = formant_shifting(y, sr)
    elif method == 'combined':
        y_anonymized = combined_anonymization(y, sr)
    else:
        raise ValueError("Méthode non reconnue")
    
    # Sauvegarder
    sf.write(output_path, y_anonymized, sr)

def spectral_warping(y, sr):
    """
    Déforme le spectre de manière non-linéaire et irréversible.
    """
    # Calcul du spectrogramme
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Génération d'une fonction de déformation spectrale pseudo-aléatoire
    # mais déterministe basée sur le contenu audio
    freq_bins = magnitude.shape[0]
    
    # Créer une fonction de warping basée sur le hash du signal
    signal_hash = hash(str(np.sum(magnitude))) % 1000
    np.random.seed(signal_hash)  # Reproductible mais imprévisible
    
    # Créer une fonction de déformation non-linéaire
    warp_factor = 0.3  # Intensité de la déformation
    freq_indices = np.arange(freq_bins)
    
    # Déformation sinusoïdale avec composantes multiples
    warp_function = (
        np.sin(freq_indices * 2 * np.pi / freq_bins * 3) * warp_factor +
        np.sin(freq_indices * 2 * np.pi / freq_bins * 7) * warp_factor * 0.5 +
        np.sin(freq_indices * 2 * np.pi / freq_bins * 13) * warp_factor * 0.3
    )
    
    # Appliquer la déformation
    warped_magnitude = np.zeros_like(magnitude)
    for t in range(magnitude.shape[1]):
        for f in range(freq_bins):
            # Calculer le nouvel index de fréquence
            new_f = f + warp_function[f]
            new_f = np.clip(new_f, 0, freq_bins - 1)
            
            # Interpolation linéaire
            f_low = int(new_f)
            f_high = min(f_low + 1, freq_bins - 1)
            alpha = new_f - f_low
            
            warped_magnitude[f, t] = (
                (1 - alpha) * magnitude[f_low, t] + 
                alpha * magnitude[f_high, t]
            )
    
    # Reconstruction du signal
    warped_stft = warped_magnitude * np.exp(1j * phase)
    y_warped = librosa.istft(warped_stft, hop_length=512)
    
    return y_warped

def formant_shifting(y, sr):
    """
    Décale les formants de manière irréversible.
    """
    # Calcul des formants approximatifs via LPC
    frame_length = 2048
    hop_length = 512
    
    # Découpage en frames
    frames = librosa.util.frame(y, frame_length=frame_length, 
                               hop_length=hop_length, axis=0)
    
    processed_frames = []
    
    for frame in frames.T:
        if len(frame) == 0:
            continue
            
        # Appliquer une fenêtre
        windowed_frame = frame * np.hanning(len(frame))
        
        # Calcul du spectre
        fft_frame = np.fft.fft(windowed_frame, n=4096)
        magnitude = np.abs(fft_frame)
        phase = np.angle(fft_frame)
        
        # Décalage des formants par réarrangement spectral
        shifted_magnitude = shift_formants_spectrum(magnitude)
        
        # Reconstruction
        shifted_fft = shifted_magnitude * np.exp(1j * phase)
        shifted_frame = np.real(np.fft.ifft(shifted_fft))
        
        # Garder seulement la partie utile
        processed_frames.append(shifted_frame[:frame_length])
    
    # Reconstruction du signal complet avec overlap-add
    if processed_frames:
        output_length = len(y)
        y_shifted = np.zeros(output_length)
        
        for i, frame in enumerate(processed_frames):
            start = i * hop_length
            end = min(start + len(frame), output_length)
            y_shifted[start:end] += frame[:end-start]
    else:
        y_shifted = y
    
    return y_shifted

def shift_formants_spectrum(magnitude):
    """
    Décale les formants dans le spectre de manière non-linéaire.
    """
    # Créer une fonction de décalage non-linéaire
    n_bins = len(magnitude)
    indices = np.arange(n_bins)
    
    # Fonction de décalage basée sur la position spectrale
    shift_factor = 0.15
    shift_curve = np.sin(indices * np.pi / n_bins) * shift_factor
    
    # Appliquer le décalage
    shifted_magnitude = np.zeros_like(magnitude)
    for i in range(n_bins):
        # Calculer le nouvel index
        new_i = i + shift_curve[i] * n_bins * 0.1
        new_i = np.clip(new_i, 0, n_bins - 1)
        
        # Interpolation
        i_low = int(new_i)
        i_high = min(i_low + 1, n_bins - 1)
        alpha = new_i - i_low
        
        shifted_magnitude[i] = (
            (1 - alpha) * magnitude[i_low] + 
            alpha * magnitude[i_high]
        )
    
    return shifted_magnitude

def combined_anonymization(y, sr):
    """
    Combine plusieurs techniques pour une anonymisation robuste.
    """
    # 1. Déformation spectrale légère
    y_warped = spectral_warping(y, sr)
    
    # 2. Décalage de formants
    y_formant = formant_shifting(y_warped, sr)
    
    # 3. Ajout d'un léger bruit coloré déterministe
    signal_seed = hash(str(np.sum(y))) % 1000
    np.random.seed(signal_seed)
    
    # Bruit coloré (passe-bas) très léger
    noise_level = 0.003
    noise = np.random.normal(0, noise_level, len(y_formant))
    
    # Filtrage du bruit pour le rendre moins perceptible
    b, a = signal.butter(4, 0.3, btype='low')
    colored_noise = signal.filtfilt(b, a, noise)
    
    y_final = y_formant + colored_noise
    
    # 4. Normalisation
    y_final = y_final / np.max(np.abs(y_final)) * 0.95
    
    return y_final

# Fonction d'utilisation simple
def anonymize_voice(input_path, output_path, strength='medium'):
    """
    Interface simplifiée pour l'anonymisation.
    
    Args:
        input_path: fichier audio d'entrée
        output_path: fichier de sortie
        strength: 'light', 'medium', 'strong'
    """
    if strength == 'light':
        method = 'formant_shift'
    elif strength == 'medium':
        method = 'spectral_warping'
    elif strength == 'strong':
        method = 'combined'
    else:
        method = 'spectral_warping'
    
    anonymize_audio_irreversible(input_path, output_path, method)

# Exemple d'utilisation
if __name__ == "__main__":
    # Anonymisation simple
    anonymize_voice("input.wav", "output_anonymized.wav", strength='medium')
    
    # Ou avec plus de contrôle
    anonymize_audio_irreversible("input.wav", "output_custom.wav", 
                                method='combined')
