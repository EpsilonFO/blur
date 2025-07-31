"""
Script to extract audio from a .mov video file and save it as an .mp3 file.
"""

import os
import argparse
from moviepy import VideoFileClip

def extract_audio_from_mov(file_path):
    # Nom du fichier sans extension
    base_name = os.path.splitext(file_path)[0]
    # Chemin de sortie pour le fichier audio
    output_audio_path = f"{base_name}.mp3"
    
    # Chargement du fichier vidéo
    video = VideoFileClip(file_path)
    
    # Extraction et sauvegarde de l'audio
    video.audio.write_audiofile(output_audio_path)
    print(f"Audio extrait et sauvegardé sous : {output_audio_path}")

parser = argparse.ArgumentParser(description="Face blurring/pixelation for images and videos using YOLOv8.")
parser.add_argument("path", help="Path to an image, video, or folder.")
args = parser.parse_args()

# Execute main function with parsed arguments
extract_audio_from_mov(args.path)
