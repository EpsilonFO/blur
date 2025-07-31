import cv2
import os
import subprocess
from tqdm import tqdm

from scripts.detect import detect_and_anonymize_faces

def process_video(input_path, output_path, use_blur='pixelate'):
    """
    Treat a video to blur or pixelate faces.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir la vidéo {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc=f"Traitement de {os.path.basename(input_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_and_anonymize_faces(frame, use_blur)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Vidéo sauvegardée : {output_path}")


def add_audio_to_video(original_video, video_no_audio, output_with_audio):
    """
    Add original audio track to a processed video file using FFmpeg.
    
    Args:
        original_video (str): Path to original video with audio
        video_no_audio (str): Path to processed video without audio
        output_with_audio (str): Path for final video with audio
    """
    # Build FFmpeg command to combine video and audio streams
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', video_no_audio,  # Input processed video
        '-i', original_video,  # Input original video (for audio)
        '-c:v', 'copy',  # Copy video stream without re-encoding
        '-c:a', 'aac',   # Encode audio as AAC
        '-map', '0:v:0',  # Map video from first input
        '-map', '1:a:0',  # Map audio from second input
        output_with_audio
    ]
    
    # Execute FFmpeg command (suppress output for cleaner console)
    subprocess.run(command)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
