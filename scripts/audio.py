import os
import tempfile
from moviepy import VideoFileClip, AudioFileClip

from scripts.fourier import fourier_transform
from scripts.subtitles import generate_subtitles

def process_audio(video_path, output_path):
    """
    Process video audio: anonymize voices and generate subtitles.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path for final output video with anonymized audio
    """
    # Generate subtitle file path
    output_srt = f"{video_path}.srt"
    
    # Use temporary directory for intermediate audio files
    with tempfile.TemporaryDirectory() as tmpdir:
        video = VideoFileClip(video_path)
        audio_path = os.path.join(tmpdir, "original_audio.wav")
        # Create anonymized audio file path
        anonymized_audio_path = os.path.join(tmpdir, "anonymized_audio.wav")
        
        # Anonymize audio by pitch shifting
        fourier_transform(video, audio_path, anonymized_audio_path)
        
        # Generate subtitles from original audio (before pitch shifting for better accuracy)
        generate_subtitles(audio_path, output_srt)
        
        # Load anonymized audio and create final video
        new_audio = AudioFileClip(anonymized_audio_path)
        final_video = video.with_audio(new_audio)
        
        # Write final video with anonymized audio
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
