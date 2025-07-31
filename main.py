import os
import argparse

from config import image_exts, video_exts
from scripts.image import process_image
from scripts.video import process_video, add_audio_to_video
from scripts.audio import process_audio
from scripts.tts_ai import process_tts
from scripts.subtitles import add_subtitles

def process(path, use_blur='pixelate', use_tts=False):
    # Generate file paths for different processing stages
    base, ext = os.path.splitext(path)
    tmp_video_path = f"{base}_tmp{ext}"  # Temporary video without audio
    blurred_output_path = f"{base}_blurred{ext}"  # Video with blurred faces and original audio
    anonymized_output_path = f"{base}_blurred_anonymized{ext}"
    subtitled_output_path = f"{base}_subtitled.srt"
    subtitled_output_video_path = f"{base}_subtitled{ext}"

    # Process image files
    if path.lower().endswith(image_exts):
        process_image(path, use_blur)
    # Process video files
    elif path.lower().endswith(video_exts):                        
        # Step 1: Process video frames
        process_video(path, tmp_video_path, use_blur=use_blur)
        # Step 2: Add original audio back
        add_audio_to_video(path, tmp_video_path, blurred_output_path)
        # Step 3: Create final version with anonymized audio
        if use_tts:
            process_tts(blurred_output_path, anonymized_output_path, subtitled_output_path, use_blur=use_blur)
        else:
            process_audio(blurred_output_path, anonymized_output_path, subtitled_output_path)
        # Clean up temporary file
        os.remove(tmp_video_path)
        add_subtitles(anonymized_output_path, subtitled_output_path, subtitled_output_video_path)

        print(f"✅ Final video generated : ", anonymized_output_path)

def main(path, use_blur='pixelate', use_tts=False):
    """
    Main processing function that handles files or directories based on input path.
    
    Args:
        path (str): Path to file or directory to process
        use_blur (bool): If True, uses Gaussian blur; if False, uses pixelation
    """

    # Check if path is a single file
    if os.path.isfile(path):
        process(path, use_blur=use_blur, use_tts=use_tts)
    # Check if path is a directory
    elif os.path.isdir(path):
        # Get list of all files in directory
        files = os.listdir(path)
        
        # Process each file in the directory
        for f in files:
            full_path = os.path.join(path, f)
            process(full_path, use_blur=use_blur, use_tts=use_tts)
    else:
        print("Invalid path. Please provide a valid image, video, or folder.")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Face blurring/pixelation for images and videos using YOLOv8.")
    parser.add_argument("path", help="Path to an image, video, or folder.")
    parser.add_argument("--blur", action="store_true", help="Use Gaussian blur instead of pixelation.")
    parser.add_argument("--tts", action="store_true", help="Use TTS for audio anonymization.")
    args = parser.parse_args()
    args.blur = 'blur' if args.blur else 'pixelate'
    args.use_tts = args.tts
    main(args.path, use_blur=args.blur, use_tts=args.use_tts)