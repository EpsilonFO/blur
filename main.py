import os
import argparse

from config import image_exts, video_exts
from scripts.image import process_image
from scripts.video import process_video, add_audio_to_video
from scripts.audio import process_audio




def main(path, use_blur='pixelate'):
    """
    Main processing function that handles files or directories based on input path.
    
    Args:
        path (str): Path to file or directory to process
        use_blur (bool): If True, uses Gaussian blur; if False, uses pixelation
    """

    # Check if path is a single file
    if os.path.isfile(path):
        # Process image files
        if path.lower().endswith(image_exts):
            process_image(path, use_blur)
        # Process video files
        elif path.lower().endswith(video_exts):
            input_path = path
            
            # Generate file paths for different processing stages
            base, ext = os.path.splitext(input_path)
            tmp_video_path = f"{base}_tmp{ext}"  # Temporary video without audio
            blurred_output_path = f"{base}_blurred{ext}"  # Video with blurred faces and original audio
            
            # Step 1: Process video frames (blur/pixelate faces)
            process_video(input_path, tmp_video_path, method=use_blur)
            
            # Step 2: Add original audio back to processed video
            add_audio_to_video(input_path, tmp_video_path, blurred_output_path)
            
            # Step 3: Create final version with anonymized audio and subtitles
            process_audio(blurred_output_path, f"{base}_blurred_anonymized{ext}")
            
            # Clean up temporary file
            os.remove(tmp_video_path)
            
            print(f"✅ Final video generated: {base}_blurred_anonymized{ext}")
    
    # Check if path is a directory
    elif os.path.isdir(path):
        # Get list of all files in directory
        files = os.listdir(path)
        
        # Process each file in the directory
        for f in files:
            full_path = os.path.join(path, f)
            
            # Process image files
            if f.lower().endswith(image_exts):
                process_image(full_path, use_blur)
            # Process video files
            elif f.lower().endswith(video_exts):
                input_path = full_path
                
                # Generate file paths for different processing stages
                base, ext = os.path.splitext(input_path)
                tmp_video_path = f"{base}_tmp{ext}"
                blurred_output_path = f"{base}_blurred{ext}"
                
                # Step 1: Process video frames
                process_video(input_path, tmp_video_path, use_blur=args.blur)
                
                # Step 2: Add original audio back
                add_audio_to_video(input_path, tmp_video_path, blurred_output_path)
                
                # Step 3: Create final version with anonymized audio
                process_audio(blurred_output_path, f"{base}_blurred_anonymized{ext}")
                
                # Clean up temporary file
                os.remove(tmp_video_path)
                
                print(f"✅ Vidéo finale générée : {base}_blurred_anonymized{ext}")
    else:
        print("Invalid path. Please provide a valid image, video, or folder.")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Face blurring/pixelation for images and videos using YOLOv8.")
    parser.add_argument("path", help="Path to an image, video, or folder.")
    parser.add_argument("--blur", action="store_true", help="Use Gaussian blur instead of pixelation.")
    args = parser.parse_args()
    args.blur = 'blur' if args.blur else 'pixelate'

    # Execute main function with parsed arguments
    main(args.path, use_blur=args.blur)