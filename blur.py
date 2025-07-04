#This file is part of Blur.

#Blur is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#Blur is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.

#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Import necessary libraries
import os
import cv2
import argparse
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import whisper
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip
from ultralytics import YOLO

# Parameters
pixel_size = 10  # Size of pixel blocks for pixelation
blur_size = 9    # Kernel size for Gaussian blur
model = YOLO("utils/yolov8n-face-lindevs.onnx", task="detect")  # Load YOLO model for face detection
image_exts = (".jpg", ".jpeg", ".png")  # Supported image formats
video_exts = (".mp4", ".mov", ".m4v")   # Supported video formats

def blur_faces(img, use_gaussian):
    """
    Blur or pixelate faces detected in an image using YOLO face detection.
    
    Args:
        img (numpy.ndarray): Input image as OpenCV array
        use_gaussian (bool): If True, applies Gaussian blur; if False, applies pixelation
    
    Returns:
        numpy.ndarray: Image with blurred/pixelated faces
    """
    # Run YOLO face detection on the input image
    results = model(img)
    
    # Iterate through all detection results
    for result in results:
        # Process each detected face bounding box
        for box in result.boxes:
            # Extract bounding box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract the face region from the image
            face = img[y1:y2, x1:x2]
            
            # Apply the selected anonymization method
            if use_gaussian:
                # Apply Gaussian blur with specified kernel size and standard deviation
                blurred_face = cv2.GaussianBlur(face, (blur_size, blur_size), 30)
            else:
                # Apply pixelation by downscaling then upscaling
                h, w = face.shape[:2]  # Get face dimensions
                # Downscale to create pixelated effect
                temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
                # Upscale back to original size with nearest neighbor to maintain pixelation
                blurred_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Replace the original face region with the anonymized version
            img[y1:y2, x1:x2] = blurred_face
    
    return img

def process_image(image_path, use_gaussian):
    """
    Process a single image file to blur/pixelate faces and save the result.
    
    Args:
        image_path (str): Path to the input image file
        use_gaussian (bool): If True, uses Gaussian blur; if False, uses pixelation
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Apply face blurring/pixelation
    img = blur_faces(img, use_gaussian)
    
    # Generate output filename by adding "_blured" suffix
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_blured{ext}"
    
    # Save the processed image
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

def pixelate_face(image, x, y, w, h, blocks=pixel_size):
    """
    Apply pixelation effect to a specific face region in an image.
    
    Args:
        image (numpy.ndarray): Input image
        x (int): X coordinate of face region
        y (int): Y coordinate of face region
        w (int): Width of face region
        h (int): Height of face region
        blocks (int): Number of pixel blocks for pixelation effect
    
    Returns:
        numpy.ndarray: Image with pixelated face region
    """
    # Extract the face region from the image
    face = image[y:y+h, x:x+w]
    (h, w) = face.shape[:2]
    
    # Create step arrays for dividing face into blocks
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    
    # Iterate through each block in the grid
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # Define the current block boundaries
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            
            # Extract the region of interest (current block)
            roi = face[startY:endY, startX:endX]
            
            # Calculate the mean color of the block
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            
            # Fill the entire block with the mean color
            cv2.rectangle(face, (startX, startY), (endX, endY), (B, G, R), -1)
    
    # Replace the original face region with the pixelated version
    image[y:y+h, x:x+w] = face
    return image

def blur_face(image, x, y, w, h, ksize=(blur_size, blur_size)):
    """
    Apply Gaussian blur to a specific face region in an image.
    
    Args:
        image (numpy.ndarray): Input image
        x (int): X coordinate of face region
        y (int): Y coordinate of face region
        w (int): Width of face region
        h (int): Height of face region
        ksize (tuple): Kernel size for Gaussian blur
    
    Returns:
        numpy.ndarray: Image with blurred face region
    """
    # Extract the face region
    face = image[y:y+h, x:x+w]
    
    # Apply Gaussian blur to the face region
    face = cv2.GaussianBlur(face, ksize, 0)
    
    # Replace the original face region with the blurred version
    image[y:y+h, x:x+w] = face
    return image

def anonymize_audio_with_librosa(audio_path, output_path, pitch_shift_steps=-4):
    """
    Anonymize audio by shifting the pitch to make voices less recognizable.
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str): Path for output anonymized audio file
        pitch_shift_steps (int): Number of semitones to shift pitch (negative for lower pitch)
    """
    # Load audio file using librosa (sr=None preserves original sample rate)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Apply pitch shifting to anonymize the audio
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_shift_steps)
    
    # Save the pitch-shifted audio to output file
    sf.write(output_path, y_shifted, sr)

def generate_subtitles(audio_path, srt_path):
    """
    Generate SRT subtitle file from audio using OpenAI Whisper speech recognition.
    
    Args:
        audio_path (str): Path to input audio file
        srt_path (str): Path for output SRT subtitle file
    """
    # Load Whisper model (base model provides good balance of speed and accuracy)
    model = whisper.load_model("base")
    
    # Transcribe the audio file
    result = model.transcribe(audio_path)
    
    # Write transcription results to SRT file
    with open(srt_path, "w", encoding="utf-8") as f:
        # Process each transcribed segment
        for i, segment in enumerate(result["segments"]):
            # Extract timing and text information
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            # Write SRT format: sequence number, timestamps, text, blank line
            f.write(f"{i+1}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

def format_timestamp(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timestamp string
    """
    # Calculate hours, minutes, seconds, and milliseconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    # Return formatted timestamp string
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def process_video(input_path, output_path, use_blur=False):
    """
    Process a video file to blur or pixelate faces in each frame.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path for output processed video file
        use_blur (bool): If True, uses blur; if False, uses pixelation
    """
    # Open video capture object
    cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame with progress bar
    for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(input_path)}"):
        # Read next frame
        ret, frame = cap.read()
        if not ret:
            break  # End of video or read error
        
        # Detect faces in current frame
        results = model.predict(source=frame, verbose=False)
        
        # Process each detection result
        for result in results:
            # Apply anonymization to each detected face
            for box in result.boxes.xyxy:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                
                # Apply selected anonymization method
                if use_blur:
                    frame = blur_face(frame, x1, y1, w, h)
                else:
                    frame = pixelate_face(frame, x1, y1, w, h)
        
        # Write processed frame to output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

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
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        # Load video using MoviePy
        video = VideoFileClip(video_path)
        
        # Extract original audio to temporary file
        audio_path = os.path.join(tmpdir, "original_audio.wav")
        video.audio.write_audiofile(audio_path)
        
        # Create anonymized audio file path
        anonymized_audio_path = os.path.join(tmpdir, "anonymized_audio.wav")
        
        # Anonymize audio by pitch shifting
        anonymize_audio_with_librosa(audio_path, anonymized_audio_path)
        
        # Debug information
        print("audio path :", audio_path, " ; output srt :", output_srt, " ; video path :", video_path)
        
        # Generate subtitles from original audio (before pitch shifting for better accuracy)
        generate_subtitles(audio_path, output_srt)
        
        # Load anonymized audio and create final video
        new_audio = AudioFileClip(anonymized_audio_path)
        final_video = video.with_audio(new_audio)
        
        # Write final video with anonymized audio
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

def main(path, use_blur=False):
    """
    Main processing function that handles files or directories based on input path.
    
    Args:
        path (str): Path to file or directory to process
        use_blur (bool): If True, uses Gaussian blur; if False, uses pixelation
    """
    # Set up argument parser for command line usage
    parser = argparse.ArgumentParser(description="Face blurring and audio anonymization with subtitles.")
    parser.add_argument("video_path", help="Path to the video to process.")
    parser.add_argument("--blur", action="store_true", help="Use Gaussian blur instead of pixelation.")
    args = parser.parse_args()

    # Check if path is a single file
    if os.path.isfile(path):
        # Process image files
        if path.lower().endswith(image_exts):
            process_image(path, use_blur)
        # Process video files
        elif path.lower().endswith(video_exts):
            input_path = args.video_path
            
            # Generate file paths for different processing stages
            base, ext = os.path.splitext(input_path)
            tmp_video_path = f"{base}_tmp{ext}"  # Temporary video without audio
            blurred_output_path = f"{base}_blurred{ext}"  # Video with blurred faces and original audio
            
            # Step 1: Process video frames (blur/pixelate faces)
            process_video(input_path, tmp_video_path, use_blur=args.blur)
            
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

    # Execute main function with parsed arguments
    main(args.path, use_blur=args.blur)
