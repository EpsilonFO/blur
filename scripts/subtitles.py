import whisper

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

def generate_subtitles(audio_path, srt_path):
    """
    Generate SRT subtitle file from audio using OpenAI Whisper speech recognition.
    
    Args:
        audio_path (str): Path to input audio file
        srt_path (str): Path for output SRT subtitle file
    """
    # Load Whisper model (base model provides good balance of speed and accuracy)
    model = whisper.load_model("small")
    
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

