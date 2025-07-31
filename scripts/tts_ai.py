"""
Script to anonymize a video by extracting its audio, transcribing it with Whisper,
synthesizing it with TTS, and replacing the original audio in the video.
"""

import os
import subprocess
import tempfile
import argparse
from moviepy import VideoFileClip
from pydub import AudioSegment
import whisper
from TTS.api import TTS
from TTS.utils.manage import ModelManager

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.utils.radam import RAdam
import collections
import torch

from config import TTS_MODEL_NAME, WHISPER_MODEL
from scripts.subtitles import generate_subtitles


torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs, 
                                      RAdam, collections.defaultdict, dict])
manager = ModelManager()
# print(manager.list_models())

# === Étape 1 : Extraction de l'audio ===
def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

# === Étape 2 : Transcription avec Whisper ===
def transcribe_audio(audio_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language="fr")
    return result["segments"]

# === Étape 3 : Synthèse vocale avec ajustement temporel ===
def synthesize_segments_with_timing(segments, output_audio_path):
    tts = TTS(model_name=TTS_MODEL_NAME,progress_bar=False, gpu=False)
    combined = AudioSegment.silent(duration=0)

    for segment in segments:
        text = segment["text"].strip()
        start = segment["start"]
        end = segment["end"]
        target_duration_ms = int((end - start) * 1000)

        if not text:
            combined += AudioSegment.silent(duration=target_duration_ms)
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tts.tts_to_file(text=text, file_path=tmp_wav.name)
            synthetic = AudioSegment.from_wav(tmp_wav.name)
            os.remove(tmp_wav.name)

        actual_duration_ms = len(synthetic)

        if actual_duration_ms < target_duration_ms:
            padding = AudioSegment.silent(duration=target_duration_ms - actual_duration_ms)
            synthetic = synthetic + padding
        else:
            synthetic = synthetic[:target_duration_ms]

        combined += synthetic

    combined.export(output_audio_path, format="wav")

def replace_audio_in_video(original_video_path, new_audio_path, output_video_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", original_video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_video_path
    ]
    subprocess.run(command, check=True)

def process_tts(path, anonymized_output_path, output_srt, use_blur='pixelate'):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "original_audio.wav")
        synthetic_audio_path = os.path.join(tmpdir, "synthetic_audio.wav")

        print("🔊 Extraction de l'audio...")
        extract_audio(path, audio_path)

        generate_subtitles(audio_path, output_srt)

        print("📝 Transcription avec Whisper...")
        segments = transcribe_audio(audio_path, WHISPER_MODEL)

        print("🗣️ Synthèse vocale avec ajustement temporel...")
        synthesize_segments_with_timing(segments, synthetic_audio_path)

        print("🎬 Remplacement de l'audio dans la vidéo...")
        replace_audio_in_video(path, synthetic_audio_path, anonymized_output_path)