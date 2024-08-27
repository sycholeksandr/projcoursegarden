from moviepy.editor import VideoFileClip
import os
import yt_dlp
from . import local_settings
import azure.cognitiveservices.speech as speechsdk
import threading
from pydub import AudioSegment
import concurrent.futures 
from .ai_funktions import transcription_ai_cleanup
# Function that downloads video from given YouTube URL
# and saves the video in specified place
def download_youtube_video(youtube_url, download_path):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Download the video
        ydl.download([youtube_url])
    video_file_name = ydl.prepare_filename(ydl.extract_info(youtube_url,
                                                             download=False))
    return video_file_name

# Function that extracts audio file from uploaded video
def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()
    return audio_path

# Function that increases the audio playing speed to get transcription faster
def increase_audio_speed(input_file, output_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    # Calculate the new frame rate to increase speed
    new_frame_rate = int(audio.frame_rate * 1.5)
    
    # Spawn a new audio segment with the new frame rate
    new_audio = audio._spawn(audio.raw_data,
                              overrides={'frame_rate': new_frame_rate})
    
    # Set the frame rate of the new audio segment
    new_audio = new_audio.set_frame_rate(new_frame_rate)
    
    # Export the new audio to the output file
    new_audio.export(output_file, format="wav")

    return output_file

def transcribe_chunk(chunk_path, start_index):
    speech_config = speechsdk.SpeechConfig(
        subscription=local_settings.AZURE_SPEECH_KEY, region='eastus')
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "uk-UA", "de-DE", "fr-FR"]
    )
    audio_input = speechsdk.AudioConfig(filename=chunk_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        auto_detect_source_language_config=auto_detect_source_language_config, 
        audio_config=audio_input
    )
    result = speech_recognizer.recognize_once()
    return start_index, result.text

def transcribe_audio_with_azure(audio_path):
    chunk_length_ms = 10000
    overlap_ms = 1000
    temp_audio_path = audio_path.replace('.wav', '_speedup.wav')
    
    increase_audio_speed(audio_path, temp_audio_path)
    audio = AudioSegment.from_file(temp_audio_path)
    
    chunks = []
    for i in range(0, len(audio), chunk_length_ms - overlap_ms):
        start = i
        end = min(i + chunk_length_ms, len(audio)) # Avoid going out of bounds
        chunk = audio[start:end]
        
        # Save the chunk as a temporary file
        chunk_path = f"{temp_audio_path}_chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append((chunk_path, start))
    
    # Transcribe each chunk concurrently 
    # and collect results with their start indices
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(transcribe_chunk, chunk_path, start):
                   start for chunk_path, start in chunks}
        for future in concurrent.futures.as_completed(futures):
            start_index, result_text = future.result()
            results.append((start_index, result_text))

    # Sort results based on start index
    results.sort(key=lambda x: x[0])

    # Combine sorted transcriptions into a single string
    transcription_result = transcription_ai_cleanup(" ".join([text for _, text in results]))

    # Clean up temporary files
    for chunk_path in chunks:
        os.remove(chunk_path[0])
    os.remove(temp_audio_path)
    
    return transcription_result  
