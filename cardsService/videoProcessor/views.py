from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .video_functions import download_youtube_video, extract_audio, transcribe_audio_with_azure
from .cards_functions import separate_transcription_into_themes
from django.views.decorators.csrf import csrf_exempt
import os
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse

@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        youtube_url = request.POST.get('youtube_url')
        if youtube_url:
            fs = FileSystemStorage()
            download_path = fs.location
            video_path = None  # Initialize video_path
            audio_path = None  # Initialize audio_path
            try:
                # Download the video from YouTube
                video_path = download_youtube_video(youtube_url, download_path)
                
                # Generate audio file path
                audio_filename = os.path.splitext(
                    os.path.basename(video_path))[0] + '.wav'
                audio_path = os.path.join(download_path, audio_filename)
                
                # Extract audio from the video
                extract_audio(video_path, audio_path)
                
                # Transcribe the extracted audio
                transcription = transcribe_audio_with_azure(audio_path)
                
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
            
            finally:
                # Cleanup: Delete the audio and video files after processing
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            # gptreturn = separate_transcription_into_themes(transcription)
            return JsonResponse({
                'message': 'Video uploaded and transcribed successfully!',
                'transcription': transcription
            })
        else:
            return JsonResponse({'error': 'No YouTube URL provided'},
                                status=400)
    return JsonResponse({'error': 'Invalid request'}, status=405)