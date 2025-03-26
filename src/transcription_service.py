import os
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


class TranscriptionService:
    def __init__(self, model_size='base'):
        self.whisper_model = whisper.load_model(model_size)
    
    @staticmethod
    def extract_video_id(url):
        """Extract YouTube video ID from URL"""
        parsed_url = urlparse(url)
        
        # Handle different YouTube URL formats
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path.strip('/')
        
        if parsed_url.netloc in ['www.youtube.com', 'youtube.com']:
            if 'v' in parse_qs(parsed_url.query):
                return parse_qs(parsed_url.query)['v'][0]
            
            # Handle embedded URL format
            if '/embed/' in parsed_url.path:
                return parsed_url.path.split('/embed/')[1]
            
            # Handle channel/watch URL format
            if '/watch' in parsed_url.path:
                return parse_qs(parsed_url.query)['v'][0]
        
        return None
    
    def get_youtube_transcript(self, youtube_url):
        """Try to get YouTube transcript"""
        video_id = self.extract_video_id(youtube_url)
        
        if not video_id:
            return None
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Process transcript
            full_text = " ".join([entry['text'] for entry in transcript])
            timestamped_transcript = [
                {
                    'text': entry['text'],
                    'start_time': entry['start'],
                    'duration': entry['duration']
                } 
                for entry in transcript
            ]
            
            return {
                'text': full_text,
                'segments': timestamped_transcript
            }
        
        except Exception:
            return None
    
    def extract_audio(self, video_path):
        """Extract audio from video"""
        video = VideoFileClip(video_path)
        audio_path = 'data/processed/extracted_audio.wav'
        video.audio.write_audiofile(audio_path)
        return audio_path
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        result = self.whisper_model.transcribe(audio_path)
        return {
            'text': result['text'],
            'segments': result['segments']
        }
    
    def get_transcription(self, source):
        """
        Unified transcription method
        - For YouTube links: Try to get YouTube transcript
        - For video files: Extract audio and transcribe
        """
        # Check if source is a YouTube URL
        if isinstance(source, str) and ('youtube.com' in source or 'youtu.be' in source):
            youtube_transcript = self.get_youtube_transcript(source)
            if youtube_transcript:
                return youtube_transcript
        
        # If not YouTube or no transcript, process video/audio
        if isinstance(source, str):  # Video file path
            audio_path = self.extract_audio(source)
            return self.transcribe_audio(audio_path)
        
        # For file upload in Streamlit
        if hasattr(source, 'name'):
            # Save uploaded file
            temp_video_path = f'data/processed/{source.name}'
            with open(temp_video_path, 'wb') as f:
                f.write(source.getvalue())
            
            # Extract audio and transcribe
            audio_path = self.extract_audio(temp_video_path)
            return self.transcribe_audio(audio_path)
        
        raise ValueError("Invalid input source for transcription")