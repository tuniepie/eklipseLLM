import os
import cv2
import tempfile
import yt_dlp as youtube_dl
# import numpy as np
import uuid
import shutil
# from moviepy import VideoFileClip
from pathlib import Path
from functools import lru_cache
import logging
from src.logger import logger

class VideoProcessor:
    def __init__(self, video_source):
        self.source = video_source
        self.video_path = None
        self.output_dir = "data/keyframes"
        os.makedirs(self.output_dir, exist_ok=True)
        self.frames = []
        self.fps = None
        self.total_frames = None
        self.temp_dir = tempfile.mkdtemp(prefix='video_chatbot_')
        self.logger = logger or logging.getLogger(__name__)
    
    def __del__(self):
        """Cleanup temporary files and directory"""
        try:
            if self.video_path and os.path.exists(self.video_path):
                os.unlink(self.video_path)
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def _generate_unique_filename(self, extension):
        """Generate a unique filename"""
        return os.path.join(
            self.temp_dir, 
            f"{uuid.uuid4()}.{extension}"
        )
    
    @lru_cache(maxsize=2)
    def download_youtube_video(self):
        """Cached YouTube video download"""
        self.logger.info(f"Downloading YouTube video: {self.source}")
        try:
            ydl_opts = {
                'format': 'bestvideo*+bestaudio/best',
                'outtmpl': self._generate_unique_filename('mp4'),
                'quiet': True,
                'no_warnings': True
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.source, download=True)
                self.video_path = ydl.prepare_filename(info_dict)
            
            self.logger.info(f"Video downloaded successfully: {self.video_path}")
            return self.video_path
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
            raise
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file with unique naming"""
        # Generate unique filename
        unique_filename = self._generate_unique_filename(
            Path(uploaded_file.name).suffix.lstrip('.')
        )
        
        # Write file with unique name
        with open(unique_filename, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        self.video_path = unique_filename
        return self.video_path
    
    def extract_frames(self, interval=1):
        try: 
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            keyframes = []
            timestamps = []
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % (fps * interval) == int(fps * interval / 2):  # Select middle frame
                    timestamp = frame_count / fps
                    keyframe_path = os.path.join(self.output_dir, f"frame_{int(frame_count)}.jpg")
                    cv2.imwrite(keyframe_path, frame)
                    keyframes.append(keyframe_path)
                    timestamps.append(timestamp)
                
                frame_count += 1
            
            cap.release()
            return {"image": keyframes, "timestamp": timestamps}

        
        except Exception as e:
            self.logger.error(f"Frame extraction error: {e}")
            raise e
    
    # def get_frame_timestamps(self):
    #     """Generate timestamps for extracted frames"""
    #     if not self.fps or not self.frames:
    #         raise ValueError("Frames not extracted")
        
    #     return [
    #         (frame_idx * self.fps) / self.fps 
    #         for frame_idx in range(0, len(self.frames), int(self.fps))
    #     ]