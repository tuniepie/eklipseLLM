# 🎥 EklipseLLM

## Overview
A powerful Streamlit application that allows semantic search and interaction with video content using advanced AI models.

## Features
- Support for YouTube links and local video files
- Multimodal content analysis
- Semantic search across video content
- Image captioning
- Speech transcription

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended but optional)

## Installation

1. Clone the repository
```bash
git clone https://github.com/tuniepie/EklipseLLM.git
cd EklipseLLM
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install system dependencies
- FFmpeg (Required for video processing)
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `choco install ffmpeg`



## Run the Application
```bash
streamlit run run.py
```

## Usage
1. Choose input type (YouTube Link or Upload Video)
2. Click "Process Video"
3. Wait for analysis
4. Ask questions about the video content

## Model Components
- Whisper: Audio Transcription
- CLIP: Semantic Understanding
- Faiss: Vector store and similarity search

## Limitations
- Requires significant computational resources
- Works best with clear audio and video
- Accuracy depends on input video quality
