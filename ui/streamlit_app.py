import os
import streamlit as st
import numpy as np
import time
import torch
from functools import lru_cache

from src.logger import logger
from src.video_processor import VideoProcessor
from src.embedding_service import EmbeddingService
from src.transcription_service import TranscriptionService
from src.context_retriever import ContextRetriever
from src.chatbot_engine import ChatbotEngine
from src.vectorstore import VectorStore


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_video(video_source):
    """Cached video processing function"""
    logger.info("Starting video processing")
    start_time = time.time()
    
    # Initialize services
    processor = VideoProcessor(video_source)
    transcriber = TranscriptionService()
    embedder = EmbeddingService()
    vectorstore = VectorStore()
    
    try:
        # Download/process video
        video_path = (processor.download_youtube_video() 
                      if 'youtube.com' in str(video_source) 
                      else processor.process_uploaded_file(video_source))
        logger.info(f"Video downloaded/processed: {video_path}")
        # Extract frames (limit to reduce processing time)
        frames = processor.extract_frames()
        # timestamps = processor.get_frame_timestamps()
        
        # Get Transcript
        if video_source and 'youtube.com' in str(video_source):
            transcript_data = transcriber.get_youtube_transcript(video_source)
        else:
            transcript_data = transcriber.get_transcription(
                video_source
            )

        context_vectorstore, image_vectorstore = vectorstore.embed_and_store(transcript_data)
        
        # Embed content
        # text_embeddings = embedder.embed_text([transcript_data['text']])

        # image_embeddings = embedder.embed_image(frames)
        
        logger.info(f"Video processing completed in {time.time() - start_time:.2f} seconds")
        
        return context_vectorstore, image_vectorstore, transcript_data
    
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise

def main():
    
    st.set_page_config(
        page_title="Video Content Chatbot", 
        page_icon="🎥", 
        layout="wide"
    )
    
    st.title("🎬 Video Content Chatbot")
    
    # Input method selection
    input_type = st.sidebar.radio(
        "Choose Input Method", 
        ["YouTube Link", "Video File Upload"]
    )
    
    # Video Input
    if input_type == "YouTube Link":
        video_source = st.sidebar.text_input("Enter YouTube Video URL")
        uploaded_file = None
    else:
        video_source = None
        uploaded_file = st.sidebar.file_uploader(
            "Upload Video File", 
            type=['mp4', 'avi', 'mov']
        )
    
    # Process Video
    if video_source or uploaded_file:
        try:
            # Use cached processing
            with st.spinner('Processing video...'):
                context_vectorstore, image_vectorstore, transcript_data = process_video(video_source or uploaded_file)
            

            # Chatbot and Context Retrieval
            retriever = ContextRetriever(context_vectorstore, image_vectorstore)
            chatbot = ChatbotEngine()
            # embedder = EmbeddingService()
            
            # Main Content Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("Video Insights")
                query = st.text_input("Ask a question about the video content")
                reference_image = st.file_uploader("Optional: Upload Reference Image", type=['png', 'jpg', 'jpeg'])

                
                if st.button("Get Answer"):
                    # Embed query
                    # query_embedding = embedder.embed_text([query])
                    # if reference_image:
                    #     image_embedding = embedder.embed_image([reference_image])
                        # query_embedding = np.concatenate([query_embedding, image_embedding])
                    
                    # Retrieve and generate response
                    text_context = retriever.retrieve_context(query)
                    logger.info(f"Retrieved context: {text_context}")
                    response = chatbot.generate_response(
                        context=text_context, 
                        query=query
                    )
                    
                    st.markdown("### 🤖 Response")
                    st.write(response)
            
            with col2:
                # st.header("Video Details")
                video_id = id = video_source.split("=")[-1]
                st.markdown(
                    "<h1 style='color: #EC5331;'>📽️ Video Details</h1>",
                    unsafe_allow_html=True,
                )
                # st.markdown(
                #     f"<p style='font-size: 18px;'><strong><span style='color:#EC5331;'>Summary:<br></span></strong> {summary}</p>",
                #     unsafe_allow_html=True,
                # )
                st.markdown(
                    f'<iframe width="490" height="315" src="https://www.youtube.com/embed/{video_id}?start={90}&autoplay=1" frameborder="0" allowfullscreen></iframe>',
                    unsafe_allow_html=True,
                )
                st.markdown("### Transcript Preview")
                transcript = st.markdown(
                    f"<div style='height: 400px; overflow-y: scroll;'>{transcript_data['text']}</div>",
                    unsafe_allow_html=True,
                )

                # st.markdown("### Transcript Preview")
                # st.write(processed_data['transcript']['text'][:500] + "...")

        
        except Exception as e:
            st.error(f"Processing Error: {e}")
            logger.error(f"Streamlit app error: {e}")
    
    else:
        st.markdown("### 🚀 Welcome to Video Content Chatbot")

if __name__ == "__main__":
    main()