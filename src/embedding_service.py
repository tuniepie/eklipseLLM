import os
import cohere
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
    
    def embed_text(self, texts):
        """Generate embeddings for text"""
        try:
            # response = self.cohere_client.embed(
            #     texts=texts,
            #     model=model,
            #     input_type='search_document'
            # )
            self.text_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            response = self.text_model.embed_documents(texts)[0]
            return np.array(response)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def embed_image(self, frames, model='embed-english-v3.0'):
        """Convert frames to base64 and generate embeddings"""
        base64_frames = [
            self._frame_to_base64(frame) for frame in frames
        ]
        
        try:
            response = self.cohere_client.embed(
                texts=base64_frames,
                model=model,
                input_type='search_document'
            )
            return np.array(response.embeddings)
        except Exception as e:
            print(f"Image embedding error: {e}")
            return None
    
    def _frame_to_base64(self, frame):
        """Convert OpenCV frame to base64"""
        import base64
        import cv2
        
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
