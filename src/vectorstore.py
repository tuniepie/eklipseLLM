import faiss
import numpy as np
from uuid import uuid4
from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from src.logger import logger
class VectorStore:
    def __init__(self):
        # Load text embedding model
        self.text_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load image embedding model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize FAISS indices
        self.context_index = None
        self.keyframe_index = None
        self.context_vectorstore = None
        self.keyframe_vectorstore = None
    
    def embed_text(self, texts):
        return np.array(self.text_model.embed_documents(texts), dtype='float32') if texts else np.array([])
    
    def embed_images(self, images):
        image_embeddings = []
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_embedding = self.clip_model.get_image_features(**inputs)
            image_embeddings.append(img_embedding.squeeze().numpy())
        return np.array(image_embeddings, dtype='float32') if image_embeddings else np.array([])
    
    def embed_and_store(self, text_data, keyframe_data = None):
        # Process text context
        texts, text_metadata = [], []
        for segment in text_data['segments']:
            if 'text' in segment:
                texts.append(segment['text'])
                text_metadata.append({'start_time': segment['start_time'], 'duration': segment['duration']})
        logger.info(f"Text segments: {texts}")
        text_embeddings = self.embed_text(texts[0])
        self.context_index = faiss.IndexFlatL2(text_embeddings.shape[1])
        # self.context_index.add(text_embeddings)
        documents=[Document(page_content=texts[i], metadata=text_metadata[i]) for i in range(len(texts))]
        self.context_vectorstore = FAISS(
            embedding_function=self.text_model, 
            index=self.context_index, 
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            # documents=[Document(page_content=texts[i], metadata=text_metadata[i]) for i in range(len(texts))]
        )
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        self.context_vectorstore.add_documents(documents=documents, ids=uuids)
        # self.context_vectorstore.add_documents(documents)
        logger.info("Text context stored")
        # Process keyframes
        if keyframe_data:

            keyframes, timestamps = keyframe_data['image'], keyframe_data['timestamp']
            keyframe_embeddings = self.embed_images(keyframes)
            self.keyframe_index = faiss.IndexFlatL2(keyframe_embeddings.shape[1])
            self.keyframe_index.add(keyframe_embeddings)
            self.keyframe_vectorstore = FAISS(
                embedding_function=self.text_model, 
                index=self.keyframe_index, 
                documents=[Document(page_content="[IMAGE]", metadata={'timestamp': timestamps[i]}) for i in range(len(timestamps))]
            )
        
        return self.context_vectorstore, self.keyframe_vectorstore

# Example usage
data = {
    "segments": [
        {"text": "K-pop!", "start_time": 10.2, "duration": 0.94},
        # {"image": "path/to/image.jpg", "start_time": 15.0, "duration": 3.0},
        {"text": "That is so awkward to watch.", "start_time": 13.4, "duration": 2.8},
        {"text": "LangGraph is the best framework for building stateful, agentic applications!", "start_time": 15.4, "duration": 2.8}
    ]
}

# def test_vectorstore():
#     embedder = VectorStore()
#     vectorstore, a = embedder.embed_and_store(data)
#     results = vectorstore.similarity_search(
#         "LangChain provides abstractions to make working with LLMs easy",
#         k=2,
#     )
#     print(results)

# test_vectorstore()