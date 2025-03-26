import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from src.vectorstore import embed_and_store

class ContextRetriever:
    def __init__(self, context_vectorstore, image_vectorstore = None,k=5):
        self.context_vectorstore = context_vectorstore
        self.image_vectorstore = image_vectorstore
        self.top_k = k


    
    def retrieve_context(self,query):
        """Retrieve most relevant context based on embedding similarity"""
        
        similarities = self.context_vectorstore.similarity_search(
            query, 
            k = self.top_k,
        )
        if self.image_vectorstore:
            image_similarities = self.image_vectorstore.similarity_search(
                query,
                k = self.top_k,
            )
        
        # Get indices of top-k similar segments
        # top_indices = similarities.argsort()[0][-top_k:][::-1]
        # print(top_indices)
        formatted_context = "\n".join([
        f"Content: {doc.page_content}\nMetadata: {doc.metadata}"
            for doc in similarities
        ])
        return f"Context:\n{formatted_context}\n\nUse this context to answer the user's query."
