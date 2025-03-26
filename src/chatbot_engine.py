import os
import cohere
from dotenv import load_dotenv

load_dotenv()

class ChatbotEngine:
    def __init__(self):
        self.cohere_client = cohere.ClientV2(os.getenv('COHERE_API_KEY'))
    
    def generate_response(self, context, query):
        """Generate conversational response using Cohere's chat endpoint"""
        try:
            message = [
                {"role": "system", "content": context},
                {
                    "role": "user",
                    "content": query,
                },
            ]
            response = self.cohere_client.chat(
                messages=message,
                model='command-r',
                # temperature=0.3,
                # max_tokens=300
            )
            return response.message.content[0].text
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I'm unable to generate a response right now."