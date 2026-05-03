# Test script to verify your Gemini API key
from dotenv import load_dotenv
import os
from google import genai

# Load the API key from your .env file
load_dotenv()

# Create a Gemini client using your API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Send a simple text prompt to Gemini 2.5 Flash
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence."
)

# Print Gemini's response
print(response.text)
