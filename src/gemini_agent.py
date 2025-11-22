import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY environment variable not set. "
        "Please set it before running the app. Example:\n"
        "  export GOOGLE_API_KEY=your_api_key_here"
    )
genai.configure(api_key=GOOGLE_API_KEY)

def gemini_chat(prompt: str) -> str:
    """
    Sends a prompt to the Gemini API using the official SDK and returns the response text.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"
