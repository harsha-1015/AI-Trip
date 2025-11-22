# Multi-Agent Tourism Planning and Weather Information System

A modern, multi-agent chatbot system that helps users plan trips, check weather, and discover tourist attractions for any city using real-time open APIs. Powered by OpenStreetMap, Open-Meteo, Overpass, and Google Gemini AI.

---

## Project Structure

    │
    ├── main_app.py # Streamlit UI (chat interface)
    ├── .env # Environment variables (API URLs, keys)
    ├── requirements.txt # Python dependencies
    ├── README.md 
    │
    ├── src/
    │ ├── orchestrator.py # Orchestrator agent (intent/location extraction, routing)
    │ ├── weather_agent.py # Weather agent (Nominatim + Open-Meteo)
    │ ├── places_agent.py # Places agent (Nominatim + Overpass + reverse geocode)
    │ └── gemini_agent.py # Google Gemini API wrapper


## Dependencies

    - Python 3.9+
    - [Streamlit](https://streamlit.io/)
    - [requests](https://pypi.org/project/requests/)
    - [python-dotenv](https://pypi.org/project/python-dotenv/)
    - [google-generativeai](https://pypi.org/project/google-generativeai/) (for Gemini API)

    Install all dependencies with:
    pip install -r requirements.txt


## APIs Used

    # Nominatim (OpenStreetMap)
    Purpose: Geocoding (convert place name to coordinates) and reverse geocoding (get canonical city name).
    Docs: https://nominatim.org/release-docs/develop/api/Search/
    # Open-Meteo
    Purpose: Get current weather (temperature, precipitation) for coordinates.
    Docs: https://open-meteo.com/en/docs
    # Overpass API (OpenStreetMap)
    Purpose: Find up to 5 tourist attractions near coordinates.
    Docs: https://wiki.openstreetmap.org/wiki/Overpass_API
    # Google Gemini AI
    Purpose: Natural language formatting and friendly responses.

## How to Install & Run
Clone the repository:
git clone <your-repo-url>
cd monu_project

  # Set up environment variables:
    Create a .env file in the root directory:
    NOMINATIM_URL=https://nominatim.openstreetmap.org/search
    OPENMETEO_URL=https://api.open-meteo.com/v1/forecast
    OVERPASS_URL=https://overpass-api.de/api/interpreter
    GOOGLE_API_KEY=your_google_gemini_api_key

## Run the app
streamlit run main_app.py


## How It Works
    User enters a query (e.g., "What's the weather in Paris? What are the top places to visit?")
    Orchestrator Agent extracts the location and intent (weather, places, or both).
    Weather Agent uses Nominatim to get coordinates, then Open-Meteo for weather.
    Places Agent uses Nominatim for coordinates, reverse geocodes for canonical city name, then Overpass for tourist attractions.
    Gemini AI formats the responses in a friendly, readable way.
    Streamlit UI displays the conversation in a modern chat interface.

## Notes
    All location and attraction data is fetched live from open APIs—no hallucinated or AI-invented places.
    If a location is not found, the system politely informs the user.
    The UI supports markdown lists for attractions and a modern chat experience.