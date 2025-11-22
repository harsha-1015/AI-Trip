import requests
import os
from src.gemini_agent import gemini_chat
from dotenv import load_dotenv

load_dotenv()

def get_weather_info(location: str) -> str:
    """
    Uses Open-Meteo APIs to get weather data,
    then formats the response using Gemini.
    """
    # Step 1: Get coordinates from Nominatim
    nominatim_url = os.getenv("NOMINATIM_URL")
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "tourism-app"}
    try:
        resp = requests.get(nominatim_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Sorry, there was a problem contacting the location service: {e}"

    # Only check for valid lat, lon, and display_name (no importance threshold)
    if (
        not data
        or not isinstance(data, list)
        or not data[0].get("lat")
        or not data[0].get("lon")
        or not data[0].get("display_name")
    ):
        return f"Sorry, I couldn't find the location '{location}'. Please check the spelling or try a different place."

    lat = data[0]["lat"]
    lon = data[0]["lon"]

    # Step 2: Get weather from Open-Meteo
    openmeteo_url = os.getenv("OPENMETEO_URL")
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation_probability",
        "timezone": "auto"
    }
    try:
        resp = requests.get(openmeteo_url, params=params, timeout=10)
        resp.raise_for_status()
        weather_data = resp.json()
    except Exception as e:
        return f"Sorry, there was a problem contacting the weather service: {e}"

    try:
        temp = weather_data["current"]["temperature_2m"]
        precip = weather_data["current"]["precipitation_probability"]
        # Use Gemini to format the response
        prompt = (
            f"You are a helpful weather assistant. "
            f"Format the following data for a user: "
            f"Location: {location}, Temperature: {temp}°C, Precipitation Probability: {precip}%."
            f"Respond as: 'In [location] it's currently [temperature]°C with a [precipitation]% chance of rain.'"
        )
        return gemini_chat(prompt)
    except Exception:
        return f"Sorry, I couldn't retrieve weather data for '{location}'."
