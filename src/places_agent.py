import requests
import os
from src.gemini_agent import gemini_chat
from dotenv import load_dotenv

load_dotenv()

nominatim_url=os.getenv("NOMINATIM_URL")

def get_places_info(location: str) -> str:
    # Step 1: Get coordinates from Nominatim
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "tourism-app"}
    resp = requests.get(nominatim_url, params=params, headers=headers)
    data = resp.json()
    # Only check for valid lat, lon, and display_name (no importance threshold)
    if (
        not data
        or not isinstance(data, list)
        or not data[0].get("lat")
        or not data[0].get("lon")
        or not data[0].get("display_name")
    ):
        return f"Sorry, I don't know about the place '{location}'. Please check the spelling or try a different place."

    lat = data[0]["lat"]
    lon = data[0]["lon"]

    # Step 1b: Reverse geocode to get canonical city name
    reverse_url = nominatim_url
    reverse_params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 10,  # city/town/village level
        "addressdetails": 1
    }
    reverse_resp = requests.get(reverse_url, params=reverse_params, headers=headers)
    canonical_city = None
    if reverse_resp.ok:
        reverse_data = reverse_resp.json()
        address = reverse_data.get("address", {})
        # Prefer city, then town, then village, then state_district, then state
        for key in ["city", "town", "village", "municipality", "county", "state_district", "state"]:
            if key in address:
                canonical_city = address[key]
                break
    if not canonical_city:
        canonical_city = data[0].get("display_name", location).split(",")[0]

    # Step 2: Get places from Overpass API
    overpass_url = os.getenv("OVERPASS_URL")
    query = f"""
    [out:json];
    (
      node["tourism"](around:10000,{lat},{lon});
      way["tourism"](around:10000,{lat},{lon});
    );
    out body;
    """
    resp = requests.post(overpass_url, data=query, headers={"Content-Type": "text/plain"})
    places_data = resp.json()
    elements = places_data.get("elements", [])
    place_names = []
    for el in elements:
        name = el.get("tags", {}).get("name")
        if name and name not in place_names:
            place_names.append(name)
        if len(place_names) >= 5:
            break
    if not place_names:
        return f"I'm sorry, but I couldn't find any tourist attractions in {canonical_city}."

    # Use Gemini to format up to 5 places, one per line, as a Markdown list, with canonical city name
    prompt = (
        f"You are a helpful tourism assistant. "
        f"Here are some places to visit in {canonical_city}:\n"
        f"{place_names}\n"
        f"Return only a Markdown bullet list (each place on its own line, starting with '- '), using only the names provided. "
        f"Do not add or invent any places. Do not include any explanation, just the list and one line of intraduction."
    )
    return gemini_chat(prompt)
