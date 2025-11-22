import spacy
import re
from src.weather_agent import get_weather_info
from src.places_agent import get_places_info

# Load spaCy model (you'll need to install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model not installed
    nlp = None
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")


def extract_location_nlp(text: str) -> str | None:
    """
    Extract location using spaCy Named Entity Recognition.
    Returns the most relevant location entity found.
    """
    if nlp is None:
        return extract_location_fallback(text)
    
    doc = nlp(text)
    
    # Extract all location entities (GPE = Geo-Political Entity, LOC = Location)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    
    if locations:
        # Prioritize locations that appear after prepositions or verbs
        text_lower = text.lower()
        location_keywords = ["in", "at", "to", "near", "visit", "go", "travel", "trip"]
        
        for loc in locations:
            loc_pos = text_lower.find(loc.lower())
            # Check if there's a relevant keyword before this location
            prefix = text_lower[:loc_pos]
            if any(keyword in prefix for keyword in location_keywords):
                return loc.lower()
        
        # If no keyword found, return the last location mentioned
        return locations[-1].lower()
    
    # Fallback to regex-based extraction
    return extract_location_fallback(text)


def extract_location_fallback(text: str) -> str | None:
    """
    Fallback regex-based location extraction when NLP is unavailable.
    """
    text_norm = text.replace("'", "'")
    text_clean = text_norm.strip()

    # Simplified patterns focusing on prepositions and common verbs
    patterns = [
        r"\b(?:visit|go|travel|trip)\s+(?:some\s+)?(?:places\s+)?(?:in|at|to|near)\s+([A-Z][A-Za-z\s]+?)(?:[,.!?]|$)",
        r"\b(?:in|at|to|near)\s+([A-Z][A-Za-z\s]+?)(?:[,.!?]|$)",
        r"\b(?:visit|go|travel)\s+(?:to\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)(?:[,.!?]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            
            # Clean up filler words
            filler_words = ["some", "places", "any", "the", "many", "few", "several"]
            words = candidate.split()
            words = [w for w in words if w.lower() not in filler_words]
            
            candidate = " ".join(words).strip()
            
            if candidate and len(candidate) > 2:
                return candidate.lower()

    # Last resort: check for capitalized words
    capitals = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", text_norm)
    NON_LOCATIONS = {"ok", "yes", "no", "hi", "hey", "hello", "thanks", "i"}
    
    for cap in reversed(capitals):  # Check from end
        if cap.lower() not in NON_LOCATIONS:
            return cap.lower()

    return None


def extract_location(text: str) -> str | None:
    """
    Main location extraction function - uses NLP when available.
    """
    return extract_location_nlp(text)


def detect_intent(query: str):
    """
    Returns two booleans: wants_weather, wants_places
    """
    q = query.lower()

    weather_keywords = [
        "weather", "temperature", "climate", "rain", "snow",
        "sunny", "forecast", "humidity", "wind"
    ]

    places_keywords = [
        "places", "attractions", "visit", "tourist", "things to do",
        "plan my trip", "plan my visit", "sightseeing", "travel guide",
        "where to go"
    ]

    wants_weather = any(k in q for k in weather_keywords)
    wants_places = any(k in q for k in places_keywords)

    # If neither keyword is present, guess the intent:
    # If query contains a location + no specific intent → give places by default.
    if not (wants_weather or wants_places):
        wants_places = True

    return wants_weather, wants_places


def handle_user_query(query: str) -> str:
    """
    Strong orchestrator:
    - Detects intent (weather / places / both)
    - Extracts location robustly using NLP
    - Calls appropriate agents
    - Combines response
    """

    location = extract_location(query)

    if not location:
        return "I'm not sure which location you are referring to. Please provide the city name."

    wants_weather, wants_places = detect_intent(query)

    responses = []

    if wants_weather:
        responses.append(get_weather_info(location))

    if wants_places:
        responses.append(get_places_info(location))

    # If neither matched (very rare)
    if not responses:
        return "Please ask about the weather or places to visit for a specific location."

    return "\n\n".join(responses)