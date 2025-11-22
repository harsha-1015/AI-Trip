from src.weather_agent import get_weather_info
from src.places_agent import get_places_info
import re


def extract_location(text: str) -> str | None:
    """
    Extracts probable location using multiple intelligent strategies.
    """

    # 0) Normalize curly apostrophes to simple ones
    text_norm = text.replace("’", "'")
    text_clean = text_norm.strip()

    # 1) Try to catch phrases after clear travel/location verbs.
    #    Order matters: more specific patterns first.
    patterns = [
        r"\bgo to\s+([A-Za-z][A-Za-z\s]+?)(?:[,.!?]|$)",
        r"\bgoing to\s+([A-Za-z][A-Za-z\s]+?)(?:[,.!?]|$)",
        r"\btravel to\s+([A-Za-z][A-Za-z\s]+?)(?:[,.!?]|$)",
        r"\btrip to\s+([A-Za-z][A-Za-z\s]+?)(?:[,.!?]|$)",
        r"\bvisit\s+([A-Za-z][A-Za-z\s]+?)(?:[,.!?]|$)",
        # generic in/at/to/for/near as a fallback
        r"\b(?:in|at|to|for|near)\s+([A-Za-z][A-Za-z\s]+?)(?:[,.!?]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()

            # 1a) Strip trailing non-location phrases from the captured chunk
            #     handle both "let's" and "let’s" via let[’']?s
            candidate = re.split(
                r"\b(?:let[’']?s|and|for|what|which|who|how|when|where|"
                r"plan(?: my)? trip|plan(?: my)? visit)\b",
                candidate,
                flags=re.IGNORECASE,
            )[0].strip()

            if candidate:
                return candidate.lower()

    # 2) If user says “plan my trip/visit”, use last capitalized phrase as location
    if any(x in text_clean.lower() for x in ["plan my trip", "plan my visit", "plan trip", "plan visit"]):
        capitals = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", text_norm)
        if capitals:
            return capitals[-1].lower()

    # 3) Fallbacks
    NON_LOCATIONS = {"ok", "yes", "no", "hi", "hey", "hello", "thanks", "thankyou", "thank you"}

    tokens = re.findall(r"[A-Za-z]+", text_norm)

    # If single token, only return if it's NOT in stopwords and length > 2
    if len(tokens) == 1:
        tok = tokens[0].lower()
        if tok not in NON_LOCATIONS and len(tok) > 2:
            return tok
        return None

    # For multi-token input, last word fallback – only if valid city-like
    if tokens:
        last = tokens[-1].lower()
        if last not in NON_LOCATIONS and len(last) > 2:
            return last

    return None


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
    - Extracts location robustly
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
