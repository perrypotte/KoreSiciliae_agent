"""
Modulo per il servizio di Geocoding.
Fornisce funzionalità per convertire indirizzi in coordinate (Latitudine, Longitudine)
utilizzando l'API di Nominatim (OpenStreetMap).
"""

import time
import requests
from typing import Optional, Dict, Any

# Costante per garantire il rispetto del rate limit di 1 req/sec
LAST_REQUEST_TIME = 0.0

def geocode_address(
    address: str, 
    user_agent: str = "ciccozeta@live.it",
    country_codes: str = "it"
) -> Optional[Dict[str, Any]]:
    """
    Esegue il geocoding di un indirizzo usando le API pubbliche di Nominatim.
    
    :param address: L'indirizzo o il nome del luogo (es. "Cattedrale di Palermo").
    :param user_agent: Nome univoco dell'app/email (obbligatorio per le policy di Nominatim).
    :param country_codes: Codice nazione ISO per restringere la ricerca (default: 'it').
    :return: Dict con 'lat', 'lon', 'display_name' o None se fallisce.
    """
    global LAST_REQUEST_TIME

    if not address or not address.strip():
        return None

    # Rispetta il rate limit di max 1 richiesta al secondo
    elapsed = time.time() - LAST_REQUEST_TIME
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": country_codes
    }
    
    headers = {
        "User-Agent": user_agent
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        LAST_REQUEST_TIME = time.time()

        if response.status_code == 200:
            data = response.json()
            if data:
                res = data[0]
                return {
                    "lat": float(res["lat"]),
                    "lon": float(res["lon"]),
                    "display_name": res["display_name"],
                    #"raw": res
                }
        else:
            print(f"⚠️ Errore API Nominatim [{response.status_code}]: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore di connessione durante il geocoding: {e}")

    return None