import json
import requests


VALHALLA_URL = "http://localhost:8002"


def calculate_routes_matrix(
    sources,
    targets,
    mode="auto",
    valhalla_url=VALHALLA_URL,
):
    """
    Calcola la matrice sources -> targets usando Valhalla.

    URL generato:

    http://localhost:8002/sources_to_targets?json={"sources":[...],"targets":[...],"costing":"auto"}
    """

    if not sources or not targets:
        return []

    # ---------------------------------------------------------
    # Costruzione payload
    # ---------------------------------------------------------

    payload = {
        "sources": [
            {
                "lat": float(source["lat"]),
                "lon": float(source["lon"]),
            }
            for source in sources
        ],
        "targets": [
            {
                "lat": float(target["lat"]),
                "lon": float(target["lon"]),
            }
            for target in targets
        ],
        "costing": mode,
    }

    # ---------------------------------------------------------
    # JSON compatto
    # ---------------------------------------------------------

    json_payload = json.dumps(
        payload,
        separators=(",", ":"),
    )

    # ---------------------------------------------------------
    # URL ESATTAMENTE nel formato Valhalla
    # ---------------------------------------------------------

    url = (
        f"{valhalla_url.rstrip('/')}"
        f"/sources_to_targets"
        f"?json={json_payload}"
    )

    print("\n========== VALHALLA URL ==========")
    print(url)
    print("==================================\n")

    # ---------------------------------------------------------
    # GET
    # ---------------------------------------------------------

    response = requests.get(
        url,
        timeout=30,
    )

    if response.status_code != 200:

        print("\n========== VALHALLA ERROR ==========")
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)
        print("====================================\n")

        response.raise_for_status()

    data = response.json()

    matrix = data.get(
        "sources_to_targets",
        []
    )

    results = []

    for source, row in zip(sources, matrix):

        for target, result in zip(targets, row):

            if result is None:
                continue

            results.append({
                "source_id": source["id"],
                "target_id": target["id"],
                "distance_km": float(result["distance"]),
                "time_seconds": int(result["time"]),
            })

    return results


def calculate_routes(
    source,
    targets,
    mode="auto",
    valhalla_url=VALHALLA_URL,
):
    """
    Calcola source -> N targets.
    """

    return calculate_routes_matrix(
        sources=[source],
        targets=targets,
        mode=mode,
        valhalla_url=valhalla_url,
    )