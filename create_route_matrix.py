import os
import requests
import psycopg2
import json
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from langchain_postgres import PGVector
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


# ============================================================
# CONFIGURAZIONE
# ============================================================

load_dotenv()

# URL del database PostgreSQL
DATABASE_URL = os.getenv("DATABASE2_URL")

# URL del tuo container Valhalla
# Se Valhalla gira sulla macchina host e questo script gira
# sulla macchina host:
VALHALLA_URL = "http://localhost:8002/sources_to_targets"

# Collection PGVector
COLLECTION_NAME = "koreSiciliae_resources_v7"

# Modalità di routing da calcolare
ROUTING_MODES = [
    "auto",
    "pedestrian",
]

# Numero massimo di coordinate per singola richiesta Valhalla.
#
# ATTENZIONE:
# Se hai molte risorse, una richiesta NxN può diventare enorme.
#
# Con 100 risorse:
# 100 x 100 = 10.000 route
#
# Con 1000 risorse:
# 1000 x 1000 = 1.000.000 route
#
# Per iniziare puoi usare None e mandare tutto in una singola
# richiesta.
#
# Se Valhalla o il server HTTP hanno problemi con richieste
# molto grandi, puoi impostare ad esempio:
#
# BATCH_SIZE = 100
#
BATCH_SIZE = 15


# ============================================================
# EMBEDDINGS
# ============================================================

# Questo serve solo per inizializzare PGVector.
#
# Se nel tuo progetto hai già un oggetto "embeddings",
# puoi sostituire questa parte con il tuo oggetto.

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b"
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """
    Crea una connessione al database PostgreSQL.
    """

    return psycopg2.connect(
        DATABASE_URL
    )


# ============================================================
# CREAZIONE TABELLA ROUTE MATRIX
# ============================================================

def create_route_matrix_table(conn):
    """
    Crea la tabella route_matrix se non esiste.

    Ogni riga rappresenta:

        source_id -> target_id

    per una specifica modalità di trasporto.
    """

    create_table_query = """
    CREATE TABLE IF NOT EXISTS route_matrix (

        source_id TEXT NOT NULL,

        target_id TEXT NOT NULL,

        mode TEXT NOT NULL,

        distance_km DOUBLE PRECISION,

        time_seconds INTEGER,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        PRIMARY KEY (
            source_id,
            target_id,
            mode
        )

    );
    """

    with conn.cursor() as cursor:

        cursor.execute(
            create_table_query
        )

    conn.commit()

    print(
        "Tabella route_matrix pronta."
    )


# ============================================================
# RECUPERA COLLECTION ID
# ============================================================

def get_collection_id(conn):
    """
    Recupera il collection_id associato alla collection PGVector.
    """

    query = """
    SELECT uuid
    FROM langchain_pg_collection
    WHERE name = %s
    """

    with conn.cursor() as cursor:

        cursor.execute(
            query,
            (
                COLLECTION_NAME,
            )
        )

        result = cursor.fetchone()

    if result is None:

        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' non trovata."
        )

    return result[0]


# ============================================================
# CARICA TUTTE LE RISORSE
# ============================================================

def load_all_resources(conn):
    """
    Recupera tutte le risorse direttamente dalla tabella
    PGVector.

    Non utilizziamo similarity_search perché qui non vogliamo
    fare una ricerca semantica.

    Vogliamo semplicemente recuperare tutti i documenti della
    collection.

    Restituisce:

    {
        resource_id: {
            "document_id": ...,
            "latitude": ...,
            "longitude": ...
        }
    }
    """

    collection_id = get_collection_id(
        conn
    )

    query = """
    SELECT
        cmetadata
    FROM langchain_pg_embedding
    WHERE collection_id = %s
    """

    resources = {}

    with conn.cursor() as cursor:

        cursor.execute(
            query,
            (
                collection_id,
            )
        )

        rows = cursor.fetchall()

    print(
        f"Documenti recuperati da PGVector: {len(rows)}"
    )

    for row in rows:

        metadata = row[0]

        if metadata is None:
            continue

        resource_id = metadata.get(
            "document_id"
        )

        latitude = metadata.get(
            "latitude"
        )

        longitude = metadata.get(
            "longitude"
        )

        # ----------------------------------------------------
        # Verifica ID
        # ----------------------------------------------------

        if resource_id is None:

            print(
                "[WARNING] Documento senza id. Ignorato."
            )

            continue

        # ----------------------------------------------------
        # Verifica coordinate
        # ----------------------------------------------------

        if latitude is None or longitude is None:

            print(
                f"[WARNING] "
                f"Risorsa {resource_id} senza coordinate. "
                f"Ignorata."
            )

            continue

        try:

            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )

        except (
            ValueError,
            TypeError
        ):

            print(
                f"[WARNING] "
                f"Coordinate non valide per "
                f"{resource_id}. Ignorata."
            )

            continue

        # ----------------------------------------------------
        # Elimina duplicati
        # ----------------------------------------------------

        if resource_id in resources:

            continue

        resources[resource_id] = {

            "document_id": resource_id,

            "latitude": latitude,

            "longitude": longitude,

        }

    print(
        f"Risorse uniche con coordinate valide: "
        f"{len(resources)}"
    )

    return resources


# ============================================================
# CHIAMATA A VALHALLA
# ============================================================

def calculate_valhalla_matrix(
    resources,
    mode,
    batch_size=BATCH_SIZE
):
    """
    Calcola la matrice completa delle route usando Valhalla
    tramite batch sulle sources.

    Tutte le risorse vengono utilizzate come targets.

    Esempio con 5 risorse e batch_size=2:

        Batch 1:
            sources = [A, B]
            targets = [A, B, C, D, E]

        Batch 2:
            sources = [C, D]
            targets = [A, B, C, D, E]

        Batch 3:
            sources = [E]
            targets = [A, B, C, D, E]

    In questo modo vengono calcolate tutte le combinazioni:

        A -> A
        A -> B
        ...
        E -> E

    Restituisce una lista di tuple:

    (
        source_id,
        target_id,
        mode,
        distance_km,
        time_seconds
    )
    """

    import json
    import urllib3


    # ========================================================
    # PREPARAZIONE RISORSE
    # ========================================================

    resource_list = list(
        resources.values()
    )


    # ========================================================
    # ID DELLE RISORSE
    # ========================================================

    resource_ids = [

        resource["document_id"]

        for resource in resource_list

    ]


    # ========================================================
    # TUTTE LE LOCATIONS
    # ========================================================

    all_locations = [

        {
            "lat": resource["latitude"],
            "lon": resource["longitude"],
        }

        for resource in resource_list

    ]


    # ========================================================
    # INFORMAZIONI
    # ========================================================

    total_resources = len(
        resource_list
    )

    total_routes = (
        total_resources
        * total_resources
    )


    print(
        f"\nCalcolo matrice Valhalla: {mode}"
    )

    print(
        f"Numero risorse: {total_resources}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Numero route teoriche: {total_routes}"
    )


    # ========================================================
    # HTTP CLIENT
    # ========================================================

    http = urllib3.PoolManager()


    # ========================================================
    # RISULTATI FINALI
    # ========================================================

    routes = []


    # ========================================================
    # BATCH DELLE SOURCES
    # ========================================================

    for batch_start in range(
        0,
        total_resources,
        batch_size
    ):

        batch_end = min(

            batch_start + batch_size,

            total_resources

        )


        # ----------------------------------------------------
        # RISORSE SOURCE DEL BATCH
        # ----------------------------------------------------

        batch_resources = resource_list[
            batch_start:batch_end
        ]


        # ----------------------------------------------------
        # ID SOURCE DEL BATCH
        # ----------------------------------------------------

        batch_source_ids = [

            resource["document_id"]

            for resource in batch_resources

        ]


        # ----------------------------------------------------
        # COORDINATE SOURCE DEL BATCH
        # ----------------------------------------------------

        batch_sources = [

            {
                "lat": resource["latitude"],
                "lon": resource["longitude"],
            }

            for resource in batch_resources

        ]


        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Batch "
            f"{batch_start // batch_size + 1}"
        )

        print(
            f"Sources: "
            f"{batch_start + 1} - {batch_end}"
        )

        print(
            f"Numero sources: "
            f"{len(batch_sources)}"
        )

        print(
            f"Numero targets: "
            f"{len(all_locations)}"
        )

        print(
            f"Route nel batch: "
            f"{len(batch_sources) * len(all_locations)}"
        )

        print(
            "=" * 60
        )


        # ====================================================
        # PAYLOAD VALHALLA
        # ====================================================

        payload = {

            "sources": batch_sources,

            "targets": all_locations,

            "costing": mode,

        }


        # ====================================================
        # SERIALIZZAZIONE JSON
        # ====================================================

        json_payload = json.dumps(

            payload,

            separators=(
                ",",
                ":"
            )

        )


        # ====================================================
        # COSTRUZIONE URL
        # ====================================================

        url = (

            VALHALLA_URL

            + "?json="

            + json_payload

        )


        # ====================================================
        # REQUEST VALHALLA
        # ====================================================

        print(
            "Invio richiesta a Valhalla..."
        )


        response = http.request(

            "POST",

            url,

            timeout=600

        )


        # ====================================================
        # CONTROLLO RISPOSTA
        # ====================================================

        response_text = (

            response.data

            .decode(
                "utf-8"
            )

        )


        print(
            "Status:",
            response.status
        )


        if response.status >= 400:

            print(
                "Risposta Valhalla:"
            )

            print(
                response_text
            )

            raise RuntimeError(

                f"Valhalla ha restituito "
                f"HTTP {response.status} "
                f"durante il batch "
                f"{batch_start}-{batch_end}."

            )


        # ====================================================
        # PARSING JSON
        # ====================================================

        data = json.loads(

            response_text

        )


        # ====================================================
        # ESTRAZIONE MATRICE
        # ====================================================

        matrix = data.get(

            "sources_to_targets"

        )


        if matrix is None:

            raise RuntimeError(

                "Risposta Valhalla senza "
                "'sources_to_targets'."

            )


        # ====================================================
        # VERIFICA DIMENSIONI MATRICE
        # ====================================================

        if len(matrix) != len(
            batch_sources
        ):

            raise RuntimeError(

                f"Numero sources nella risposta "
                f"Valhalla ({len(matrix)}) diverso "
                f"dal numero di sources richieste "
                f"({len(batch_sources)})."

            )


        # ====================================================
        # ITERAZIONE SOURCES
        # ====================================================

        for source_index, source_id in enumerate(

            batch_source_ids

        ):


            source_results = matrix[

                source_index

            ]


            # ------------------------------------------------
            # VERIFICA TARGETS
            # ------------------------------------------------

            if len(source_results) != len(

                all_locations

            ):

                raise RuntimeError(

                    f"Numero targets nella risposta "
                    f"per source {source_id} "
                    f"non corretto."

                )


            # =================================================
            # ITERAZIONE TARGETS
            # =================================================

            for target_index, target_id in enumerate(

                resource_ids

            ):


                route = source_results[

                    target_index

                ]


                # --------------------------------------------
                # ROUTE CON ERRORE
                # --------------------------------------------

                if "error" in route:

                    print(

                        f"[WARNING] "
                        f"Errore route "
                        f"{source_id} -> "
                        f"{target_id}: "
                        f"{route['error']}"

                    )


                    routes.append(

                        (

                            source_id,

                            target_id,

                            mode,

                            None,

                            None,

                        )

                    )


                    continue


                # --------------------------------------------
                # DISTANZA
                # --------------------------------------------

                distance_km = route.get(

                    "distance"

                )


                # --------------------------------------------
                # TEMPO
                # --------------------------------------------

                time_seconds = route.get(

                    "time"

                )


                # --------------------------------------------
                # SALVA RISULTATO
                # --------------------------------------------

                routes.append(

                    (

                        source_id,

                        target_id,

                        mode,

                        distance_km,

                        time_seconds,

                    )

                )


        # ====================================================
        # PROGRESSO
        # ====================================================

        routes_completed = (

            batch_end

            * total_resources

        )


        print(

            f"Progresso: "
            f"{routes_completed} / "
            f"{total_routes} route"

        )


    # ========================================================
    # RISULTATO FINALE
    # ========================================================

    print(

        f"\nRoute calcolate: "
        f"{len(routes)}"

    )


    return routes
# ============================================================
# SALVA ROUTE NEL DATABASE
# ============================================================

def save_routes(
    conn,
    routes
):
    """
    Salva le route nel database.

    Utilizza ON CONFLICT per evitare duplicati.

    Se una route esiste già:

        source_id
        target_id
        mode

    viene aggiornata.
    """

    if not routes:

        return

    query = """

    INSERT INTO route_matrix (

        source_id,

        target_id,

        mode,

        distance_km,

        time_seconds

    )

    VALUES %s

    ON CONFLICT (
        source_id,
        target_id,
        mode
    )

    DO UPDATE SET

        distance_km = EXCLUDED.distance_km,

        time_seconds = EXCLUDED.time_seconds,

        created_at = CURRENT_TIMESTAMP;

    """

    with conn.cursor() as cursor:

        execute_values(

            cursor,

            query,

            routes,

            page_size=5000,

        )

    conn.commit()

    print(
        f"Salvate {len(routes)} route nel database."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "ROUTE MATRIX GENERATOR"
    )

    print(
        "=========================================="
    )

    # ========================================================
    # CONNESSIONE DATABASE
    # ========================================================

    conn = get_db_connection()

    try:

        # ====================================================
        # CREA TABELLA
        # ====================================================

        create_route_matrix_table(
            conn
        )

        # ====================================================
        # CARICA RISORSE
        # ====================================================

        resources = load_all_resources(
            conn
        )

        if not resources:

            raise RuntimeError(

                "Nessuna risorsa valida "
                "trovata nel database."

            )

        # ====================================================
        # CALCOLA OGNI MODALITÀ
        # ====================================================

        for mode in ROUTING_MODES:

            print(
                "\n"
                + "=" * 50
            )

            print(
                f"MODALITÀ: {mode}"
            )

            print(
                "=" * 50
            )

            # -----------------------------------------------
            # CALCOLA MATRICE
            # -----------------------------------------------

            routes = calculate_valhalla_matrix(

                resources,

                mode,

            )

            # -----------------------------------------------
            # SALVA NEL DATABASE
            # -----------------------------------------------

            save_routes(

                conn,

                routes,

            )

        print(
            "\n=========================================="
        )

        print(
            "COMPLETATO"
        )

        print(
            "=========================================="
        )

    finally:

        conn.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()