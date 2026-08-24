from langchain.tools import tool
import chainlit as cl
from sparse_retriever import SparseRetriever
from route_matrix_repository import RouteMatrixRepository
from geocoding import geocode_address
from valhalla_tool import calculate_routes, calculate_routes_matrix
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_postgres import PGVector
from typing import Literal
from planning_state import PlanningState

load_dotenv()

embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="koreSiciliae_resources_v6",
    connection=os.getenv("DATABASE2_URL"),
)

route_matrix_repository = RouteMatrixRepository()

sparse_retriever = SparseRetriever(
    collection_id=os.getenv('COLLECTION_ID', 'a8d572b6-ac8b-4d21-9132-6d2c808e2d6a')
)

############## HELPER ########################

# $in non funziona per come vengono salvati gli array di resource_types su postgres, quindi bisogna costruire un filtro OR manualmente
# Non voglio farlo fare all'agente perché aggiungo complessità inutile e rischio di errori

def build_filter(f):
    conditions = []
    excluded_document_ids=cl.user_session.get("state").get_selected_places_ids()


    # 1. Filtro su resource_types
    if "resource_types" in f and f["resource_types"]:
        conditions.append({
            "$or": [
                {"resource_types": v}
                for v in f["resource_types"]
            ]
        })

    #TODO al momento non puoi mettere lo stesso luogo, anche in giorni diversi
    # 2. Filtro per escludere i document_id ($ne) 
    if excluded_document_ids:
        for doc_id in excluded_document_ids:
            conditions.append({"document_id": {"$ne": doc_id}})

    # Ritorno coerente in base al numero di condizioni
    if len(conditions) == 0:
        return {}

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}

############## TOOLS ################################

@tool()
def get_place_info(document_id:str=None, title:str=None):
    """Ottiene i dettagli di una singola risorsa/luogo
Args:
document_id (str): ID univoco del documento di cui si vuole recuperare i dettagli.
title (str): Nome del punto di interesse di cui si vuole recuperare i dettagli quando non è disponibile l'id.
"""
    if document_id:
        document=vector_store.similarity_search(query=" ",k=1,filter={"document_id":document_id})
    else:
        document=vector_store.similarity_search(query=title,k=1)
    return document

@tool()
def add_place_to_itinerary(document_id: str, place_name: str,visit_duration: float,travel_time: float=0.0,travel_time_to_accomodation: float=0.0):
    """Aggiunge un luogo selezionato all'itinerario dell'utente per un giorno specifico.
Args:
document_id (str): ID univoco del documento del luogo da aggiungere (uuid).
place_name (str): Nome del luogo da aggiungere.
visit_duration (float): Durata della visita in ore.
travel_time (float): Tempo di viaggio in ore dalla tappa precedente.
travel_time_to_accomodation (float): Tempo necessario per tornare all'alloggio a partire da qui"""

    # Il controllo sul tempo è fatto anche a priori su search_places, ma lo mettiamo qui perché è aggirabile
    # TODO: comunque qui è da sistemare perché è fatto troppo semplice.
    state=cl.user_session.get("state")
    current_day_itinerary=state.get_current_day()
    remaining_time=current_day_itinerary.remaining_time

    if remaining_time>= visit_duration + travel_time:
        remaining_time = remaining_time - (visit_duration + travel_time)
        #cl.user_session.get("state", {})["remaining_time"] = remaining_time
        current_day_itinerary.add_place({
            "document_id": document_id,
            "place_name": place_name,
            "visit_duration": visit_duration,
            "travel_time": travel_time,
            "travel_time_to_accomodation": travel_time_to_accomodation,
        })
        #TODO ricorda che al momento il tempo di ritorno non viene visualizzato come tempo rimanente, dovrei considerarlo quando finisco la giornata oppure quando do l'itinerario completo
        current_day_itinerary.remaining_time=remaining_time
        state.set_current_day(current_day_itinerary)
        cl.user_session.set("state",state)
        return f"Il luogo '{place_name}' è stato aggiunto all'itinerario per il giorno {current_day_itinerary.day}. Tempo rimanente per il giorno: {round(current_day_itinerary.remaining_time)} ore."
    else:
        return f"Non c'è abbastanza tempo rimanente per aggiungere '{place_name}' all'itinerario. Tempo necessario: {visit_duration + travel_time} ore, tempo rimanente: {current_day_itinerary.remaining_time} ore. Tempo per tornare all'alloggio partendo dall'ultima tappa: {current_day_itinerary.selected_places[-1].travel_time_to_accomodation}"

@tool()
def finish_day(ending_place:str=None):
    """Segna il giorno corrente come completato e passa al giorno successivo.
Questo accade sotto richiesta dell'utente oppure quando il tempo rimanente per il giorno corrente è esaurito.
Args:
ending_place (str): Luogo in cui termina la giornata successiva (Es. Via esempio 17, Enna)."""
    state = cl.user_session.get("state")
    if state:
        if  state.get_current_day().day>=state.number_of_days:
            return "ToolError: Non puoi superare il numero di giorni stabilito in fase di pianificazione! Chiedi all'utente se vuole completare l'itinerario!"
        current_day_itinerary= state.get_current_day()
        state.next_day()
        if ending_place:
            state.add_new_day(current_day_itinerary.ending_place["display_name"],ending_place)
        else:
            state.add_new_day(current_day_itinerary.ending_place["display_name"],current_day_itinerary.ending_place["display_name"])
        cl.user_session.set("state",state)
        return "Tool eseguito con successo: nuova giornata pronta per accogliere nuove tappe"
    else:
        return "ToolError: Stato pianificazione non inizializzato"

@tool()
def finish_itinerary():
    """Completa l'itinerario, mostrandolo cosi all'utente e azzerando lo stato della pianificazione.
Può essere usato anche per resettare completamente l'itinerario per semplicità"""
    temp_state = cl.user_session.get("state")
    cl.user_session.set("state",PlanningState())
    
    return f"L'itinerario è stato completato e chiuso. Ecco il riepilogo da mostrare all'utente: {temp_state.to_prompt}. Per ogni giornata mostra anche luogo di partenza e luogo di arrivo e i tempi per andare da un luogo all'altro e per ogni giornata solo nell'ultimo luogo il tempo necessario per tornare all'alloggio"

@tool()
def update_planning_constraints(days: int, transport: Literal["auto", "pedestrian"], Daily_hours: float, Max_distance_km: float=70,current_day: int = 1,starting_place: str=None,ending_place:str=None):
    """Aggiorna o inizializza i vincoli di pianificazione nella sessione dell'utente.
Args:
days (int): Numero di giorni per il viaggio.
transport (str): Mezzo di trasporto preferito tra auto e camminare a piedi ("auto" o "pedestrian").
Max_distance_km (float): Distanza massima tra un luogo e l'altro (opzionale, default 70km).
Daily_hours (float): Ore giornaliere disponibili per le attività.
current_day (int): Giorno corrente dell'itinerario.
starting_place (str): Luogo in cui parte la PRIMISSIMA giornata (Es. Stazione centrale di Enna).
ending_place (str): Luoco in cui termina la PRIMISSIMA giornata (Es. Via esempio 17, Enna).
"""
    state=cl.user_session.get("state",PlanningState())
    #TODO implementare meglio la modifica della pianificazione che è abbastanza complicata, tenere conto di tutte le variabili
    # che influenzano anche tutto quello che è già stato fatto
    if(state.is_started()):
        state.max_distance_km=Max_distance_km
        cl.user_session.set("state",state)
    else:
        new_state=PlanningState(
                            True,
                            number_of_days=days,
                            transport=transport,
                            max_distance_km=Max_distance_km,
                            daily_hours=Daily_hours,
                            current_day=current_day) #0 per inizializzare e l'agente chiamerà per la prima volta finish day che dovrà
                                                     #inserire un nuovo itinerary day
        new_state.add_new_day(starting_place,ending_place)
        cl.user_session.set("state",new_state)
    return f"Vincoli di pianificazione aggiornati: {days} giorni, {transport} mezzo di trasporto, {Max_distance_km} km raggio massimo, {Daily_hours} ore giornaliere. Il giorno corrente è {current_day}"

@tool(response_format="content_and_artifact")
def search_places(query: str,filter: dict = None):
    """Ricava informazioni per aiutare l'utente a trovare luoghi da inserire nel proprio itinerario"""
    print("Raw filter input:", filter)
    raw_filter = filter
    filter=build_filter(filter) if filter else {}
    print(f"\n\nTool called with query: {query} and filter: {filter}\n\n")

    retrieved_docs = vector_store.similarity_search(query, k=30,filter=filter)
    dense_docs=retrieved_docs

    sparse_docs = sparse_retriever.invoke(
    query=query,
    k=30,
    resource_types=raw_filter
    )

    # for doc in sparse_docs[0:2]:
    #     print(f"\n\nSparse doc: {doc}\n\n")

    fused_docs = sparse_retriever.reciprocal_rank_fusion([dense_docs, sparse_docs])
    # print(len(fused_docs))

    client = NVIDIARerank(
        model="nv-rerank-qa-mistral-4b:1",
        top_n=10 
        # api_key=os.getenv("NVIDIA_API_KEY"),
        )

    reranked_fused_docs = client.compress_documents(
        query=query,
        # documents=[Document(page_content=passage) for passage in fused_docs],
        documents=fused_docs,
        )
    # print(len(reranked_fused_docs))

    state=cl.user_session.get("state")

    if state.planning_mode==False:
        serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in reranked_fused_docs
            )
        return serialized, reranked_fused_docs

    current_itinerary_day=state.get_current_day() #Tipo ItineraryDay
    max_distance_km=state.max_distance_km
    remaining_time=current_itinerary_day.remaining_time #hours
   # ============================================================
   # 1. Recuperiamo gli ID dei candidati
   # ============================================================
    reranked_reranked_fused_docs = reranked_fused_docs

    candidate_ids = [
        doc.metadata["document_id"]
        for doc in reranked_reranked_fused_docs
    ]

    # ============================================================
    # 2. Recuperiamo il punto di partenza della prossima tratta
    # ============================================================

    selected_places=current_itinerary_day.selected_places
    current_day=current_itinerary_day.day
    print(f"Current day {current_day}\n")
    print(f"Last place day: {selected_places[-1].get("day")}") if selected_places else None
    if selected_places:
        # Abbiamo già selezionato almeno un luogo:
        # la posizione corrente è l'ultimo luogo selezionato.
        last_selected = selected_places[-1]

        current_id = last_selected["document_id"]

        print(f"Current position = last selected place: {current_id}")

    else:
        # Nessun luogo ancora selezionato:
        # partiamo dall'indirizzo iniziale dinamico dell'utente.
        #TODO vedere perché mi usciva come lista non so perché mentre ending place no
        starting_point = current_itinerary_day.starting_place[0]

        # Non dovrebbe accadere teoricamente
        if not starting_point:
            print("Starting point non disponibile")
            reranked_fused_docs = []
        else:
            current_id = starting_point.get("id", "starting_point")#Prende default la seconda stringa

            print(
                f"Current position = dynamic starting point: "
                f"{starting_point}"
            )


    # ============================================================
    # 3. Calcoliamo current -> candidati
    # ============================================================
    current_to_candidates = {}
    transport=state.transport
    if selected_places:

        # --------------------------------------------------------
        # CASO NORMALE:
        # ultimo POI selezionato -> candidati
        #
        # Questi dati li abbiamo già nella route_matrix DB.
        # --------------------------------------------------------

        selected_place_id = selected_places[-1]["document_id"]
        print(
            f"\nGetting DB route matrix entries "
            f"from {selected_place_id}"
        )

        route_matrix_entries = route_matrix_repository.get_matrix(
            source_ids=[selected_place_id],
            target_ids=candidate_ids,
            mode=transport,
        )

        for entry in route_matrix_entries:
            current_to_candidates[entry.target_id] = {
                "distance_km": entry.distance_km,
                "time_seconds": entry.time_seconds,
            }

    else:

        # --------------------------------------------------------
        # PRIMA TAPPA:
        # starting point dinamico -> candidati
        #
        # Non possiamo usare la route_matrix DB perché lo starting
        # point non è un POI conosciuto.
        # --------------------------------------------------------

        #TODO vedere perché mi usciva come lista non so perché mentre ending place no
        starting_point = current_itinerary_day.starting_place[0]

        if starting_point:

            start_location = {
                "id": "starting_point",
                "lat": starting_point["lat"],
                "lon": starting_point["lon"],
            }

            candidate_locations = []

            for doc in reranked_reranked_fused_docs:

                md = doc.metadata

                candidate_locations.append({
                    "id": md["document_id"],
                    "lat": md["latitude"],
                    "lon": md["longitude"],
                })

            runtime_routes = calculate_routes(
                source=start_location,
                targets=candidate_locations,
                mode=transport,
            )

            for route in runtime_routes:
                current_to_candidates[route["target_id"]] = {
                    "distance_km": route["distance_km"],
                    "time_seconds": route["time_seconds"],
                }


    # ============================================================
    # 4. Calcoliamo candidato -> destinazione finale
    # ============================================================

    final_destination = current_itinerary_day.ending_place

    candidate_to_final = {}


    if final_destination:

        final_location = {
            "id": "final_destination",
            "lat": final_destination["lat"],
            "lon": final_destination["lon"],
        }
        candidate_locations = []

        for doc in reranked_reranked_fused_docs:

            md = doc.metadata

            candidate_locations.append({
                "id": md["document_id"],
                "lat": md["latitude"],
                "lon": md["longitude"],
            })

        # N sorgenti -> 1 destinazione finale
        print("LOG: Calcolo viaggio di ritorno all'alloggio...")
        runtime_routes = calculate_routes_matrix(
            sources=candidate_locations,
            targets=[final_location],
            mode=transport,
        )

        for route in runtime_routes:
            candidate_to_final[route["source_id"]] = {
                "distance_km": route["distance_km"],
                "time_seconds": route["time_seconds"],
            }
    # ============================================================
    # 5. Filtriamo i candidati
    # ============================================================

    filtered_docs = []

    for doc in reranked_reranked_fused_docs:

        document_id = doc.metadata["document_id"]

        # --------------------------------------------------------
        # Dati current -> candidato
        # --------------------------------------------------------

        current_route = current_to_candidates.get(document_id)

        if not current_route:
            print(
                f"Skipping {document_id}: "
                f"nessuna rotta current -> candidato"
            )
            continue

        current_distance = current_route["distance_km"]
        current_time_seconds = current_route["time_seconds"]


        # --------------------------------------------------------
        # Dati candidato -> destinazione finale
        # --------------------------------------------------------

        final_route = candidate_to_final.get(document_id)

        if not final_route:
            print(
                f"Skipping {document_id}: "
                f"nessuna rotta candidato -> destinazione finale"
            )
            continue

        final_distance = final_route["distance_km"]
        final_time_seconds = final_route["time_seconds"]


        # --------------------------------------------------------
        # Durata visita
        # --------------------------------------------------------

        import re

        raw_duration = doc.metadata.get("visit_duration", 0)

        match = re.search(r"\d+(?:\.\d+)?", str(raw_duration))

        visit_duration = float(match.group()) if match else 0.0


        # --------------------------------------------------------
        # Tempo totale necessario per rendere la tappa fattibile
        #
        # current
        #    ↓
        # candidate
        #    ↓ visita
        # candidate
        #    ↓
        # final destination
        # --------------------------------------------------------

        travel_to_candidate_hours = current_time_seconds / 3600
        travel_to_final_hours = final_time_seconds / 3600

        total_required_time = (
            travel_to_candidate_hours
            + visit_duration
            + travel_to_final_hours
        )


        # --------------------------------------------------------
        # Controllo tempo
        # --------------------------------------------------------

        if total_required_time > remaining_time:

            print(
                f"Skipping {document_id}: "
                f"required={total_required_time:.2f}h, "
                f"remaining={remaining_time:.2f}h"
            )

            continue


        # --------------------------------------------------------
        # Controllo distanza
        #
        # Qui assumiamo che Max_distance_km sia il massimo
        # consentito per una singola tratta.
        # --------------------------------------------------------

        if max_distance_km is not None:

            if current_distance > max_distance_km:

                print(
                    f"Skipping {document_id}: "
                    f"current -> candidate = "
                    f"{current_distance:.2f} km > "
                    f"{max_distance_km} km"
                )

                continue

            if final_distance > max_distance_km:

                print(
                    f"Skipping {document_id}: "
                    f"candidate -> final = "
                    f"{final_distance:.2f} km > "
                    f"{max_distance_km} km"
                )

                continue


        # --------------------------------------------------------
        # Arricchiamo il documento
        # --------------------------------------------------------

        doc.metadata["distance_km_from_last_selected"] = current_distance

        doc.metadata["travel_time_hours_from_last_selected"] = (
            travel_to_candidate_hours
        )

        doc.metadata["distance_km_to_final_destination"] = final_distance

        doc.metadata["travel_time_hours_to_final_destination"] = (
            travel_to_final_hours
        )

        doc.metadata["total_required_time_hours"] = total_required_time

        doc.metadata["visit_duration_hours"] = visit_duration

        doc.metadata["planning"]=True

        filtered_docs.append(doc)


    reranked_fused_docs = filtered_docs

    #DEBUG
    #print(reranked_fused_docs[0])
    #print(f"documenti: {len(reranked_fused_docs)}")
    
    #Caso semplificato, documenti non compatibili ai vincoli di pianificazione
    if(len(reranked_fused_docs)==0):
        return ("Non sono presenti documenti che rispettano la distanza o i limiti di tempo insieme ai luoghi di arrivo e partenza dell'utente. Proponi all'utente di estendere la distanza massima", [])
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_fused_docs
    )
    return serialized, reranked_fused_docs

# Lista esportabile centralizzata
ALL_TOOLS = [get_place_info,
             add_place_to_itinerary,
             finish_day,
             update_planning_constraints,
             search_places,
             finish_itinerary]