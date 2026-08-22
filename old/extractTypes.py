import json

def estrai_e_salva_unici(percorso_input_json, percorso_output_json):
    # Inizializziamo i set per memorizzare i valori unici
    resource_types_distinti = set()
    section_headers_distinti = set()

    try:
        # 1. Legge il file JSON originale
        with open(percorso_input_json, 'r', encoding='utf-8') as file:
            dati = json.load(file)
            
            if isinstance(dati, list):
                for elemento in dati:
                    res_type = elemento.get('resource_type')
                    if res_type:
                        resource_types_distinti.add(res_type.strip())
                        
                    sec_header = elemento.get('section_header')
                    if sec_header:
                        sec_header = sec_header.strip()
                        
                        # Se l'header inizia con "Come raggiungere", salviamo solo la radice
                        if sec_header.lower().startswith("come raggiungere"):
                            sec_header = "Come raggiungere"
                            
                        section_headers_distinti.add(sec_header)
            else:
                print("Errore: Il file JSON di input deve contenere una lista di oggetti.")
                return
        
        # 2. Crea la struttura dati da salvare (convertendo i set in liste ordinate)
        risultato = {
            "distinct_resource_types": sorted(list(resource_types_distinti)),
            "distinct_section_headers": sorted(list(section_headers_distinti))
        }
        
        # 3. Salva i dati strutturati in un nuovo file JSON
        with open(percorso_output_json, 'w', encoding='utf-8') as file_output:
            json.dump(risultato, file_output, ensure_ascii=False, indent=2)
            
        print(f"Successo! I valori distinti sono stati salvati in '{percorso_output_json}'.")

    except FileNotFoundError:
        print(f"Errore: Il file '{percorso_input_json}' non è stato trovato.")
    except json.JSONDecodeError:
        print("Errore: Il file di input non contiene un JSON valido.")

estrai_e_salva_unici('resources_chunks2.json', 'types_sections.json')