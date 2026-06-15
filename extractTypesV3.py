import json

def estrai_e_salva_unici(percorso_input_json, percorso_output_json):
    resource_types_distinti = set()
    section_headers_distinti = set()

    try:
        with open(percorso_input_json, 'r', encoding='utf-8') as file:
            dati = json.load(file)

            if isinstance(dati, list):
                for elemento in dati:
                    # --- RESOURCE TYPES (ORA LISTA) ---
                    res_types = elemento.get('resource_types')
                    if res_types and isinstance(res_types, list):
                        for res_type in res_types:
                            if res_type:
                                resource_types_distinti.add(res_type.strip())

                    # --- SECTION HEADER ---
                    sec_header = elemento.get('section_header')
                    if sec_header:
                        sec_header = sec_header.strip()

                        if sec_header.lower().startswith("come raggiungere"):
                            sec_header = "Come raggiungere"

                        section_headers_distinti.add(sec_header)
            else:
                print("Errore: Il file JSON di input deve contenere una lista di oggetti.")
                return

        risultato = {
            "distinct_resource_types": sorted(list(resource_types_distinti)),
            "distinct_section_headers": sorted(list(section_headers_distinti))
        }

        with open(percorso_output_json, 'w', encoding='utf-8') as file_output:
            json.dump(risultato, file_output, ensure_ascii=False, indent=2)

        print(f"Successo! I valori distinti sono stati salvati in '{percorso_output_json}'.")

    except FileNotFoundError:
        print(f"Errore: Il file '{percorso_input_json}' non è stato trovato.")
    except json.JSONDecodeError:
        print("Errore: Il file di input non contiene un JSON valido.")


estrai_e_salva_unici('resources_chunks3.json', 'types_sections3.json')