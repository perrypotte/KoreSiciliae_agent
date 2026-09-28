import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Inserisci l'URL di condivisione con permessi di visualizzazione pubblica
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CaXDxTbCGTADRdtzJ9vEarx7QRAGgnWYgg_xrphYKus/edit?gid=1314634099#gid=1314634099"

# Mappatura delle domande
QUESTIONS = {
    "Q1": "È stato facile capire come utilizzare l'applicazione.",
    "Q2": "L'assistente ha compreso correttamente le mie richieste.",
    "Q3": "Le risposte dell'assistente erano chiare e comprensibili.",
    "Q4": "L'assistente ha tenuto conto delle informazioni e preferenze fornite.",
    "Q5": "I suggerimenti mi hanno aiutato a scegliere i luoghi da visitare.",
    "Q6": "L'assistente ha offerto varietà senza eccessive ripetizioni.",
    "Q7": "Le informazioni sui luoghi proposti erano chiare e complete.",
    "Q8": "L'assistente ha gestito bene le regole e i limiti per gli spostamenti.",
    "Q9": "L'applicazione ha reso più semplice la pianificazione.",
    "Q10": "Esperienza complessiva soddisfacente.",
}


def parse_likert(val) -> float:
    """Mappa le risposte testuali della scala Likert in valori interi 1-5."""
    if pd.isna(val):
        return np.nan

    # Se il dato è già numerico
    try:
        num = float(val)
        if 1.0 <= num <= 5.0:
            return num
    except (ValueError, TypeError):
        pass

    s = str(val).strip().lower()

    # Gestione 'Neutrale' / 'Né d'accordo né in disaccordo'
    if any(k in s for k in ["neutr", "indifferent", "né", "ne'", "ne d"]):
        return 3.0
    # Gestione disaccordo
    if "fort" in s and "disaccord" in s:
        return 1.0
    if "disaccord" in s:
        return 2.0
    # Gestione accordo
    if "fort" in s and ("accord" in s or "daccord" in s):
        return 5.0
    if "accord" in s or "daccord" in s:
        return 4.0

    return np.nan


def get_csv_url(sheet_url: str) -> str:
    """Ricava l'endpoint diretto di export CSV dal link di condivisione."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("URL di Google Fogli non valido.")
    sheet_id = match.group(1)

    gid_match = re.search(r"[#&]gid=([0-9]+)", sheet_url)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def load_raw_data() -> pd.DataFrame:
    """Carica i dati dal foglio Google o genera un dataset sintetico di risposte grezze."""
    try:
        csv_url = get_csv_url(GOOGLE_SHEET_URL)
        return pd.read_csv(csv_url)
    except Exception:
        # Esempio sintetico con risposte qualitative grezze (righe = rispondenti)
#         mock_csv = """Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10
# D'accordo,Fortemente d'accordo,D'accordo,Neutrale,D'accordo,Fortemente d'accordo,D'accordo,Fortemente in disaccordo,D'accordo,Fortemente d'accordo
# Fortemente d'accordo,Fortemente d'accordo,D'accordo,D'accordo,D'accordo,D'accordo,Neutrale,In disaccordo,Fortemente d'accordo,Fortemente d'accordo
# D'accordo,D'accordo,Neutrale,D'accordo,Fortemente d'accordo,Fortemente d'accordo,D'accordo,Fortemente in disaccordo,D'accordo,D'accordo
# D'accordo,Fortemente d'accordo,D'accordo,Neutrale,D'accordo,D'accordo,D'accordo,In disaccordo,D'accordo,Fortemente d'accordo
# Neutrale,D'accordo,D'accordo,In disaccordo,D'accordo,Fortemente d'accordo,D'accordo,Fortemente in disaccordo,Fortemente d'accordo,D'accordo"""
#         return pd.read_csv(io.StringIO(mock_csv))
        return


# 1. Caricamento dati grezzi e mapping Likert
raw_df = load_raw_data()
print(len(raw_df))
# Se il foglio contiene una colonna Timestamp o ID rispondente, isola le colonne delle domande
question_cols = [c for c in raw_df.columns if re.search(r"Q\d+", c, re.I)]
if not question_cols:
    # Prende le prime 10 colonne utili se non esplicitamente nominate Q1-Q10
    question_cols = raw_df.columns[-10:].tolist()

# Applica la conversione numerica
numeric_df = raw_df[question_cols].map(parse_likert)
numeric_df.columns = [f"Q{i+1}" for i in range(len(question_cols))]

# 2. Calcolo aggregato delle metriche
means = numeric_df.mean().values
medians = numeric_df.median().values
stds = numeric_df.std().values
labels = numeric_df.columns.tolist()

# Chiusura del poligono per il grafico polare
n_axes = len(labels)
angles = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
angles += angles[:1]
means_closed = np.append(means, means[0])
medians_closed = np.append(medians, medians[0])

# 3. Tracciamento Spider Graph
fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

# Impostazione orientamento assi (Nord = 0 rad, senso orario)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Assegnazione etichette e limiti asse Likert (1 - 5)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
ax.set_ylim(1, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(
    ["1", "2", "3", "4", "5"], color="gray", size=9, fontweight="semibold"
)
ax.set_rlabel_position(25)

# Tracciamento Media e Mediana
ax.plot(
    angles,
    means_closed,
    color="#1f77b4",
    linewidth=2.2,
    label="Media",
)
ax.fill(angles, means_closed, color="#1f77b4", alpha=0.20)

ax.plot(
    angles,
    medians_closed,
    color="#d62728",
    linewidth=1.8,
    linestyle="--",
    label="Mediana",
)

# Tabella descrittiva a lato o legenda
legend_text = [
    f"{q_id}: {QUESTIONS.get(q_id, '')[:45]}..." for q_id in labels[:10]
]
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.15, 1.05),
    fontsize=9,
    title="Affermazioni",
)

plt.title(
    "Spider graph",
    size=13,
    weight="bold",
    pad=25,
)
plt.tight_layout()
plt.savefig("./spidergraph.png", dpi=300, bbox_inches="tight")
plt.show()