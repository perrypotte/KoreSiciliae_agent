import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# URL del foglio Google (se disponibile)
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CaXDxTbCGTADRdtzJ9vEarx7QRAGgnWYgg_xrphYKus/edit?gid=1314634099#gid=1314634099"

# 1. Definizione delle macro-categorie per l'analisi
CATEGORIE_DOMANDE = {
    "Q1": "Usabilità",                 # Facile da usare
    "Q9": "Usabilità",                 # Pianificazione semplice
    "Q2": "Capacità dell'agente",      # Comprensione richieste
    "Q4": "Capacità dell'agente",      # Mantenimento preferenze e contesto
    "Q8": "Capacità dell'agente",      # Gestione regole e vincoli
    "Q3": "Capacità dell'agente",      # Chiarezza risposte
    "Q5": "Qualità dei contenuti",     # Aiuto nella scelta dei luoghi
    "Q6": "Qualità dei contenuti",     # Varietà dei suggerimenti
    "Q7": "Qualità dei contenuti",     # Completezza informazioni
    "Q10": "Soddisfazione generale"    # Valutazione complessiva
}


def get_csv_url(sheet_url: str) -> str:
    """Ricava l'endpoint diretto di export CSV dal link di condivisione."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("URL di Google Fogli non valido.")
    sheet_id = match.group(1)

    gid_match = re.search(r"[#&]gid=([0-9]+)", sheet_url)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

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

# 2. Preparazione dei dati
raw_df = load_raw_data()
print(len(raw_df))
question_cols = [c for c in raw_df.columns if re.search(r"Q\d+", c, re.I)]
if not question_cols:
    # Prende le prime 10 colonne utili se non esplicitamente nominate Q1-Q10
    question_cols = raw_df.columns[-10:].tolist()
# Applica la conversione con il metodo aggiornato (.map)
numeric_df = raw_df[question_cols].map(parse_likert)
numeric_df.columns = [f"Q{i+1}" for i in range(len(question_cols))]
# Trasformazione in formato "Long" (necessario per Seaborn e categorizzazioni)
df_melted = numeric_df.melt(var_name="Domanda", value_name="Punteggio")

# Assegnazione della categoria a ciascuna riga
df_melted["Categoria"] = df_melted["Domanda"].map(CATEGORIE_DOMANDE)

# Ordinamento per Categoria per raggrupparle visivamente nel grafico
df_melted = df_melted.sort_values(by=["Categoria", "Domanda"])

# 3. Creazione del Grafico
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Traccia il boxplot orizzontale
ax = sns.boxplot(
    data=df_melted, 
    x="Punteggio", 
    y="Categoria", 
    hue="Categoria", 
    dodge=False,        # Evita che le barre si sfasino sull'asse Y
    palette="pastel",   # Colori tenui per i box
    width=0.6,
    showfliers=False,    # Nascondiamo gli outlier del boxplot perché useremo lo stripplot
    showmeans=True,  # Attiva il calcolo e la visualizzazione della media
    meanprops={
        "marker": "D",  # 'D' = Diamond (rombo)
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 6,
    },
    medianprops={"color": "red", "linewidth": 2},  # Mediana evidenziata in rosso
)

# Aggiunge i punti reali sopra il boxplot per vedere la densità delle risposte
sns.stripplot(
    data=df_melted, 
    x="Punteggio", 
    y="Categoria", 
    color="black", 
    alpha=0.5, 
    jitter=0.2,         # Disperde leggermente i punti sovrapposti
    size=5
)

# Formattazione
# plt.title("Distribuzione delle Risposte per Categoria (Scala Likert)", size=14, weight="bold", pad=20)
plt.title("Boxplot delle risposte per categoria", size=14, weight="bold", pad=20)
plt.xlabel("(1 = Fort. Disaccordo, 5 = Fort. D'accordo)", size=11)
plt.ylabel("Macro categorie", size=11)
plt.xlim(0.5, 5.5)
plt.xticks([1, 2, 3, 4, 5])

# Gestione della legenda per evitare duplicati
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, title="Macro-Categorie", loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.savefig("./boxplot.png", dpi=300, bbox_inches="tight")
plt.show()