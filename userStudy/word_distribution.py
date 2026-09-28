import json
import re
import matplotlib.pyplot as plt
import numpy as np


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text.lower()))


def plot_word_distribution(json_path: str = "resources_chunks7.json", bin_width: int = 25):
    # 1. Caricamento dati
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    word_counts = np.array([
        count_words(item["chunk"])
        for item in chunks
        if item.get("chunk")
    ])

    if len(word_counts) == 0:
        print("Nessun documento trovato.")
        return

    # 2. Calcolo esclusivo delle statistiche richieste
    n_docs = len(word_counts)
    mean_val = float(np.mean(word_counts))
    std_val = float(np.std(word_counts))
    min_val = int(np.min(word_counts))
    max_val = int(np.max(word_counts))

    # 3. Costruzione dei bin regolari
    bins = np.arange(
        min_val - (min_val % bin_width),
        max_val + bin_width * 2,
        bin_width
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    # 4. Istogramma a frequenze assolute
    ax.hist(
        word_counts,
        bins=bins,
        color="#4682b4",
        edgecolor="#1c3b5e",
        linewidth=1.0,
        alpha=0.75,
        label="Frequenza osservata"
    )

    # 5. Linea per la media
    ax.axvline(
        mean_val,
        color="#d9381e",
        linestyle="--",
        linewidth=1.8,
        label=f"Media ({mean_val:.1f})"
    )

    # 6. Riquadro statistiche
    stats_box = (
        f"Totale documenti: {n_docs}\n"
        f"Media ($\mu$): {mean_val:.1f} parole\n"
        f"Dev. Standard ($\sigma$): {std_val:.1f} parole\n"
        f"Minimo: {min_val} parole\n"
        f"Massimo: {max_val} parole"
    )

    ax.text(
        0.97, 0.93,
        stats_box,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9", edgecolor="#cccccc", alpha=0.95)
    )

    # 7. Etichette e griglia
    ax.set_title("Distribuzione della lunghezza dei documenti", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Numero di parole per documento", fontsize=10.5)
    ax.set_ylabel("Numero di documenti", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    plt.tight_layout()

    plt.savefig("istogramma_statistiche_richieste.png")
    plt.show()


if __name__ == "__main__":
    plot_word_distribution()