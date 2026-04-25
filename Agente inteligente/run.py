# Aula 08 - Sistema de Recomendação Probabilístico
# Visual e interativo (Naive Bayes)
# Python 3.10+ | Requer: matplotlib

from collections import defaultdict
from typing import Dict, List, Set
import math

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button


# =========================================================
# Dataset genérico (preferências → categoria)
# =========================================================

DATASET = [
    (["acao", "rapido", "popular"], "filme_blockbuster"),
    (["acao", "ficcao", "longo"], "filme_blockbuster"),
    (["comedia", "curto", "popular"], "conteudo_leve"),
    (["comedia", "nicho"], "conteudo_leve"),
    (["educativo", "longo", "nicho"], "curso_online"),
    (["educativo", "curto"], "video_educativo"),
    (["ficcao", "nicho"], "conteudo_cult"),
    (["ficcao", "longo"], "serie"),
]


FEATURES = [
    "acao",
    "comedia",
    "ficcao",
    "educativo",
    "curto",
    "longo",
    "popular",
    "nicho",
]


# =========================================================
# Modelo Naive Bayes simples
# =========================================================

class NaiveBayes:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total = 0
        self.vocab = set()

    def fit(self, data):
        for features, label in data:
            self.class_counts[label] += 1
            self.total += 1

            for f in features:
                self.feature_counts[label][f] += 1
                self.vocab.add(f)

    def predict(self, features: List[str]) -> Dict[str, float]:
        results = {}
        vocab_size = len(self.vocab)

        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / self.total)

            total_features = sum(self.feature_counts[label].values())

            for f in features:
                count = self.feature_counts[label][f] + 1
                prob = count / (total_features + vocab_size)
                log_prob += math.log(prob)

            results[label] = log_prob

        max_log = max(results.values())
        probs = {k: math.exp(v - max_log) for k, v in results.items()}
        total = sum(probs.values())

        return {k: v / total for k, v in probs.items()}


# =========================================================
# Interface visual
# =========================================================

class RecommenderUI:
    def __init__(self):
        self.nb = NaiveBayes()
        self.nb.fit(DATASET)

        self.selected: Set[str] = set()

        self.fig = plt.figure(figsize=(12, 7))

        self.ax_checks = self.fig.add_axes([0.05, 0.4, 0.2, 0.4])
        self.ax_button = self.fig.add_axes([0.05, 0.25, 0.2, 0.08])
        self.ax_plot = self.fig.add_axes([0.3, 0.2, 0.65, 0.6])
        self.ax_text = self.fig.add_axes([0.3, 0.05, 0.65, 0.1])

        self.ax_text.axis("off")

        self.checks = CheckButtons(
            self.ax_checks,
            FEATURES,
            [False]*len(FEATURES)
        )

        self.btn = Button(self.ax_button, "Recomendar")

        self.checks.on_clicked(self.toggle)
        self.btn.on_clicked(self.recommend)

        self.draw_empty()

    def toggle(self, label):
        if label in self.selected:
            self.selected.remove(label)
        else:
            self.selected.add(label)

    def draw_empty(self):
        self.ax_plot.clear()
        self.ax_plot.set_title("Probabilidade de recomendação")
        self.ax_plot.set_ylim(0, 1)

        self.ax_text.clear()
        self.ax_text.text(0, 0.5,
            "Selecione características e clique em 'Recomendar'",
            fontsize=11)

        self.fig.canvas.draw_idle()

    def recommend(self, event):
        if not self.selected:
            return

        probs = self.nb.predict(list(self.selected))

        labels = list(probs.keys())
        values = list(probs.values())

        self.ax_plot.clear()
        self.ax_plot.bar(labels, values)
        self.ax_plot.set_ylim(0, 1)
        self.ax_plot.set_title("Probabilidade de recomendação")

        best = max(probs, key=probs.get)

        self.ax_text.clear()
        self.ax_text.text(0, 0.5,
            f"Recomendação: {best}",
            fontsize=12,
            fontweight="bold"
        )

        self.fig.canvas.draw_idle()

    def run(self):
        plt.suptitle(
            "Sistema de Recomendação Probabilístico (Naive Bayes)",
            fontsize=14,
            fontweight="bold"
        )
        plt.show()


# =========================================================
# Execução
# =========================================================

if __name__ == "__main__":
    ui = RecommenderUI()
    ui.run()