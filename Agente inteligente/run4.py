# Aula 08 - Agente Virtual Inteligente (Smart Assistant)
# Python 3.10+ | Requer: matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from collections import defaultdict
import math


# =========================================================
# Modelo probabilístico simples
# =========================================================

class SmartAgent:
    def __init__(self):
        self.priors = {
            "assistir_filme": 0.25,
            "estudar": 0.25,
            "dormir": 0.25,
            "sair": 0.25,
        }

        self.likelihoods = {
            "cansado": {
                "dormir": 0.7,
                "assistir_filme": 0.4,
                "estudar": 0.2,
                "sair": 0.3,
            },
            "tempo_livre": {
                "assistir_filme": 0.6,
                "sair": 0.6,
                "estudar": 0.3,
                "dormir": 0.4,
            },
            "chuva": {
                "assistir_filme": 0.7,
                "estudar": 0.5,
                "dormir": 0.5,
                "sair": 0.1,
            },
            "noite": {
                "dormir": 0.8,
                "assistir_filme": 0.5,
                "estudar": 0.4,
                "sair": 0.2,
            },
        }

    def decide(self, observations):
        scores = {}

        for action in self.priors:
            log_prob = math.log(self.priors[action])

            for obs in observations:
                prob = self.likelihoods.get(obs, {}).get(action, 0.3)
                log_prob += math.log(prob)

            scores[action] = log_prob

        max_log = max(scores.values())
        probs = {k: math.exp(v - max_log) for k, v in scores.items()}
        total = sum(probs.values())

        return {k: v / total for k, v in probs.items()}


# =========================================================
# Interface visual
# =========================================================

class SmartUI:
    def __init__(self):
        self.agent = SmartAgent()
        self.selected = set()

        self.fig = plt.figure(figsize=(12, 7))

        self.ax_checks = self.fig.add_axes([0.05, 0.4, 0.2, 0.4])
        self.ax_button = self.fig.add_axes([0.05, 0.25, 0.2, 0.08])
        self.ax_plot = self.fig.add_axes([0.3, 0.2, 0.65, 0.6])
        self.ax_text = self.fig.add_axes([0.3, 0.05, 0.65, 0.1])

        self.ax_text.axis("off")

        self.checks = CheckButtons(
            self.ax_checks,
            ["cansado", "tempo_livre", "chuva", "noite"],
            [False]*4
        )

        self.btn = Button(self.ax_button, "Decidir")

        self.checks.on_clicked(self.toggle)
        self.btn.on_clicked(self.run_agent)

        self.draw_empty()

    def toggle(self, label):
        if label in self.selected:
            self.selected.remove(label)
        else:
            self.selected.add(label)

    def draw_empty(self):
        self.ax_plot.clear()
        self.ax_plot.set_ylim(0, 1)
        self.ax_plot.set_title("Decisão do agente")

        self.ax_text.clear()
        self.ax_text.text(
            0, 0.5,
            "Selecione o contexto e clique em 'Decidir'",
            fontsize=11
        )

        self.fig.canvas.draw_idle()

    def run_agent(self, event):
        if not self.selected:
            return

        probs = self.agent.decide(self.selected)

        labels = list(probs.keys())
        values = list(probs.values())

        self.ax_plot.clear()
        self.ax_plot.bar(labels, values)
        self.ax_plot.set_ylim(0, 1)

        best = max(probs, key=probs.get)

        self.ax_text.clear()
        self.ax_text.text(
            0, 0.5,
            f"Decisão recomendada: {best}",
            fontsize=12,
            fontweight="bold"
        )

        self.fig.canvas.draw_idle()

    def run(self):
        plt.suptitle(
            "Agente Virtual Inteligente (Decisão sob Incerteza)",
            fontsize=14,
            fontweight="bold"
        )
        plt.show()


if __name__ == "__main__":
    ui = SmartUI()
    ui.run()
