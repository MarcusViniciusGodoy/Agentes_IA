# Aula 07 - Conhecimento, Representação e Lógica
# Sistema especialista interativo para diagnóstico em usinagem
# Python 3.10+ | Requer: matplotlib

from dataclasses import dataclass
from typing import Callable, List, Set, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import CheckButtons, Button


# =========================================================
# Estruturas básicas
# =========================================================

@dataclass
class Rule:
    name: str
    condition: Callable[[Set[str]], bool]
    conclusion: str
    description: str


class KnowledgeBase:
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Rule] = []
        self.inference_log: List[str] = []

    def add_fact(self, fact: str):
        if fact not in self.facts:
            self.facts.add(fact)
            self.inference_log.append(f"FATO: {fact}")

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def infer(self):
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.condition(self.facts) and rule.conclusion not in self.facts:
                    self.facts.add(rule.conclusion)
                    self.inference_log.append(
                        f"{rule.name}: {rule.description} -> {rule.conclusion}"
                    )
                    changed = True


# =========================================================
# Base de conhecimento
# =========================================================

def build_kb() -> KnowledgeBase:
    kb = KnowledgeBase()

    kb.add_rule(Rule(
        name="R1",
        condition=lambda f: "temperatura_alta" in f and "corrente_alta" in f,
        conclusion="sobrecarga_spindle",
        description="temperatura alta + corrente alta sugerem sobrecarga no spindle"
    ))

    kb.add_rule(Rule(
        name="R2",
        condition=lambda f: "vibracao_alta" in f and "rugosidade_alta" in f,
        conclusion="desgaste_ferramenta",
        description="vibração alta + rugosidade alta sugerem desgaste de ferramenta"
    ))

    kb.add_rule(Rule(
        name="R3",
        condition=lambda f: "sobrecarga_spindle" in f,
        conclusion="reduzir_avanco",
        description="sobrecarga no spindle recomenda reduzir avanço"
    ))

    kb.add_rule(Rule(
        name="R4",
        condition=lambda f: "desgaste_ferramenta" in f and "desvio_dimensional" in f,
        conclusion="troca_ferramenta",
        description="desgaste + desvio dimensional indicam troca da ferramenta"
    ))

    kb.add_rule(Rule(
        name="R5",
        condition=lambda f: "vibracao_alta" in f and "temperatura_alta" in f,
        conclusion="processo_instavel",
        description="vibração alta + temperatura alta indicam processo instável"
    ))

    kb.add_rule(Rule(
        name="R6",
        condition=lambda f: "processo_instavel" in f,
        conclusion="inspecao_imediata",
        description="processo instável exige inspeção imediata"
    ))

    kb.add_rule(Rule(
        name="R7",
        condition=lambda f: "troca_ferramenta" in f and "inspecao_imediata" in f,
        conclusion="risco_parada",
        description="troca necessária + inspeção imediata indicam risco de parada"
    ))

    return kb


# =========================================================
# Layout visual
# =========================================================

INPUT_FACTS = [
    "temperatura_alta",
    "corrente_alta",
    "vibracao_alta",
    "rugosidade_alta",
    "desvio_dimensional",
]

NODE_POSITIONS: Dict[str, Tuple[float, float]] = {
    # Entradas
    "temperatura_alta": (2, 8),
    "corrente_alta": (2, 6.5),
    "vibracao_alta": (2, 5),
    "rugosidade_alta": (2, 3.5),
    "desvio_dimensional": (2, 2),

    # Diagnósticos
    "sobrecarga_spindle": (6, 7),
    "desgaste_ferramenta": (6, 4),
    "processo_instavel": (6, 1.8),

    # Ações
    "reduzir_avanco": (10, 7),
    "troca_ferramenta": (10, 4),
    "inspecao_imediata": (10, 1.8),

    # Estado final
    "risco_parada": (13.5, 4),
}

EDGES = [
    ("temperatura_alta", "sobrecarga_spindle"),
    ("corrente_alta", "sobrecarga_spindle"),

    ("vibracao_alta", "desgaste_ferramenta"),
    ("rugosidade_alta", "desgaste_ferramenta"),

    ("sobrecarga_spindle", "reduzir_avanco"),

    ("desgaste_ferramenta", "troca_ferramenta"),
    ("desvio_dimensional", "troca_ferramenta"),

    ("vibracao_alta", "processo_instavel"),
    ("temperatura_alta", "processo_instavel"),

    ("processo_instavel", "inspecao_imediata"),

    ("troca_ferramenta", "risco_parada"),
    ("inspecao_imediata", "risco_parada"),
]


def format_label(text: str) -> str:
    return text.replace("_", "\n")


def draw_node(ax, x, y, text, active, kind="default"):
    width = 2.2
    height = 0.9

    if active:
        face = "#90e0ef"
        edge = "#023047"
        lw = 2.2
    else:
        face = "#f1f3f5"
        edge = "#868e96"
        lw = 1.2

    if kind == "final" and active:
        face = "#ffd6a5"

    rect = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face
    )
    ax.add_patch(rect)

    ax.text(
        x, y,
        format_label(text),
        ha="center", va="center",
        fontsize=9, fontweight="bold"
    )


def draw_graph(ax, active_facts: Set[str]):
    ax.clear()
    ax.set_xlim(0, 15)
    ax.set_ylim(0.5, 9)
    ax.axis("off")
    ax.set_title("Mapa de inferência simbólica", fontsize=13, fontweight="bold")

    # arestas
    for src, dst in EDGES:
        x1, y1 = NODE_POSITIONS[src]
        x2, y2 = NODE_POSITIONS[dst]

        edge_active = src in active_facts and dst in active_facts
        lw = 2.6 if edge_active else 1.0
        alpha = 1.0 if edge_active else 0.25

        ax.annotate(
            "",
            xy=(x2 - 1.15, y2),
            xytext=(x1 + 1.15, y1),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=lw,
                alpha=alpha
            )
        )

    for node, (x, y) in NODE_POSITIONS.items():
        kind = "final" if node == "risco_parada" else "default"
        draw_node(ax, x, y, node, node in active_facts, kind=kind)

    ax.text(1.3, 8.7, "Entradas observadas", fontsize=10, fontweight="bold")
    ax.text(5.3, 8.7, "Diagnósticos", fontsize=10, fontweight="bold")
    ax.text(9.5, 8.7, "Ações recomendadas", fontsize=10, fontweight="bold")
    ax.text(12.7, 8.7, "Estado final", fontsize=10, fontweight="bold")


def build_summary(kb: KnowledgeBase, selected_inputs: Set[str]) -> str:
    inferred = [f for f in kb.facts if f not in selected_inputs]

    lines = [
        "RESUMO",
        "",
        "Entradas selecionadas:",
    ]

    if selected_inputs:
        lines += [f"• {x}" for x in sorted(selected_inputs)]
    else:
        lines += ["• nenhuma"]

    lines += ["", "Fatos inferidos:"]

    if inferred:
        lines += [f"• {x}" for x in sorted(inferred)]
    else:
        lines += ["• nenhum"]

    lines += ["", "Regras disparadas:"]

    fired = [log for log in kb.inference_log if log.startswith("R")]
    if fired:
        lines += [f"• {log}" for log in fired]
    else:
        lines += ["• nenhuma"]

    return "\n".join(lines)


# =========================================================
# Interface interativa
# =========================================================

def run_interactive_demo():
    selected_inputs = set()

    fig = plt.figure(figsize=(15, 8))

    ax_graph = fig.add_axes([0.23, 0.10, 0.53, 0.82])
    ax_checks = fig.add_axes([0.03, 0.40, 0.15, 0.35])
    ax_button_infer = fig.add_axes([0.03, 0.28, 0.15, 0.06])
    ax_button_reset = fig.add_axes([0.03, 0.19, 0.15, 0.06])
    ax_text = fig.add_axes([0.79, 0.10, 0.19, 0.82])

    ax_text.axis("off")

    checks = CheckButtons(ax_checks, INPUT_FACTS, [False] * len(INPUT_FACTS))
    btn_infer = Button(ax_button_infer, "Inferir")
    btn_reset = Button(ax_button_reset, "Resetar")

    ax_checks.set_title("Sinais do processo", fontsize=11, fontweight="bold")

    draw_graph(ax_graph, selected_inputs)
    text_box = ax_text.text(
        0.0, 1.0,
        "Selecione sinais observados\n\ne clique em Inferir.",
        va="top", fontsize=10
    )

    def on_check(label):
        if label in selected_inputs:
            selected_inputs.remove(label)
        else:
            selected_inputs.add(label)

        draw_graph(ax_graph, selected_inputs)
        text_box.set_text(
            "Seleção atual:\n\n" +
            ("\n".join(f"• {x}" for x in sorted(selected_inputs)) if selected_inputs else "• nenhuma")
        )
        fig.canvas.draw_idle()

    def on_infer(event):
        kb = build_kb()
        for fact in selected_inputs:
            kb.add_fact(fact)
        kb.infer()

        draw_graph(ax_graph, kb.facts)
        text_box.set_text(build_summary(kb, selected_inputs))
        fig.canvas.draw_idle()

    def on_reset(event):
        selected_inputs.clear()

        for i in range(len(INPUT_FACTS)):
            if checks.get_status()[i]:
                checks.set_active(i)

        draw_graph(ax_graph, selected_inputs)
        text_box.set_text("Selecione sinais observados\n\ne clique em Inferir.")
        fig.canvas.draw_idle()

    checks.on_clicked(on_check)
    btn_infer.on_clicked(on_infer)
    btn_reset.on_clicked(on_reset)

    plt.suptitle(
        "Aula 07 - Sistema Especialista Interativo para Diagnóstico em Usinagem",
        fontsize=14,
        fontweight="bold"
    )
    plt.show()


if __name__ == "__main__":
    run_interactive_demo()