# Aula 07 - Conhecimento, Representação e Lógica
# Exemplo final visual:
# Sistema especialista para inspeção e triagem em esteira
#
# Python 3.10+
# Requer: matplotlib, matplotlib.animation

from dataclasses import dataclass
from typing import Callable, List, Set, Dict, Optional
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle


# =========================================================
# Estruturas de conhecimento
# =========================================================

@dataclass
class Rule:
    name: str
    condition: Callable[[Set[str]], bool]
    conclusion: str


class KnowledgeBase:
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Rule] = []
        self.log: List[str] = []

    def add_fact(self, fact: str):
        self.facts.add(fact)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def infer(self):
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.condition(self.facts) and rule.conclusion not in self.facts:
                    self.facts.add(rule.conclusion)
                    self.log.append(f"{rule.name} -> {rule.conclusion}")
                    changed = True


# =========================================================
# Peça em inspeção
# =========================================================

@dataclass
class Part:
    id: int
    temperature: float
    mass: float
    crack: bool
    dimensional_error: float
    color_tag: str

    x: float = 0.0
    y: float = 5.0
    speed: float = 0.12
    decision: Optional[str] = None
    inspected: bool = False
    diverted: bool = False

    def color(self):
        if self.decision == "peca_aprovada":
            return "#90be6d"
        elif self.decision == "peca_reinspecao":
            return "#f9c74f"
        elif self.decision == "peca_rejeitada":
            return "#f94144"
        return "#adb5bd"


# =========================================================
# Regras simbólicas
# =========================================================

def build_kb(part: Part) -> KnowledgeBase:
    kb = KnowledgeBase()

    # Conversão de medições em fatos simbólicos
    if part.temperature > 80:
        kb.add_fact("temperatura_alta")

    if part.mass < 48:
        kb.add_fact("massa_baixa")
    elif part.mass > 52:
        kb.add_fact("massa_alta")
    else:
        kb.add_fact("massa_ok")

    if part.crack:
        kb.add_fact("trinca_detectada")

    if abs(part.dimensional_error) > 0.60:
        kb.add_fact("desvio_dimensional_alto")
    elif abs(part.dimensional_error) > 0.25:
        kb.add_fact("desvio_dimensional_moderado")
    else:
        kb.add_fact("dimensional_ok")

    if part.color_tag == "vermelho":
        kb.add_fact("lote_critico")

    # Regras
    kb.add_rule(Rule(
        "R1",
        lambda f: "trinca_detectada" in f,
        "peca_rejeitada"
    ))

    kb.add_rule(Rule(
        "R2",
        lambda f: "desvio_dimensional_alto" in f,
        "peca_rejeitada"
    ))

    kb.add_rule(Rule(
        "R3",
        lambda f: "temperatura_alta" in f and "lote_critico" in f,
        "peca_reinspecao"
    ))

    kb.add_rule(Rule(
        "R4",
        lambda f: "massa_baixa" in f and "dimensional_ok" not in f,
        "peca_reinspecao"
    ))

    kb.add_rule(Rule(
        "R5",
        lambda f: "desvio_dimensional_moderado" in f and "trinca_detectada" not in f,
        "peca_reinspecao"
    ))

    kb.add_rule(Rule(
        "R6",
        lambda f: "massa_ok" in f and "dimensional_ok" in f and "trinca_detectada" not in f,
        "peca_aprovada"
    ))

    kb.infer()

    # Prioridade de decisão
    if "peca_rejeitada" in kb.facts:
        part.decision = "peca_rejeitada"
    elif "peca_reinspecao" in kb.facts:
        part.decision = "peca_reinspecao"
    elif "peca_aprovada" in kb.facts:
        part.decision = "peca_aprovada"
    else:
        part.decision = "peca_reinspecao"

    return kb


# =========================================================
# Geração de peças
# =========================================================

def generate_part(part_id: int) -> Part:
    return Part(
        id=part_id,
        temperature=random.uniform(45, 95),
        mass=random.uniform(46, 54),
        crack=random.random() < 0.18,
        dimensional_error=random.uniform(-0.8, 0.8),
        color_tag=random.choice(["azul", "verde", "vermelho"]),
        x=0.5,
        y=5.0
    )


# =========================================================
# Visualização
# =========================================================

def draw_factory(ax, parts: List[Part], current_part: Optional[Part], kb: Optional[KnowledgeBase], counters: Dict[str, int]):
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Esteira principal
    ax.add_patch(Rectangle((0.5, 4.5), 10.0, 1.0, facecolor="#ced4da", edgecolor="black"))
    # Desvio para reinspeção
    ax.add_patch(Rectangle((10.5, 5.0), 3.0, 2.2, facecolor="#dee2e6", edgecolor="black"))
    # Desvio para descarte
    ax.add_patch(Rectangle((10.5, 2.8), 3.0, 1.4, facecolor="#dee2e6", edgecolor="black"))

    ax.text(2.5, 5.8, "Entrada / Transporte", fontsize=11, fontweight="bold")
    ax.text(11.1, 7.5, "Reinspeção", fontsize=11, fontweight="bold")
    ax.text(11.1, 2.0, "Descarte", fontsize=11, fontweight="bold")
    ax.text(13.6, 5.8, "Aprovadas", fontsize=11, fontweight="bold")

    # Região do sensor
    ax.add_patch(Rectangle((6.0, 4.25), 1.2, 1.5, facecolor="none", edgecolor="#1d3557", linewidth=2.0))
    ax.text(6.6, 6.0, "Inspeção", ha="center", fontsize=10, fontweight="bold")

    # Desenhando peças
    for p in parts:
        if p.decision == "peca_reinspecao" and p.diverted:
            draw_y = 6.4
        elif p.decision == "peca_rejeitada" and p.diverted:
            draw_y = 3.4
        else:
            draw_y = p.y

        circ = Circle((p.x, draw_y), 0.28, facecolor=p.color(), edgecolor="black", linewidth=1.5)
        ax.add_patch(circ)
        ax.text(p.x, draw_y, str(p.id), ha="center", va="center", fontsize=8, fontweight="bold")

    # Painel lateral
    info_x = 14.0
    lines = [
        "RESUMO",
        "",
        f"Aprovadas: {counters['peca_aprovada']}",
        f"Reinspeção: {counters['peca_reinspecao']}",
        f"Rejeitadas: {counters['peca_rejeitada']}",
        "",
    ]

    if current_part is not None and kb is not None:
        lines += [
            f"PEÇA ATUAL: {current_part.id}",
            f"Temperatura: {current_part.temperature:.1f} °C",
            f"Massa: {current_part.mass:.1f} g",
            f"Trinca: {'sim' if current_part.crack else 'não'}",
            f"Desvio dim.: {current_part.dimensional_error:+.2f} mm",
            f"Lote: {current_part.color_tag}",
            "",
            "FATOS:",
        ]

        facts_sorted = sorted(kb.facts)
        lines += [f"• {f}" for f in facts_sorted[:8]]

        if len(facts_sorted) > 8:
            lines.append("• ...")

        lines += ["", "REGRAS DISPARADAS:"]
        if kb.log:
            lines += [f"• {r}" for r in kb.log]
        else:
            lines += ["• nenhuma"]

        lines += ["", f"DECISÃO: {current_part.decision}"]

    ax.text(info_x, 9.5, "\n".join(lines), va="top", fontsize=9)

    ax.set_title("Sistema especialista para inspeção e triagem em esteira", fontsize=14, fontweight="bold")


# =========================================================
# Simulação
# =========================================================

def main():
    random.seed(7)

    fig, ax = plt.subplots(figsize=(15, 6))

    parts: List[Part] = [generate_part(1)]
    current_part = parts[0]
    current_kb = None

    counters = {
        "peca_aprovada": 0,
        "peca_reinspecao": 0,
        "peca_rejeitada": 0,
    }

    next_part_id = 2
    sensor_x = 6.6
    exit_x = 14.0
    approved_y = 5.0

    def update(frame):
        nonlocal current_part, current_kb, next_part_id

        # gera nova peça quando houver espaço
        if len(parts) < 5:
            last_x = max([p.x for p in parts], default=2.0)
            if last_x > 2.4:
                parts.append(generate_part(next_part_id))
                next_part_id += 1

        for p in parts:
            # região antes da inspeção
            if p.x < sensor_x:
                p.x += p.speed

            # inspeção ocorre uma vez
            elif not p.inspected:
                current_part = p
                current_kb = build_kb(p)
                p.inspected = True
                p.x += p.speed

            # após inspeção, segue conforme decisão
            else:
                if p.decision == "peca_aprovada":
                    p.x += p.speed

                elif p.decision == "peca_reinspecao":
                    if not p.diverted and p.x >= 10.4:
                        p.diverted = True
                    p.x += p.speed

                elif p.decision == "peca_rejeitada":
                    if not p.diverted and p.x >= 10.4:
                        p.diverted = True
                    p.x += p.speed

        # contabiliza e remove peças que saíram da tela
        remaining = []
        for p in parts:
            if p.x >= exit_x:
                counters[p.decision] += 1
            else:
                remaining.append(p)
        parts[:] = remaining

        draw_factory(ax, parts, current_part, current_kb, counters)

    ani = animation.FuncAnimation(fig, update, frames=220, interval=120, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()