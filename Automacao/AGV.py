# Aula 07 - Conhecimento, Representação e Lógica
# Exemplo visual com movimento:
# AGV em ambiente industrial usando regras simbólicas
#
# Python 3.10+
# Requer: matplotlib, numpy

from dataclasses import dataclass
from typing import Callable, List, Set, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


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
# Ambiente
# =========================================================

FREE = 0
OBSTACLE = 1
CHARGE = 2
DELIVERY = 3

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "STOP": (0, 0),
}

DIR_ORDER = ["RIGHT", "DOWN", "LEFT", "UP"]


class FactoryEnv:
    def __init__(self, grid: List[List[int]], start: Tuple[int, int]):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.robot_pos = start
        self.robot_dir = "RIGHT"
        self.battery = 100
        self.carrying_item = False
        self.delivery_request = True

        self.charge_pos = self.find_cell(CHARGE)
        self.delivery_pos = self.find_cell(DELIVERY)

    def find_cell(self, value: int) -> Tuple[int, int]:
        pos = np.argwhere(self.grid == value)
        if len(pos) == 0:
            raise ValueError(f"Célula com valor {value} não encontrada.")
        return tuple(pos[0])

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, pos: Tuple[int, int]) -> bool:
        if not self.in_bounds(pos):
            return False
        return self.grid[pos] != OBSTACLE

    def cell_ahead(self) -> Tuple[int, int]:
        dr, dc = ACTIONS[self.robot_dir]
        return (self.robot_pos[0] + dr, self.robot_pos[1] + dc)

    def obstacle_ahead(self) -> bool:
        nxt = self.cell_ahead()
        return (not self.in_bounds(nxt)) or (self.grid[nxt] == OBSTACLE)

    def at_charge_station(self) -> bool:
        return self.robot_pos == self.charge_pos

    def at_delivery_station(self) -> bool:
        return self.robot_pos == self.delivery_pos

    def move(self, action: str):
        if action == "STOP":
            return

        if action in ACTIONS:
            self.robot_dir = action
            dr, dc = ACTIONS[action]
            nxt = (self.robot_pos[0] + dr, self.robot_pos[1] + dc)
            if self.is_free(nxt):
                self.robot_pos = nxt
                self.battery = max(0, self.battery - 2)

    def recharge(self):
        if self.at_charge_station():
            self.battery = min(100, self.battery + 8)

    def deliver(self):
        if self.at_delivery_station() and self.delivery_request:
            self.delivery_request = False
            self.carrying_item = False

    def load_item(self):
        # simplificação: começa com pedido ativo e item disponível
        if not self.carrying_item and self.delivery_request:
            self.carrying_item = True


# =========================================================
# Regras simbólicas
# =========================================================

def build_kb(percepts: Set[str]) -> KnowledgeBase:
    kb = KnowledgeBase()
    for p in percepts:
        kb.add_fact(p)

    kb.add_rule(Rule(
        name="R1",
        condition=lambda f: "bateria_baixa" in f and "na_estacao_carga" in f,
        conclusion="acao_recarregar"
    ))

    kb.add_rule(Rule(
        name="R2",
        condition=lambda f: "bateria_baixa" in f and "na_estacao_carga" not in f,
        conclusion="acao_ir_carga"
    ))

    kb.add_rule(Rule(
        name="R3",
        condition=lambda f: "pedido_entrega" in f and "bateria_baixa" not in f,
        conclusion="acao_ir_entrega"
    ))

    kb.add_rule(Rule(
        name="R4",
        condition=lambda f: "obstaculo_frente" in f,
        conclusion="acao_desviar"
    ))

    kb.add_rule(Rule(
        name="R5",
        condition=lambda f: "na_area_entrega" in f and "pedido_entrega" in f,
        conclusion="acao_entregar"
    ))

    kb.add_rule(Rule(
        name="R6",
        condition=lambda f: "sem_tarefa" in f,
        conclusion="acao_parar"
    ))

    kb.infer()
    return kb


# =========================================================
# Agente
# =========================================================

def get_percepts(env: FactoryEnv) -> Set[str]:
    facts = set()

    if env.obstacle_ahead():
        facts.add("obstaculo_frente")

    if env.battery <= 30:
        facts.add("bateria_baixa")

    if env.at_charge_station():
        facts.add("na_estacao_carga")

    if env.at_delivery_station():
        facts.add("na_area_entrega")

    if env.delivery_request:
        facts.add("pedido_entrega")

    if not env.delivery_request:
        facts.add("sem_tarefa")

    return facts


def step_toward(current: Tuple[int, int], target: Tuple[int, int]) -> str:
    r, c = current
    tr, tc = target

    if c < tc:
        return "RIGHT"
    if c > tc:
        return "LEFT"
    if r < tr:
        return "DOWN"
    if r > tr:
        return "UP"
    return "STOP"


def choose_action(env: FactoryEnv, kb: KnowledgeBase) -> str:
    f = kb.facts

    if "acao_recarregar" in f:
        return "RECHARGE"

    if "acao_entregar" in f:
        return "DELIVER"

    if "acao_desviar" in f:
        # estratégia simples: gira para uma direção livre
        for d in DIR_ORDER:
            dr, dc = ACTIONS[d]
            nxt = (env.robot_pos[0] + dr, env.robot_pos[1] + dc)
            if env.is_free(nxt):
                return d
        return "STOP"

    if "acao_ir_carga" in f:
        return step_toward(env.robot_pos, env.charge_pos)

    if "acao_ir_entrega" in f:
        return step_toward(env.robot_pos, env.delivery_pos)

    if "acao_parar" in f:
        return "STOP"

    return "STOP"


# =========================================================
# Visualização
# =========================================================

def build_display(env: FactoryEnv):
    display = np.copy(env.grid)
    return display


def draw_scene(ax, env: FactoryEnv, kb: KnowledgeBase, step_idx: int, action: str):
    ax.clear()

    display = build_display(env)

    cmap = plt.matplotlib.colors.ListedColormap([
        "#f8f9fa",  # livre
        "#343a40",  # obstáculo
        "#90e0ef",  # carga
        "#ffd166",  # entrega
    ])

    ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=3)

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels(range(env.cols))
    ax.set_yticklabels(range(env.rows))
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    rr, rc = env.robot_pos
    ax.plot(rc, rr, marker="o", markersize=18)
    ax.text(rc, rr, "AGV", ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_title("AGV com agente baseado em conhecimento", fontsize=14, fontweight="bold")

    summary = [
        f"Passo: {step_idx}",
        f"Posição: {env.robot_pos}",
        f"Direção: {env.robot_dir}",
        f"Bateria: {env.battery}%",
        f"Pedido de entrega: {'sim' if env.delivery_request else 'não'}",
        f"Ação escolhida: {action}",
        "",
        "Fatos atuais:",
    ]

    current_facts = sorted(kb.facts)
    if current_facts:
        summary.extend([f"• {fact}" for fact in current_facts])
    else:
        summary.append("• nenhum")

    summary.append("")
    summary.append("Inferências:")

    if kb.log:
        summary.extend([f"• {line}" for line in kb.log])
    else:
        summary.append("• nenhuma")

    ax.text(
        env.cols + 0.3, 0.2,
        "\n".join(summary),
        va="top",
        fontsize=10
    )

    ax.set_xlim(-0.5, env.cols + 5.8)


# =========================================================
# Simulação
# =========================================================

def main():
    grid = [
        [0, 0, 0, 0, 1, 0, 0, 3],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [2, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]

    env = FactoryEnv(grid=grid, start=(2, 0))
    env.load_item()

    fig, ax = plt.subplots(figsize=(14, 6))

    max_steps = 30

    def update(frame):
        percepts = get_percepts(env)
        kb = build_kb(percepts)
        action = choose_action(env, kb)

        if action == "RECHARGE":
            env.recharge()
        elif action == "DELIVER":
            env.deliver()
        else:
            env.move(action)

        draw_scene(ax, env, kb, frame, action)

    ani = animation.FuncAnimation(
        fig, update, frames=max_steps, interval=700, repeat=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()