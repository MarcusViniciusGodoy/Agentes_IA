# Aula 02 - Agentes Inteligentes: PEAS + Determinístico vs Estocástico (Matplotlib)
# Mini-projeto VISUAL: Vacuum World (aspirador) com sujeira que pode reaparecer
# Python 3.10+ | Requer: matplotlib

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


Pos = Tuple[int, int]

ACTIONS: Dict[str, Pos] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "SUCK": (0, 0),  # ação de limpar
}


# =========================
# PEAS + classificação
# =========================

@dataclass
class PEAS:
    performance: List[str]
    environment: List[str]
    actuators: List[str]
    sensors: List[str]


def peas_for_vacuum() -> PEAS:
    return PEAS(
        performance=[
            "Maximizar área limpa (minimizar sujeira restante)",
            "Minimizar passos (eficiência)",
            "Evitar colisões com paredes",
        ],
        environment=[
            "Sala em grade 2D",
            "Cada célula pode estar limpa ou suja",
            "No modo estocástico, sujeira pode reaparecer",
        ],
        actuators=[
            "Mover: UP, DOWN, LEFT, RIGHT",
            "Limpar: SUCK",
        ],
        sensors=[
            "Posição atual",
            "Leitura local: célula atual está suja/limpa",
            "Mapa completo de sujeira (totalmente observável, para simplificar)",
        ],
    )


def classify_environment(dirt_spawn_prob: float) -> Dict[str, str]:
    return {
        "Observabilidade": "Totalmente observável (mapa de sujeira é conhecido)",
        "Determinismo": "Determinístico" if dirt_spawn_prob == 0.0 else f"Estocástico (spawn_sujeira={dirt_spawn_prob:.2f})",
        "Episódico/Sequencial": "Sequencial (limpar agora afeta estados futuros)",
        "Estático/Dinâmico": "Dinâmico (sujeira pode reaparecer)" if dirt_spawn_prob > 0 else "Estático (sujeira só diminui)",
        "Discreto/Contínuo": "Discreto (estados e ações discretos)",
        "Agentes": "Único agente",
    }


# =========================
# Ambiente
# =========================

@dataclass
class StepInfo:
    action: str
    hit_wall: bool
    cleaned: bool
    spawned: int
    dirty_cells: int
    clean_ratio: float


class VacuumWorld:
    """
    - grid de 0/1 (0 limpo, 1 sujo)
    - dirt_spawn_prob: a cada passo, cada célula limpa tem chance de virar suja
      (isso gera estocasticidade e dinâmica)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        start: Pos,
        dirt_init_prob: float = 0.35,
        dirt_spawn_prob: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.dirt_init_prob = dirt_init_prob
        self.dirt_spawn_prob = dirt_spawn_prob
        self.rng = random.Random(seed)

        self.grid: List[List[int]] = [[0]*cols for _ in range(rows)]
        self.pos: Pos = start

    def reset(self):
        self.pos = self.start
        self.grid = [
            [1 if self.rng.random() < self.dirt_init_prob else 0 for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        # garante que tem pelo menos um pouco de sujeira pra não ficar “trivial”
        if sum(sum(r) for r in self.grid) == 0:
            self.grid[self.rows//2][self.cols//2] = 1

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def dirty_cells(self) -> int:
        return sum(sum(r) for r in self.grid)

    def clean_ratio(self) -> float:
        total = self.rows * self.cols
        return 1.0 - (self.dirty_cells() / total)

    def spawn_dirt(self) -> int:
        """Em modo estocástico: pode surgir sujeira em células limpas."""
        if self.dirt_spawn_prob <= 0:
            return 0
        spawned = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0 and self.rng.random() < self.dirt_spawn_prob:
                    self.grid[r][c] = 1
                    spawned += 1
        return spawned

    def step(self, action: str) -> StepInfo:
        hit_wall = False
        cleaned = False

        if action == "SUCK":
            r, c = self.pos
            if self.grid[r][c] == 1:
                self.grid[r][c] = 0
                cleaned = True
        else:
            dr, dc = ACTIONS[action]
            cand = (self.pos[0] + dr, self.pos[1] + dc)
            if self.in_bounds(cand):
                self.pos = cand
            else:
                hit_wall = True

        spawned = self.spawn_dirt()
        d = self.dirty_cells()
        cr = self.clean_ratio()

        return StepInfo(
            action=action,
            hit_wall=hit_wall,
            cleaned=cleaned,
            spawned=spawned,
            dirty_cells=d,
            clean_ratio=cr,
        )


# =========================
# Agente (mesmo nos dois ambientes)
# =========================

class SimpleVacuumAgent:
    """
    Estratégia simples e bem explicável:
    1) Se a célula atual está suja -> SUCK
    2) Senão -> anda tentando visitar áreas novas
       (mantém memória curta de posições recentes)
    """

    def __init__(self, memory: int = 10):
        self.memory = memory
        self.recent: List[Pos] = []

    def choose_action(self, env: VacuumWorld) -> str:
        r, c = env.pos
        if env.grid[r][c] == 1:
            return "SUCK"

        # tenta mover para uma célula menos visitada (evita loop)
        candidates = []
        for a in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = ACTIONS[a]
            nxt = (r + dr, c + dc)
            if env.in_bounds(nxt):
                penalty = self.recent.count(nxt)
                candidates.append((penalty, a))

        candidates.sort(key=lambda x: x[0])
        best_penalty = candidates[0][0]
        best_actions = [a for p, a in candidates if p == best_penalty]
        return random.choice(best_actions)

    def update_memory(self, pos: Pos):
        self.recent.append(pos)
        if len(self.recent) > self.memory:
            self.recent.pop(0)


# =========================
# Simulação + animação
# =========================

@dataclass
class EpisodeStats:
    clean_ratio_over_time: List[float]
    dirty_over_time: List[int]
    hits: int
    cleaned_total: int
    spawned_total: int


def run_episode(env: VacuumWorld, steps: int = 120) -> EpisodeStats:
    agent = SimpleVacuumAgent(memory=12)

    hits = 0
    cleaned_total = 0
    spawned_total = 0
    clean_ratio = []
    dirty = []

    for _ in range(steps):
        a = agent.choose_action(env)
        info = env.step(a)

        if info.hit_wall:
            hits += 1
        if info.cleaned:
            cleaned_total += 1
        spawned_total += info.spawned

        clean_ratio.append(info.clean_ratio)
        dirty.append(info.dirty_cells)

        agent.update_memory(env.pos)

    return EpisodeStats(clean_ratio, dirty, hits, cleaned_total, spawned_total)


def animate_episode(ax, env: VacuumWorld, steps: int = 120, title: str = ""):
    agent = SimpleVacuumAgent(memory=12)

    im = ax.imshow(env.grid, origin="upper")
    ax.set_title(title)

    agent_dot = ax.scatter(
        env.pos[1], env.pos[0], s=120, edgecolors="k", label="Agente")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, linewidth=0.5)
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.legend(loc="upper right")

    plt.ion()
    for t in range(steps):
        a = agent.choose_action(env)
        env.step(a)
        agent.update_memory(env.pos)

        im.set_data(env.grid)
        agent_dot.set_offsets([[env.pos[1], env.pos[0]]])
        ax.set_xlabel(
            f"passo {t+1} | sujas={env.dirty_cells()} | limpo={100*env.clean_ratio():.1f}%")

        plt.pause(0.06)  # velocidade da animação
    plt.ioff()


def plot_metrics(ax, det_stats: EpisodeStats, sto_stats: EpisodeStats):
    ax.plot(det_stats.clean_ratio_over_time, label="Determinístico")
    ax.plot(sto_stats.clean_ratio_over_time, label="Estocástico")
    ax.set_title("Área limpa ao longo do tempo")
    ax.set_xlabel("passos")
    ax.set_ylabel("fração limpa")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linewidth=0.5)
    ax.legend(loc="best")


# =========================
# Execução
# =========================

def main():
    rows, cols = 8, 12
    start = (rows // 2, cols // 2)

    peas = peas_for_vacuum()
    print("=== AULA 02 (Vacuum World) - PEAS ===")
    print("\nP (Performance):")
    for x in peas.performance:
        print(f"- {x}")
    print("\nE (Environment):")
    for x in peas.environment:
        print(f"- {x}")
    print("\nA (Actuators):")
    for x in peas.actuators:
        print(f"- {x}")
    print("\nS (Sensors):")
    for x in peas.sensors:
        print(f"- {x}")

    env_det = VacuumWorld(
        rows, cols, start, dirt_init_prob=0.35, dirt_spawn_prob=0.0, seed=2026)
    env_sto = VacuumWorld(
        rows, cols, start, dirt_init_prob=0.35, dirt_spawn_prob=0.03, seed=2026)

    print("\n=== Classificação do Ambiente ===")
    print("\n[Determinístico]")
    for k, v in classify_environment(env_det.dirt_spawn_prob).items():
        print(f"- {k}: {v}")
    print("\n[Estocástico]")
    for k, v in classify_environment(env_sto.dirt_spawn_prob).items():
        print(f"- {k}: {v}")

    # Episódios para métricas (sem animação)
    env_det.reset()
    env_sto.reset()
    det_stats = run_episode(env_det, steps=120)
    sto_stats = run_episode(env_sto, steps=120)

    print("\n=== Resumo do episódio (120 passos) ===")
    print("[Determinístico]")
    print(f"- hits na parede: {det_stats.hits}")
    print(f"- limpou total: {det_stats.cleaned_total}")
    print(f"- sujeira gerada: {det_stats.spawned_total}")
    print(
        f"- área limpa final: {100*det_stats.clean_ratio_over_time[-1]:.1f}%")

    print("\n[Estocástico]")
    print(f"- hits na parede: {sto_stats.hits}")
    print(f"- limpou total: {sto_stats.cleaned_total}")
    print(f"- sujeira gerada: {sto_stats.spawned_total}")
    print(
        f"- área limpa final: {100*sto_stats.clean_ratio_over_time[-1]:.1f}%")

    # Plots principais
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # animações lado a lado (rodando sequencialmente é mais estável)
    env_det.reset()
    animate_episode(ax1, env_det, steps=120,
                    title="Determinístico (sem sujeira nova)")

    env_sto.reset()
    animate_episode(ax2, env_sto, steps=120,
                    title="Estocástico (sujeira reaparece)")

    plot_metrics(ax3, det_stats, sto_stats)

    plt.tight_layout()
    plt.show()

    print("\nMensagem-chave (pra fechar a aula):")
    print("- O agente é o MESMO. O ambiente mudou.")
    print("- No estocástico, o mundo 'bagunça de volta' (sujeira reaparece).")
    print("- A mesma política pode não 'vencer' um ambiente dinâmico/estocástico.")


if __name__ == "__main__":
    main()
