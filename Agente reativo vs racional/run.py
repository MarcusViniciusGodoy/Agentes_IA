# Aula 01 - Agentes: Reativo vs Racional (Utilidade) em GridWorld
# Versão com visual bonito via matplotlib
# Python 3.10+ | Requer: matplotlib

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


Pos = Tuple[int, int]  # (row, col)

ACTIONS: Dict[str, Pos] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class StepResult:
    new_pos: Pos
    hit_wall_or_obstacle: bool
    reached_goal: bool


class GridWorld:
    def __init__(self, rows: int, cols: int, start: Pos, goal: Pos, obstacles: List[Pos]):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, p: Pos) -> bool:
        return self.in_bounds(p) and (p not in self.obstacles)

    def step(self, pos: Pos, action: str) -> StepResult:
        dr, dc = ACTIONS[action]
        cand = (pos[0] + dr, pos[1] + dc)

        if not self.is_free(cand):
            return StepResult(new_pos=pos, hit_wall_or_obstacle=True, reached_goal=(pos == self.goal))

        return StepResult(new_pos=cand, hit_wall_or_obstacle=False, reached_goal=(cand == self.goal))

    def to_matrix(self) -> List[List[int]]:
        """
        Matriz para plot:
          0 = livre
          1 = obstáculo
        """
        M = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for (r, c) in self.obstacles:
            M[r][c] = 1
        return M


# =========================
# Agentes
# =========================

class ReactiveAgent:
    """
    Agente reativo:
    escolhe a ação que mais reduz Manhattan (local), tentando alternativas se bloqueado.
    """
    def choose_action(self, env: GridWorld, pos: Pos) -> str:
        ranked = sorted(
            ACTIONS.keys(),
            key=lambda a: manhattan(self._peek(pos, a), env.goal)
        )
        for a in ranked:
            nxt = self._peek(pos, a)
            if env.is_free(nxt):
                return a
        return random.choice(list(ACTIONS.keys()))

    def _peek(self, pos: Pos, action: str) -> Pos:
        dr, dc = ACTIONS[action]
        return (pos[0] + dr, pos[1] + dc)


class UtilityAgent:
    """
    Agente racional (utilidade):
    escolhe ação que maximiza U = bônus_goal - dist - penal_hit - penal_loop
    """
    def __init__(self, w_goal=10.0, w_dist=1.0, w_hit=3.0, w_loop=1.5):
        self.w_goal = w_goal
        self.w_dist = w_dist
        self.w_hit = w_hit
        self.w_loop = w_loop

    def choose_action(self, env: GridWorld, pos: Pos, visited_recent: List[Pos]) -> str:
        scores = {a: self.utility(env, pos, a, visited_recent) for a in ACTIONS.keys()}
        best = max(scores.values())
        best_actions = [a for a, s in scores.items() if abs(s - best) < 1e-12]
        return random.choice(best_actions)

    def utility(self, env: GridWorld, pos: Pos, action: str, visited_recent: List[Pos]) -> float:
        res = env.step(pos, action)
        nxt = res.new_pos

        dist_term = -self.w_dist * manhattan(nxt, env.goal)
        goal_term = self.w_goal if res.reached_goal else 0.0
        hit_term = -self.w_hit if res.hit_wall_or_obstacle else 0.0
        loop_term = -self.w_loop if nxt in visited_recent else 0.0

        return goal_term + dist_term + hit_term + loop_term


# =========================
# Simulação
# =========================

@dataclass
class EpisodeStats:
    success: bool
    steps: int
    hits: int
    loops: int
    trajectory: List[Pos]


def run_episode(env: GridWorld, agent_type: str, max_steps: int = 80, seed: Optional[int] = None) -> EpisodeStats:
    if seed is not None:
        random.seed(seed)

    reactive = ReactiveAgent()
    rational = UtilityAgent()

    pos = env.start
    hits = 0
    loops = 0
    visited_recent: List[Pos] = []
    traj: List[Pos] = [pos]

    for t in range(1, max_steps + 1):
        if pos == env.goal:
            return EpisodeStats(True, t - 1, hits, loops, traj)

        if agent_type == "reativo":
            action = reactive.choose_action(env, pos)
        elif agent_type == "racional":
            action = rational.choose_action(env, pos, visited_recent)
        else:
            raise ValueError("agent_type deve ser 'reativo' ou 'racional'.")

        res = env.step(pos, action)

        if res.hit_wall_or_obstacle:
            hits += 1
        if res.new_pos in visited_recent:
            loops += 1

        pos = res.new_pos
        traj.append(pos)

        visited_recent.append(pos)
        if len(visited_recent) > 6:
            visited_recent.pop(0)

    return EpisodeStats(pos == env.goal, max_steps, hits, loops, traj)


def evaluate(env: GridWorld, episodes: int = 50, max_steps: int = 80, seed: int = 123) -> Dict[str, Dict[str, float]]:
    results = {"reativo": [], "racional": []}
    for agent in results.keys():
        for i in range(episodes):
            ep_seed = seed + i * 17
            results[agent].append(run_episode(env, agent, max_steps=max_steps, seed=ep_seed))

    def summarize(lst: List[EpisodeStats]) -> Dict[str, float]:
        success_rate = sum(1 for s in lst if s.success) / len(lst)
        steps_succ = [s.steps for s in lst if s.success]
        avg_steps = sum(steps_succ) / len(steps_succ) if steps_succ else float("inf")
        avg_hits = sum(s.hits for s in lst) / len(lst)
        avg_loops = sum(s.loops for s in lst) / len(lst)
        return {
            "sucesso": success_rate,
            "passos_medios_sucesso": avg_steps,
            "hits_medios": avg_hits,
            "loops_medios": avg_loops,
        }

    return {agent: summarize(lst) for agent, lst in results.items()}


# =========================
# Plot
# =========================

def plot_trajectory(ax, env: GridWorld, traj: List[Pos], title: str):
    M = env.to_matrix()  # 0 livre, 1 obstáculo

    ax.imshow(M, origin="upper")  # padrão: (0,0) no canto superior esquerdo
    ax.set_title(title)

    # marca start e goal
    ax.scatter(env.start[1], env.start[0], marker="s", s=120, edgecolors="k", label="Start")
    ax.scatter(env.goal[1], env.goal[0], marker="*", s=180, edgecolors="k", label="Goal")

    # trajetória: converter (r,c) -> (x=c, y=r)
    xs = [p[1] for p in traj]
    ys = [p[0] for p in traj]
    ax.plot(xs, ys, linewidth=2)

    # melhora visual (grid)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, linewidth=0.5)
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)  # mantém topo para baixo

    ax.legend(loc="upper right")


def plot_metrics(ax, metrics: Dict[str, Dict[str, float]]):
    agents = ["reativo", "racional"]
    success = [metrics[a]["sucesso"] for a in agents]
    steps = [metrics[a]["passos_medios_sucesso"] for a in agents]

    ax.bar([0, 1], success)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Reativo", "Racional"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Taxa de sucesso")

    # segundo eixo para passos médios
    ax2 = ax.twinx()
    ax2.plot([0, 1], steps, marker="o")
    ax2.set_ylabel("Passos médios (quando tem sucesso)")


def main():
    # Ambiente fixo
    rows, cols = 7, 10
    start = (5, 1)
    goal = (1, 8)
    obstacles = [
        (1, 3), (1, 4), (1, 5),
        (2, 5),
        (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 7), (5, 7),
    ]
    env = GridWorld(rows, cols, start, goal, obstacles)

    # Métricas (vários episódios)
    metrics = evaluate(env, episodes=60, max_steps=90, seed=2026)

    # Um episódio “demonstrativo” por agente (mesma seed para comparação)
    demo_seed = 7
    ep_rea = run_episode(env, "reativo", max_steps=90, seed=demo_seed)
    ep_rat = run_episode(env, "racional", max_steps=90, seed=demo_seed)

    print("=== AULA 01 (Matplotlib) - Resultados ===")
    for a in ["reativo", "racional"]:
        m = metrics[a]
        print(f"\n{a.upper()}")
        print(f"- sucesso: {100*m['sucesso']:.1f}%")
        print(f"- passos médios (sucesso): {m['passos_medios_sucesso']:.1f}")
        print(f"- hits médios: {m['hits_medios']:.1f}")
        print(f"- loops médios: {m['loops_medios']:.1f}")

    # Plot
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    plot_trajectory(
        ax1, env, ep_rea.trajectory,
        f"Reativo | sucesso={ep_rea.success} | passos={ep_rea.steps}"
    )
    plot_trajectory(
        ax2, env, ep_rat.trajectory,
        f"Racional | sucesso={ep_rat.success} | passos={ep_rat.steps}"
    )
    plot_metrics(ax3, metrics)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()