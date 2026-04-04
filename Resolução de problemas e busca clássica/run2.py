# Aula 05 - Busca Clássica: DFS vs BFS em Blocos Empilhados por Robô
# Exemplo visual com animação
# Python 3.10+

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# Tipos
# =========================================================

State = Tuple[Tuple[str, ...], ...]   # tupla de pilhas; topo = último elemento
Move = Tuple[int, int]                # mover topo da pilha i para a pilha j


# =========================================================
# Ambiente: mundo dos blocos
# =========================================================

class BlockWorld:
    def __init__(
        self,
        initial_state: State,
        goal_state: State,
        block_colors: Dict[str, str],
        max_stack_height: int = 4
    ):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.block_colors = block_colors
        self.num_stacks = len(initial_state)
        self.max_stack_height = max_stack_height

    def is_goal(self, state: State) -> bool:
        return state == self.goal_state

    def valid_moves(self, state: State) -> List[Move]:
        moves = []

        for src in range(self.num_stacks):
            if len(state[src]) == 0:
                continue

            for dst in range(self.num_stacks):
                if src == dst:
                    continue
                if len(state[dst]) >= self.max_stack_height:
                    continue
                moves.append((src, dst))

        return moves

    def apply_move(self, state: State, move: Move) -> State:
        src, dst = move
        stacks = [list(stack) for stack in state]

        block = stacks[src].pop()
        stacks[dst].append(block)

        return tuple(tuple(stack) for stack in stacks)

    def successors(self, state: State) -> List[Tuple[State, Move]]:
        return [(self.apply_move(state, move), move) for move in self.valid_moves(state)]

    def render_text(self, state: Optional[State] = None) -> str:
        if state is None:
            state = self.initial_state

        lines = []
        for i, stack in enumerate(state):
            lines.append(f"Pilha {i + 1}: {list(stack)}")
        return "\n".join(lines)


# =========================================================
# Estruturas de resultado
# =========================================================

@dataclass
class SearchResult:
    path: Optional[List[State]]
    moves: Optional[List[Move]]
    visited_count: int
    visit_order: Dict[State, int]
    frontier_sizes: List[int]
    max_frontier_size: int
    execution_time_ms: float


# =========================================================
# Busca em profundidade (DFS)
# =========================================================

def dfs(world: BlockWorld) -> SearchResult:
    start_time = time.perf_counter()

    stack = [(world.initial_state, [world.initial_state], [])]
    visited = set()
    visit_order = {}
    frontier_sizes = []
    visited_count = 0

    while stack:
        frontier_sizes.append(len(stack))
        current, path, moves = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        visited_count += 1
        visit_order[current] = visited_count

        if world.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                path=path,
                moves=moves,
                visited_count=visited_count,
                visit_order=visit_order,
                frontier_sizes=frontier_sizes,
                max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
                execution_time_ms=elapsed
            )

        # reversed para tornar o comportamento mais previsível visualmente
        succ = world.successors(current)
        for next_state, move in reversed(succ):
            if next_state not in visited:
                stack.append((next_state, path + [next_state], moves + [move]))

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        path=None,
        moves=None,
        visited_count=visited_count,
        visit_order=visit_order,
        frontier_sizes=frontier_sizes,
        max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
        execution_time_ms=elapsed
    )


# =========================================================
# Busca em largura (BFS)
# =========================================================

def bfs(world: BlockWorld) -> SearchResult:
    start_time = time.perf_counter()

    queue = deque([(world.initial_state, [world.initial_state], [])])
    visited = {world.initial_state}
    visit_order = {world.initial_state: 1}
    frontier_sizes = []
    visited_count = 0
    order_counter = 1

    while queue:
        frontier_sizes.append(len(queue))
        current, path, moves = queue.popleft()
        visited_count += 1

        if world.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                path=path,
                moves=moves,
                visited_count=visited_count,
                visit_order=visit_order,
                frontier_sizes=frontier_sizes,
                max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
                execution_time_ms=elapsed
            )

        for next_state, move in world.successors(current):
            if next_state not in visited:
                visited.add(next_state)
                order_counter += 1
                visit_order[next_state] = order_counter
                queue.append((next_state, path + [next_state], moves + [move]))

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        path=None,
        moves=None,
        visited_count=visited_count,
        visit_order=visit_order,
        frontier_sizes=frontier_sizes,
        max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
        execution_time_ms=elapsed
    )


# =========================================================
# Cenário didático
# =========================================================

def create_block_world() -> BlockWorld:
    # topo = último item de cada tupla
    initial_state: State = (
        ("A", "C"),
        ("B",),
        ("D",)
    )

    goal_state: State = (
        ("D", "C", "B", "A"),
        (),
        ()
    )

    block_colors = {
        "A": "#e76f51",
        "B": "#2a9d8f",
        "C": "#e9c46a",
        "D": "#457b9d",
    }

    return BlockWorld(
        initial_state=initial_state,
        goal_state=goal_state,
        block_colors=block_colors,
        max_stack_height=4
    )


# =========================================================
# Utilidades de impressão
# =========================================================

def format_move(move: Move) -> str:
    src, dst = move
    return f"P{src + 1} → P{dst + 1}"

def print_summary(result: SearchResult, name: str):
    if result.path is None:
        print(f"\n[{name}] sem solução encontrada.")
        return

    print(f"\n[{name}]")
    print(f"Passos na solução: {len(result.path) - 1}")
    print(f"Nós visitados: {result.visited_count}")
    print(f"Pico da fronteira: {result.max_frontier_size}")
    print(f"Tempo de execução: {result.execution_time_ms:.3f} ms")

    if result.moves:
        seq = " | ".join(format_move(m) for m in result.moves)
        print(f"Movimentos: {seq}")


# =========================================================
# Visualização das pilhas
# =========================================================

def draw_single_state(
    ax,
    world: BlockWorld,
    state: State,
    title: str,
    move_text: str = "",
    highlight_goal: bool = False
):
    ax.clear()

    num_stacks = world.num_stacks
    max_h = world.max_stack_height

    ax.set_xlim(0.5, num_stacks + 0.5)
    ax.set_ylim(0, max_h + 1.3)
    ax.set_xticks(range(1, num_stacks + 1))
    ax.set_xticklabels([f"Pilha {i}" for i in range(1, num_stacks + 1)], fontsize=11)
    ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")

    for i in range(num_stacks):
        x = i + 1

        # base
        ax.plot([x - 0.35, x + 0.35], [0.3, 0.3], linewidth=4)

        # eixo vertical ilustrativo
        ax.plot([x, x], [0.3, max_h + 0.2], linewidth=1.0, linestyle="--", alpha=0.4)

        stack = state[i]
        for h, block in enumerate(stack):
            y = h + 0.35
            rect = Rectangle(
                (x - 0.28, y),
                0.56,
                0.65,
                edgecolor="black",
                facecolor=world.block_colors.get(block, "lightgray"),
                linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(x, y + 0.325, block, ha="center", va="center",
                    fontsize=12, fontweight="bold")

    if highlight_goal and state == world.goal_state:
        title = title + "\nObjetivo alcançado"

    if move_text:
        ax.set_title(f"{title}\n{move_text}", fontsize=13)
    else:
        ax.set_title(title, fontsize=13)


# =========================================================
# Painel de métricas
# =========================================================

def plot_metrics(ax, dfs_result: SearchResult, bfs_result: SearchResult):
    labels = ["Nós visitados", "Passos", "Pico fronteira", "Tempo (ms)"]

    dfs_values = [
        dfs_result.visited_count,
        len(dfs_result.path) - 1 if dfs_result.path else 0,
        dfs_result.max_frontier_size,
        dfs_result.execution_time_ms,
    ]

    bfs_values = [
        bfs_result.visited_count,
        len(bfs_result.path) - 1 if bfs_result.path else 0,
        bfs_result.max_frontier_size,
        bfs_result.execution_time_ms,
    ]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, dfs_values, width, label="DFS")
    ax.bar(x + width / 2, bfs_values, width, label="BFS")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_title("Comparação de Métricas")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot_frontier_evolution(ax, dfs_result: SearchResult, bfs_result: SearchResult):
    ax.plot(dfs_result.frontier_sizes, label="DFS")
    ax.plot(bfs_result.frontier_sizes, label="BFS")
    ax.set_title("Evolução do Tamanho da Fronteira")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Fronteira")
    ax.grid(True, alpha=0.3)
    ax.legend()


# =========================================================
# Animação comparativa
# =========================================================

def animate_comparison(world: BlockWorld, dfs_result: SearchResult, bfs_result: SearchResult):
    if dfs_result.path is None or bfs_result.path is None:
        print("Não foi possível animar: uma das buscas não encontrou solução.")
        return

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.2])

    ax_dfs = fig.add_subplot(gs[0, 0])
    ax_bfs = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])
    ax_metrics = fig.add_subplot(gs[1, 0:2])
    ax_frontier = fig.add_subplot(gs[1, 2])

    plot_metrics(ax_metrics, dfs_result, bfs_result)
    plot_frontier_evolution(ax_frontier, dfs_result, bfs_result)

    ax_text.axis("off")

    max_frames = max(len(dfs_result.path), len(bfs_result.path))

    def state_at(path: List[State], frame: int) -> State:
        if frame < len(path):
            return path[frame]
        return path[-1]

    def move_text(moves: List[Move], frame: int) -> str:
        if frame == 0:
            return "Estado inicial"
        if frame - 1 < len(moves):
            return f"Movimento: {format_move(moves[frame - 1])}"
        return "Solução concluída"

    def update(frame: int):
        dfs_state = state_at(dfs_result.path, frame)
        bfs_state = state_at(bfs_result.path, frame)

        draw_single_state(
            ax_dfs,
            world,
            dfs_state,
            title=f"DFS — etapa {min(frame, len(dfs_result.path)-1)}",
            move_text=move_text(dfs_result.moves, frame),
            highlight_goal=True
        )

        draw_single_state(
            ax_bfs,
            world,
            bfs_state,
            title=f"BFS — etapa {min(frame, len(bfs_result.path)-1)}",
            move_text=move_text(bfs_result.moves, frame),
            highlight_goal=True
        )

        ax_text.clear()
        ax_text.axis("off")

        dfs_steps = len(dfs_result.path) - 1 if dfs_result.path else 0
        bfs_steps = len(bfs_result.path) - 1 if bfs_result.path else 0

        text = (
            "Planejamento de Movimentos do Robô\n\n"
            f"Estado inicial:\n{world.initial_state}\n\n"
            f"Estado objetivo:\n{world.goal_state}\n\n"
            f"DFS:\n"
            f"  • passos = {dfs_steps}\n"
            f"  • visitados = {dfs_result.visited_count}\n"
            f"  • fronteira máx. = {dfs_result.max_frontier_size}\n"
            f"  • tempo = {dfs_result.execution_time_ms:.3f} ms\n\n"
            f"BFS:\n"
            f"  • passos = {bfs_steps}\n"
            f"  • visitados = {bfs_result.visited_count}\n"
            f"  • fronteira máx. = {bfs_result.max_frontier_size}\n"
            f"  • tempo = {bfs_result.execution_time_ms:.3f} ms\n\n"
            "Leitura didática:\n"
            "• cada estado representa uma configuração das pilhas;\n"
            "• cada ação move apenas o bloco do topo;\n"
            "• BFS tende a achar a solução mais curta em número de movimentos;\n"
            "• DFS pode encontrar uma solução válida, mas não necessariamente mínima."
        )
        ax_text.text(0.0, 1.0, text, va="top", fontsize=11)

        fig.suptitle("Busca Clássica em Blocos Empilhados por Robô — DFS vs BFS", fontsize=16)

    anim = FuncAnimation(
        fig,
        update,
        frames=max_frames + 6,
        interval=1100,
        repeat=True
    )

    plt.tight_layout()
    plt.show()

    return anim


# =========================================================
# Execução principal
# =========================================================

def main():
    world = create_block_world()

    print("=== AULA 05 - Busca Clássica em IA ===\n")
    print("Cenário: robô reorganizando blocos em pilhas")
    print("Cada ação move o bloco do topo de uma pilha para outra.\n")

    print("Estado inicial:")
    print(world.render_text(world.initial_state))

    print("\nEstado objetivo:")
    print(world.render_text(world.goal_state))

    dfs_result = dfs(world)
    bfs_result = bfs(world)

    print_summary(dfs_result, "DFS")
    print_summary(bfs_result, "BFS")

    print("\nMensagem-chave da aula:")
    print("• O problema não é espacial como um mapa, mas continua sendo busca em espaço de estados.")
    print("• Cada configuração das pilhas é um estado.")
    print("• Cada movimento válido do robô é uma ação.")
    print("• BFS normalmente encontra a solução com menor número de movimentos.")
    print("• DFS pode explorar mais profundamente antes de encontrar o objetivo.")

    animate_comparison(world, dfs_result, bfs_result)


if __name__ == "__main__":
    main()