# Aula 06A - Busca Heurística: A* em Grid (versão mais visual)
# Python 3.10+ | Requer: matplotlib, numpy

import heapq
from typing import Tuple, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


Pos = Tuple[int, int]

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Grid:
    def __init__(self, grid: List[List[int]], start: Pos, goal: Pos):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, p: Pos) -> bool:
        return self.in_bounds(p) and self.grid[p[0]][p[1]] == 0

    def neighbors(self, p: Pos) -> List[Pos]:
        res = []
        for dr, dc in ACTIONS.values():
            nxt = (p[0] + dr, p[1] + dc)
            if self.is_free(nxt):
                res.append(nxt)
        return res


def reconstruct_path(came_from: Dict[Pos, Optional[Pos]], goal: Pos) -> Optional[List[Pos]]:
    if goal not in came_from:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def astar(grid: Grid):
    frontier = []
    heapq.heappush(frontier, (0, grid.start))

    came_from: Dict[Pos, Optional[Pos]] = {grid.start: None}
    g_cost: Dict[Pos, int] = {grid.start: 0}
    f_cost: Dict[Pos, int] = {grid.start: manhattan(grid.start, grid.goal)}

    closed: Set[Pos] = set()
    expansion_order: List[Pos] = []

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        closed.add(current)
        expansion_order.append(current)

        if current == grid.goal:
            break

        for nxt in grid.neighbors(current):
            tentative_g = g_cost[current] + 1

            if nxt not in g_cost or tentative_g < g_cost[nxt]:
                g_cost[nxt] = tentative_g
                f = tentative_g + manhattan(nxt, grid.goal)
                f_cost[nxt] = f
                heapq.heappush(frontier, (f, nxt))
                came_from[nxt] = current

    path = reconstruct_path(came_from, grid.goal)

    return {
        "path": path,
        "closed": closed,
        "came_from": came_from,
        "g_cost": g_cost,
        "f_cost": f_cost,
        "expansion_order": expansion_order,
    }


def build_display_matrix(grid: Grid, closed: Set[Pos], path: Optional[List[Pos]]) -> np.ndarray:
    """
    Códigos visuais:
    0 = livre
    1 = obstáculo
    2 = expandido
    3 = caminho
    4 = start
    5 = goal
    """
    display = np.zeros((grid.rows, grid.cols), dtype=int)

    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.grid[r][c] == 1:
                display[r, c] = 1

    for r, c in closed:
        if (r, c) != grid.start and (r, c) != grid.goal:
            display[r, c] = 2

    if path:
        for r, c in path:
            if (r, c) != grid.start and (r, c) != grid.goal:
                display[r, c] = 3

    sr, sc = grid.start
    gr, gc = grid.goal
    display[sr, sc] = 4
    display[gr, gc] = 5

    return display


def draw_arrow_path(ax, path: List[Pos]):
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dx = c2 - c1
        dy = r2 - r1

        ax.arrow(
            c1, r1,
            dx * 0.72, dy * 0.72,
            head_width=0.16,
            head_length=0.16,
            length_includes_head=True,
            linewidth=2
        )


def annotate_cells(ax, grid: Grid, g_cost: Dict[Pos, int], show_heuristic: bool = True):
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.grid[r][c] == 1:
                ax.text(c, r, "X", ha="center", va="center", fontsize=12, fontweight="bold")
            else:
                if show_heuristic:
                    h = manhattan((r, c), grid.goal)
                    label = f"h={h}"
                    if (r, c) in g_cost:
                        label = f"g={g_cost[(r, c)]}\n{label}"
                    ax.text(c, r, label, ha="center", va="center", fontsize=8)
                else:
                    ax.text(c, r, f"({r},{c})", ha="center", va="center", fontsize=8)


def plot_astar_visual(grid: Grid, result: dict):
    path = result["path"]
    closed = result["closed"]
    g_cost = result["g_cost"]

    display = build_display_matrix(grid, closed, path)

    cmap = ListedColormap([
        "#f8f9fa",  # livre
        "#2f2f2f",  # obstáculo
        "#a8dadc",  # expandido
        "#ffd166",  # caminho
        "#06d6a0",  # start
        "#ef476f",  # goal
    ])

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.2])

    ax = fig.add_subplot(gs[0, 0])
    info = fig.add_subplot(gs[0, 1])

    ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=5)

    ax.set_title("A* em Grid — obstáculos, expansão e caminho final", fontsize=14, fontweight="bold")
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.set_xticklabels(range(grid.cols))
    ax.set_yticklabels(range(grid.rows))
    ax.set_xlim(-0.5, grid.cols - 0.5)
    ax.set_ylim(grid.rows - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, grid.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.rows, 1), minor=True)
    ax.grid(which="minor", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_cells(ax, grid, g_cost, show_heuristic=True)

    if path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, linewidth=3)
        draw_arrow_path(ax, path)

    sr, sc = grid.start
    gr, gc = grid.goal

    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=12, fontweight="bold")

    info.axis("off")

    path_cost = len(path) - 1 if path else None
    expanded = len(closed)
    free_cells = sum(cell == 0 for row in grid.grid for cell in row)
    obstacle_cells = sum(cell == 1 for row in grid.grid for cell in row)

    summary = [
        "RESUMO DA EXECUÇÃO",
        "",
        f"Dimensão do grid: {grid.rows} x {grid.cols}",
        f"Células livres: {free_cells}",
        f"Obstáculos: {obstacle_cells}",
        "",
        f"Início: {grid.start}",
        f"Objetivo: {grid.goal}",
        "",
        f"Nós expandidos: {expanded}",
        f"Custo do caminho: {path_cost if path else 'não encontrado'}",
        "",
        "LEGENDA",
        "• Branco: célula livre",
        "• Preto: obstáculo",
        "• Azul claro: nó expandido",
        "• Amarelo: caminho final",
        "• Verde: início",
        "• Vermelho: objetivo",
        "",
        "LEITURA DIDÁTICA",
        "• g(n): custo acumulado",
        "• h(n): estimativa até o objetivo",
        "• f(n) = g(n) + h(n)",
        "• A* usa custo real + heurística",
    ]

    info.text(0.0, 1.0, "\n".join(summary), va="top", fontsize=11)

    plt.tight_layout()
    plt.show()


def main():
    grid_map = [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (4, 7)

    env = Grid(grid_map, start, goal)
    result = astar(env)

    path = result["path"]
    closed = result["closed"]

    print("=== AULA 06A - A* EM GRID (VERSÃO VISUAL) ===")
    if path:
        print(f"- custo do caminho: {len(path) - 1}")
    else:
        print("- nenhum caminho encontrado")
    print(f"- nós expandidos: {len(closed)}")

    plot_astar_visual(env, result)

    print("\nMensagem-chave:")
    print("- A* expande menos nós porque usa heurística.")
    print("- A heurística Manhattan aproxima a distância restante até o objetivo.")
    print("- A visualização ajuda a perceber que a busca não é aleatória: ela é guiada.")


if __name__ == "__main__":
    main()