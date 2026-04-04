# Aula 05 - Busca Clássica: DFS vs BFS em Mapa Urbano
# Exemplo robusto com visualização e gráficos
# Python 3.10+

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import time

Pos = Tuple[int, int]

ACTIONS: Dict[str, Pos] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


# =========================
# Ambiente
# =========================

class CityMap:
    def __init__(self, grid: List[List[int]], start: Pos, goal: Pos):
        """
        grid:
            0 = rua livre
            1 = obstáculo (prédio / rua bloqueada)
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, p: Pos) -> bool:
        r, c = p
        return self.in_bounds(p) and self.grid[r][c] == 0

    def neighbors(self, p: Pos) -> List[Pos]:
        res = []
        for dr, dc in ACTIONS.values():
            nxt = (p[0] + dr, p[1] + dc)
            if self.is_free(nxt):
                res.append(nxt)
        return res

    def render_text(self, path: Optional[List[Pos]] = None) -> str:
        path_set = set(path) if path else set()
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                p = (r, c)
                if p == self.start:
                    row.append("S")
                elif p == self.goal:
                    row.append("G")
                elif p in path_set:
                    row.append("*")
                elif self.grid[r][c] == 1:
                    row.append("#")
                else:
                    row.append(".")
            out.append(" ".join(row))
        return "\n".join(out)


# =========================
# Estrutura de resultado
# =========================

@dataclass
class SearchResult:
    path: Optional[List[Pos]]
    visited_count: int
    visit_order: Dict[Pos, int]
    frontier_sizes: List[int]
    max_frontier_size: int
    execution_time_ms: float


# =========================
# DFS
# =========================

def dfs(city: CityMap) -> SearchResult:
    start_time = time.perf_counter()

    stack = [(city.start, [city.start])]
    visited = set()
    visit_order = {}
    frontier_sizes = []
    visited_count = 0

    while stack:
        frontier_sizes.append(len(stack))
        current, path = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        visit_order[current] = visited_count + 1
        visited_count += 1

        if current == city.goal:
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                path=path,
                visited_count=visited_count,
                visit_order=visit_order,
                frontier_sizes=frontier_sizes,
                max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
                execution_time_ms=elapsed,
            )

        # reversed para manter comportamento visual mais previsível
        for nxt in reversed(city.neighbors(current)):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        path=None,
        visited_count=visited_count,
        visit_order=visit_order,
        frontier_sizes=frontier_sizes,
        max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
        execution_time_ms=elapsed,
    )


# =========================
# BFS
# =========================

def bfs(city: CityMap) -> SearchResult:
    start_time = time.perf_counter()

    queue = deque([(city.start, [city.start])])
    visited = {city.start}
    visit_order = {city.start: 1}
    frontier_sizes = []
    visited_count = 0
    step = 1

    while queue:
        frontier_sizes.append(len(queue))
        current, path = queue.popleft()
        visited_count += 1

        if current == city.goal:
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                path=path,
                visited_count=visited_count,
                visit_order=visit_order,
                frontier_sizes=frontier_sizes,
                max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
                execution_time_ms=elapsed,
            )

        for nxt in city.neighbors(current):
            if nxt not in visited:
                visited.add(nxt)
                step += 1
                visit_order[nxt] = step
                queue.append((nxt, path + [nxt]))

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        path=None,
        visited_count=visited_count,
        visit_order=visit_order,
        frontier_sizes=frontier_sizes,
        max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
        execution_time_ms=elapsed,
    )


# =========================
# Visualização do mapa
# =========================

def build_display_matrix(city: CityMap) -> np.ndarray:
    display = np.zeros((city.rows, city.cols))
    for r in range(city.rows):
        for c in range(city.cols):
            if city.grid[r][c] == 1:
                display[r, c] = -1
    return display


def plot_map(ax, city: CityMap, result: SearchResult, title: str, show_visit_numbers: bool = False):
    display = build_display_matrix(city)
    ax.imshow(display, cmap="Greys", origin="upper")

    ax.set_xticks(np.arange(-0.5, city.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, city.rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    if show_visit_numbers:
        for (r, c), order in result.visit_order.items():
            if (r, c) != city.start and (r, c) != city.goal:
                ax.text(c, r, str(order), ha="center", va="center", fontsize=6)

    if result.path:
        path_r = [p[0] for p in result.path]
        path_c = [p[1] for p in result.path]
        ax.plot(path_c, path_r, linewidth=3, marker="o")

    sr, sc = city.start
    gr, gc = city.goal
    ax.scatter(sc, sr, s=180, marker="s")
    ax.scatter(gc, gr, s=220, marker="*")

    ax.set_title(title)


def plot_exploration_heatmap(ax, city: CityMap, result: SearchResult, title: str):
    mat = np.full((city.rows, city.cols), np.nan)

    for r in range(city.rows):
        for c in range(city.cols):
            if city.grid[r][c] == 1:
                mat[r, c] = -1

    for (r, c), order in result.visit_order.items():
        mat[r, c] = order

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")

    image = np.ma.masked_where(mat == -1, mat)
    ax.imshow(image, cmap=cmap, origin="upper")

    ax.set_xticks(np.arange(-0.5, city.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, city.rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.set_title(title)


# =========================
# Gráficos de comparação
# =========================

def plot_metrics(ax, dfs_result: SearchResult, bfs_result: SearchResult):
    labels = ["Nós visitados", "Passos no caminho",
              "Pico da fronteira", "Tempo (ms)"]

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
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Comparação de Métricas")
    ax.legend()


def plot_frontier_evolution(ax, dfs_result: SearchResult, bfs_result: SearchResult):
    ax.plot(dfs_result.frontier_sizes, label="DFS")
    ax.plot(bfs_result.frontier_sizes, label="BFS")
    ax.set_title("Evolução do Tamanho da Fronteira")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Tamanho da fronteira")
    ax.legend()
    ax.grid(True)


# =========================
# Cenário maior
# =========================

def create_large_city_map() -> CityMap:
    grid = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (12, 14)
    return CityMap(grid, start, goal)


# =========================
# Relatório textual
# =========================

def print_summary(result: SearchResult, name: str):
    steps = len(result.path) - 1 if result.path else "-"
    print(f"\n[{name}]")
    print(f"Passos no caminho: {steps}")
    print(f"Nós visitados: {result.visited_count}")
    print(f"Pico da fronteira: {result.max_frontier_size}")
    print(f"Tempo de execução: {result.execution_time_ms:.3f} ms")


# =========================
# Execução principal
# =========================

def main():
    city = create_large_city_map()

    print("=== AULA 05 - Busca Clássica em IA ===\n")
    print("Cenário: planejamento de rota para carro autônomo")
    print("0 = rua livre | 1 = obstáculo\n")
    print("Mapa textual:")
    print(city.render_text())

    dfs_result = dfs(city)
    bfs_result = bfs(city)

    print_summary(dfs_result, "DFS")
    print_summary(bfs_result, "BFS")

    print("\n=== Comparação Final ===")
    print(
        f"DFS -> passos: {len(dfs_result.path)-1 if dfs_result.path else '-'} | "
        f"visitados: {dfs_result.visited_count} | "
        f"fronteira máx.: {dfs_result.max_frontier_size} | "
        f"tempo: {dfs_result.execution_time_ms:.3f} ms"
    )
    print(
        f"BFS -> passos: {len(bfs_result.path)-1 if bfs_result.path else '-'} | "
        f"visitados: {bfs_result.visited_count} | "
        f"fronteira máx.: {bfs_result.max_frontier_size} | "
        f"tempo: {bfs_result.execution_time_ms:.3f} ms"
    )

    print("\nMensagem-chave da aula:")
    print("• BFS garante caminho mínimo em número de passos.")
    print("• DFS pode encontrar uma solução diferente, sem garantir o menor caminho.")
    print("• O custo computacional depende da estratégia de exploração.")
    print("• Em aplicações reais, não basta encontrar a solução: importa também o custo para encontrá-la.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_map(
        axes[0, 0],
        city,
        dfs_result,
        f"DFS - Caminho\nPassos: {len(dfs_result.path)-1 if dfs_result.path else '-'}",
        show_visit_numbers=False
    )

    plot_map(
        axes[0, 1],
        city,
        bfs_result,
        f"BFS - Caminho\nPassos: {len(bfs_result.path)-1 if bfs_result.path else '-'}",
        show_visit_numbers=False
    )

    plot_metrics(axes[0, 2], dfs_result, bfs_result)

    plot_exploration_heatmap(
        axes[1, 0], city, dfs_result, "DFS - Ordem de Exploração")
    plot_exploration_heatmap(
        axes[1, 1], city, bfs_result, "BFS - Ordem de Exploração")
    plot_frontier_evolution(axes[1, 2], dfs_result, bfs_result)

    fig.suptitle("Busca Clássica em Mapa Urbano - DFS vs BFS", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
