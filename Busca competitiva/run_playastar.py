# Aula 06D - Labirinto jogável com BFS e A* (versão corrigida e expandida)
# Python 3.10+ | Requer: matplotlib, numpy
#
# Recursos:
# - jogador se move com setas do teclado
# - botão para resolver com BFS
# - botão para resolver com A*
# - animação da expansão passo a passo
# - terrenos com custos diferentes
# - edição de obstáculos com o mouse
# - exibição opcional de g, h e f
# - comparação entre jogador, BFS e A*
#
# Controles:
# - setas: mover jogador
# - clique esquerdo: alterna obstáculo/livre
# - clique direito: alterna terreno livre/custo alto
# - botão "Resolver BFS"
# - botão "Resolver A*"
# - botão "Animar"
# - botão "Reiniciar"
# - botão "Trocar mapa"
# - botão "Mostrar custos"

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button

Pos = Tuple[int, int]

ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class SearchResult:
    name: str
    path: Optional[List[Pos]]
    closed: Set[Pos]
    came_from: Dict[Pos, Optional[Pos]]
    g_cost: Dict[Pos, int]
    f_cost: Dict[Pos, int]
    expansion_order: List[Pos]
    total_cost: Optional[int]


class GridGame:
    def __init__(self, grid: List[List[int]], start: Pos, goal: Pos):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_obstacle(self, p: Pos) -> bool:
        r, c = p
        return self.grid[r][c] == 1

    def is_free(self, p: Pos) -> bool:
        return self.in_bounds(p) and not self.is_obstacle(p)

    def cell_cost(self, p: Pos) -> int:
        r, c = p
        val = self.grid[r][c]
        if val == 2:
            return 5  # terreno caro
        return 1

    def neighbors(self, p: Pos) -> List[Pos]:
        res = []
        for dr, dc in ACTIONS.values():
            nxt = (p[0] + dr, p[1] + dc)
            if self.is_free(nxt):
                res.append(nxt)
        return res


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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


def path_total_cost(path: Optional[List[Pos]], env: GridGame) -> Optional[int]:
    if not path:
        return None
    total = 0
    for p in path[1:]:
        total += env.cell_cost(p)
    return total


def bfs(env: GridGame) -> SearchResult:
    frontier = deque([env.start])
    came_from: Dict[Pos, Optional[Pos]] = {env.start: None}
    closed: Set[Pos] = set()
    expansion_order: List[Pos] = []

    while frontier:
        current = frontier.popleft()

        if current in closed:
            continue

        closed.add(current)
        expansion_order.append(current)

        if current == env.goal:
            break

        for nxt in env.neighbors(current):
            if nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)

    path = reconstruct_path(came_from, env.goal)

    g_cost: Dict[Pos, int] = {}
    if path:
        g_cost[path[0]] = 0
        for i in range(1, len(path)):
            g_cost[path[i]] = i  # BFS mostra profundidade/passos

    return SearchResult(
        name="BFS",
        path=path,
        closed=closed,
        came_from=came_from,
        g_cost=g_cost,
        f_cost={},
        expansion_order=expansion_order,
        total_cost=path_total_cost(path, env),
    )


def astar(env: GridGame) -> SearchResult:
    frontier = []
    heapq.heappush(frontier, (0, env.start))

    came_from: Dict[Pos, Optional[Pos]] = {env.start: None}
    g_cost: Dict[Pos, int] = {env.start: 0}
    f_cost: Dict[Pos, int] = {env.start: manhattan(env.start, env.goal)}

    closed: Set[Pos] = set()
    expansion_order: List[Pos] = []

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in closed:
            continue

        closed.add(current)
        expansion_order.append(current)

        if current == env.goal:
            break

        for nxt in env.neighbors(current):
            tentative_g = g_cost[current] + env.cell_cost(nxt)

            if nxt not in g_cost or tentative_g < g_cost[nxt]:
                g_cost[nxt] = tentative_g
                f = tentative_g + manhattan(nxt, env.goal)
                f_cost[nxt] = f
                came_from[nxt] = current
                heapq.heappush(frontier, (f, nxt))

    path = reconstruct_path(came_from, env.goal)

    return SearchResult(
        name="A*",
        path=path,
        closed=closed,
        came_from=came_from,
        g_cost=g_cost,
        f_cost=f_cost,
        expansion_order=expansion_order,
        total_cost=path_total_cost(path, env),
    )


class MazeSearchApp:
    def __init__(self):
        self.maps = self.build_maps()
        self.map_index = 0
        self.env = self.load_map(self.map_index)

        self.player_pos = self.env.start
        self.player_path = [self.env.start]
        self.player_won = False

        self.astar_result: Optional[SearchResult] = None
        self.bfs_result: Optional[SearchResult] = None
        self.active_result: Optional[SearchResult] = None

        self.show_cost_labels = False
        self.animation_result: Optional[SearchResult] = None
        self.animation_step = 0
        self.animation_running = False
        self.anim: Optional[FuncAnimation] = None
        self._anim_ref: Optional[FuncAnimation] = None

        self.status_text = "Setas movem o jogador. Clique esquerdo cria/remove obstáculo."

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.4, 1.5])

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.info_ax = self.fig.add_subplot(gs[0, 1])

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        self.reset_ax = self.fig.add_axes([0.08, 0.02, 0.10, 0.05])
        self.bfs_ax = self.fig.add_axes([0.20, 0.02, 0.10, 0.05])
        self.astar_ax = self.fig.add_axes([0.32, 0.02, 0.10, 0.05])
        self.anim_ax = self.fig.add_axes([0.44, 0.02, 0.10, 0.05])
        self.map_ax = self.fig.add_axes([0.56, 0.02, 0.10, 0.05])
        self.cost_ax = self.fig.add_axes([0.68, 0.02, 0.12, 0.05])

        self.reset_button = Button(self.reset_ax, "Reiniciar")
        self.bfs_button = Button(self.bfs_ax, "Resolver BFS")
        self.astar_button = Button(self.astar_ax, "Resolver A*")
        self.anim_button = Button(self.anim_ax, "Animar")
        self.map_button = Button(self.map_ax, "Trocar mapa")
        self.cost_button = Button(self.cost_ax, "Mostrar custos")

        self.reset_button.on_clicked(self.on_reset_clicked)
        self.bfs_button.on_clicked(self.on_bfs_clicked)
        self.astar_button.on_clicked(self.on_astar_clicked)
        self.anim_button.on_clicked(self.on_anim_clicked)
        self.map_button.on_clicked(self.on_map_clicked)
        self.cost_button.on_clicked(self.on_cost_clicked)

    def build_maps(self):
        return [
            {
                "name": "Mapa 1",
                "grid": [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 2, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0],
                    [0, 2, 2, 2, 2, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "start": (0, 0),
                "goal": (4, 7),
            },
            {
                "name": "Mapa 2",
                "grid": [
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 2, 1, 0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 2, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [1, 1, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                ],
                "start": (0, 0),
                "goal": (5, 7),
            },
        ]

    def load_map(self, index: int) -> GridGame:
        m = self.maps[index]
        grid_copy = [row[:] for row in m["grid"]]
        return GridGame(grid_copy, m["start"], m["goal"])

    def reset_state(self):
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
            self._anim_ref = None

        self.env = self.load_map(self.map_index)
        self.player_pos = self.env.start
        self.player_path = [self.env.start]
        self.player_won = False
        self.astar_result = None
        self.bfs_result = None
        self.active_result = None
        self.animation_result = None
        self.animation_step = 0
        self.animation_running = False
        self.status_text = "Setas movem o jogador. Clique esquerdo cria/remove obstáculo."
        self.draw()

    def clear_search_results(self):
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
            self._anim_ref = None

        self.astar_result = None
        self.bfs_result = None
        self.active_result = None
        self.animation_result = None
        self.animation_step = 0
        self.animation_running = False

    def build_display_matrix(self) -> np.ndarray:
        """
        0 = livre
        1 = obstáculo
        2 = caminho do jogador
        3 = caminho do algoritmo ativo
        4 = início
        5 = objetivo
        6 = jogador atual
        7 = terreno caro
        8 = expansão animada
        """
        display = np.zeros((self.env.rows, self.env.cols), dtype=int)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.grid[r][c] == 1:
                    display[r, c] = 1
                elif self.env.grid[r][c] == 2:
                    display[r, c] = 7

        for r, c in self.player_path:
            if (r, c) != self.env.start and (r, c) != self.env.goal:
                display[r, c] = 2

        if self.active_result and self.active_result.path:
            for r, c in self.active_result.path:
                if (r, c) != self.env.start and (r, c) != self.env.goal:
                    display[r, c] = 3

        if self.animation_result:
            upto = min(self.animation_step, len(self.animation_result.expansion_order))
            for r, c in self.animation_result.expansion_order[:upto]:
                if (r, c) not in (self.env.start, self.env.goal):
                    if display[r, c] in (0, 7):
                        display[r, c] = 8

        sr, sc = self.env.start
        gr, gc = self.env.goal
        pr, pc = self.player_pos

        display[sr, sc] = 4
        display[gr, gc] = 5
        display[pr, pc] = 6

        return display

    def draw_grid(self):
        self.ax.clear()
        display = self.build_display_matrix()

        cmap = ListedColormap([
            "#f8f9fa",  # livre
            "#2f2f2f",  # obstáculo
            "#a8dadc",  # caminho jogador
            "#ffd166",  # caminho algoritmo
            "#06d6a0",  # início
            "#ef476f",  # objetivo
            "#118ab2",  # jogador
            "#f4a261",  # terreno caro
            "#cdb4db",  # expansão
        ])

        self.ax.imshow(display, cmap=cmap, origin="upper", vmin=0, vmax=8)

        self.ax.set_xticks(range(self.env.cols))
        self.ax.set_yticks(range(self.env.rows))
        self.ax.set_xticklabels(range(self.env.cols))
        self.ax.set_yticklabels(range(self.env.rows))
        self.ax.set_xlim(-0.5, self.env.cols - 0.5)
        self.ax.set_ylim(self.env.rows - 0.5, -0.5)

        self.ax.set_xticks(np.arange(-0.5, self.env.cols, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.env.rows, 1), minor=True)
        self.ax.grid(which="minor", linewidth=1.2)
        self.ax.tick_params(which="minor", bottom=False, left=False)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                pos = (r, c)
                val = self.env.grid[r][c]

                if val == 1:
                    self.ax.text(c, r, "X", ha="center", va="center", fontsize=12, fontweight="bold")
                elif self.show_cost_labels and self.active_result:
                    g = self.active_result.g_cost.get(pos)
                    h = manhattan(pos, self.env.goal)
                    f = self.active_result.f_cost.get(pos)

                    if self.active_result.name == "BFS":
                        if g is not None:
                            label = f"g={g}\nh={h}"
                        else:
                            label = f"h={h}"
                    else:
                        if g is not None and f is not None:
                            label = f"g={g}\nh={h}\nf={f}"
                        else:
                            label = f"h={h}"

                    self.ax.text(c, r, label, ha="center", va="center", fontsize=7)
                else:
                    if val == 2:
                        self.ax.text(c, r, "C5", ha="center", va="center", fontsize=9, fontweight="bold")
                    else:
                        self.ax.text(c, r, f"({r},{c})", ha="center", va="center", fontsize=7, alpha=0.55)

        sr, sc = self.env.start
        gr, gc = self.env.goal
        pr, pc = self.player_pos

        self.ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")
        self.ax.text(gc, gr, "G", ha="center", va="center", fontsize=12, fontweight="bold")
        self.ax.text(pc, pr, "P", ha="center", va="center", fontsize=12, fontweight="bold")

        self.ax.set_title(
            f"Labirinto jogável com BFS e A* — {self.maps[self.map_index]['name']}\n{self.status_text}",
            fontsize=13,
            fontweight="bold",
        )

    def draw_paths_overlay(self):
        if len(self.player_path) > 1:
            px = [p[1] for p in self.player_path]
            py = [p[0] for p in self.player_path]
            self.ax.plot(px, py, linewidth=2)

        if self.active_result and self.active_result.path:
            px = [p[1] for p in self.active_result.path]
            py = [p[0] for p in self.active_result.path]
            self.ax.plot(px, py, linewidth=3)

            for i in range(len(self.active_result.path) - 1):
                r1, c1 = self.active_result.path[i]
                r2, c2 = self.active_result.path[i + 1]
                dx = c2 - c1
                dy = r2 - r1
                self.ax.arrow(
                    c1,
                    r1,
                    dx * 0.70,
                    dy * 0.70,
                    head_width=0.12,
                    head_length=0.12,
                    length_includes_head=True,
                    linewidth=1.5,
                )

    def draw_info_panel(self):
        self.info_ax.clear()
        self.info_ax.axis("off")

        player_steps = len(self.player_path) - 1
        player_cost = path_total_cost(self.player_path, self.env)

        bfs_cost = self.bfs_result.total_cost if self.bfs_result else None
        astar_cost = self.astar_result.total_cost if self.astar_result else None
        bfs_exp = len(self.bfs_result.closed) if self.bfs_result else None
        astar_exp = len(self.astar_result.closed) if self.astar_result else None

        lines = [
            "PAINEL DIDÁTICO",
            "",
            f"Mapa: {self.maps[self.map_index]['name']}",
            f"Início: {self.env.start}",
            f"Objetivo: {self.env.goal}",
            "",
            f"Passos do jogador: {player_steps}",
            f"Custo do jogador: {player_cost}",
            f"Posição atual: {self.player_pos}",
            "",
            f"BFS custo: {bfs_cost if bfs_cost is not None else '--'}",
            f"BFS nós expandidos: {bfs_exp if bfs_exp is not None else '--'}",
            f"A* custo: {astar_cost if astar_cost is not None else '--'}",
            f"A* nós expandidos: {astar_exp if astar_exp is not None else '--'}",
            "",
            "LEGENDA",
            "• Branco: livre",
            "• Preto: obstáculo",
            "• Laranja: terreno custo 5",
            "• Azul claro: caminho jogador",
            "• Amarelo: caminho do algoritmo",
            "• Roxo: expansão animada",
            "",
            "LEITURA DIDÁTICA",
            "• BFS minimiza passos",
            "• A* usa custo real + heurística",
            "• g(n): custo acumulado",
            "• h(n): Manhattan",
            "• f(n) = g(n) + h(n)",
        ]

        if bfs_cost is not None and astar_cost is not None:
            lines += [
                "",
                "COMPARAÇÃO BFS x A*",
                f"Diferença de custo: {bfs_cost - astar_cost}",
                "• BFS pode achar menos passos,",
                "  mas ignorar terreno caro",
                "• A* tende a preferir custo menor",
            ]

        self.info_ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)

    def draw(self):
        self.draw_grid()
        self.draw_paths_overlay()
        self.draw_info_panel()
        self.fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.12, wspace=0.18)
        self.fig.canvas.draw_idle()

    def try_move(self, direction: str):
        if self.player_won or self.animation_running:
            return

        dr, dc = ACTIONS[direction]
        nxt = (self.player_pos[0] + dr, self.player_pos[1] + dc)

        if not self.env.is_free(nxt):
            self.status_text = "Movimento inválido: borda ou obstáculo."
            self.draw()
            return

        self.player_pos = nxt
        self.player_path.append(nxt)
        step_cost = self.env.cell_cost(nxt)
        self.status_text = f"Jogador moveu para {nxt} com custo {step_cost}."

        if self.player_pos == self.env.goal:
            self.player_won = True
            self.status_text = "Você chegou ao objetivo."

        self.draw()

    def cell_from_event(self, event) -> Optional[Pos]:
        if event.inaxes != self.ax:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        c = int(event.xdata)
        r = int(event.ydata)

        if not (0 <= r < self.env.rows and 0 <= c < self.env.cols):
            return None

        return (r, c)

    def on_mouse_click(self, event):
        pos = self.cell_from_event(event)
        if pos is None or self.animation_running:
            return

        if pos in (self.env.start, self.env.goal, self.player_pos):
            return

        r, c = pos

        if event.button == 1:
            current = self.env.grid[r][c]
            if current == 1:
                self.env.grid[r][c] = 0
                self.status_text = f"Obstáculo removido em {pos}."
            else:
                self.env.grid[r][c] = 1
                self.status_text = f"Obstáculo criado em {pos}."
            self.clear_search_results()
            self.draw()

        elif event.button == 3:
            current = self.env.grid[r][c]
            if current == 2:
                self.env.grid[r][c] = 0
                self.status_text = f"Terreno caro removido em {pos}."
            elif current == 0:
                self.env.grid[r][c] = 2
                self.status_text = f"Terreno caro criado em {pos}."
            self.clear_search_results()
            self.draw()

    def on_key_press(self, event):
        if event.key in ACTIONS:
            self.try_move(event.key)

    def on_reset_clicked(self, event):
        self.reset_state()

    def on_bfs_clicked(self, event):
        self.clear_search_results()
        self.bfs_result = bfs(self.env)
        self.active_result = self.bfs_result

        if self.bfs_result.path:
            self.status_text = (
                f"BFS encontrou caminho com custo {self.bfs_result.total_cost} "
                f"e {len(self.bfs_result.closed)} nós expandidos."
            )
            print("\n=== RESOLUÇÃO COM BFS ===")
            print(f"Caminho: {self.bfs_result.path}")
            print(f"Custo: {self.bfs_result.total_cost}")
            print(f"Nós expandidos: {len(self.bfs_result.closed)}")
        else:
            self.status_text = "BFS não encontrou caminho."
            print("\n=== RESOLUÇÃO COM BFS ===")
            print("Nenhum caminho encontrado.")

        self.draw()

    def on_astar_clicked(self, event):
        self.clear_search_results()
        self.astar_result = astar(self.env)
        self.active_result = self.astar_result

        if self.astar_result.path:
            self.status_text = (
                f"A* encontrou caminho com custo {self.astar_result.total_cost} "
                f"e {len(self.astar_result.closed)} nós expandidos."
            )
            print("\n=== RESOLUÇÃO COM A* ===")
            print(f"Caminho: {self.astar_result.path}")
            print(f"Custo: {self.astar_result.total_cost}")
            print(f"Nós expandidos: {len(self.astar_result.closed)}")
        else:
            self.status_text = "A* não encontrou caminho."
            print("\n=== RESOLUÇÃO COM A* ===")
            print("Nenhum caminho encontrado.")

        self.draw()

    def animate_search(self, result: SearchResult):
        self.animation_result = result
        self.animation_step = 0
        self.animation_running = True
        self.active_result = None

        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
            self._anim_ref = None

        total_frames = len(result.expansion_order)

        if total_frames == 0:
            self.animation_running = False
            self.active_result = result
            self.status_text = f"{result.name} não expandiu nós."
            self.draw()
            return

        def update(frame):
            self.animation_step = frame + 1
            self.status_text = (
                f"Animando {result.name}: expansão {self.animation_step}/"
                f"{total_frames}"
            )

            if self.animation_step >= total_frames:
                self.animation_running = False
                self.active_result = result
                self.status_text = (
                    f"Animação concluída: {result.name} expandiu {len(result.closed)} nós."
                )

            self.draw()
            return []

        self.anim = FuncAnimation(
            self.fig,
            update,
            frames=total_frames,
            interval=350,
            repeat=False,
            blit=False,
            cache_frame_data=False,
        )
        self._anim_ref = self.anim
        self.fig.canvas.draw_idle()

    def on_anim_clicked(self, event):
        if self.active_result is None:
            self.status_text = "Resolva primeiro com BFS ou A* para animar."
            self.draw()
            return

        self.animate_search(self.active_result)

    def on_map_clicked(self, event):
        self.map_index = (self.map_index + 1) % len(self.maps)
        self.reset_state()

    def on_cost_clicked(self, event):
        self.show_cost_labels = not self.show_cost_labels
        self.status_text = (
            "Exibição de g, h e f ativada."
            if self.show_cost_labels
            else "Exibição de g, h e f desativada."
        )
        self.draw()

    def run(self):
        print("=== AULA 06D - LABIRINTO JOGÁVEL COM BFS E A* ===")
        print("Controles:")
        print("- setas: mover jogador")
        print("- clique esquerdo: criar/remover obstáculo")
        print("- clique direito: criar/remover terreno caro")
        print("- botão 'Resolver BFS'")
        print("- botão 'Resolver A*'")
        print("- botão 'Animar'")
        print("- botão 'Mostrar custos'")
        print("\nMensagem-chave:")
        print("- BFS ignora custo de terreno e olha níveis da busca.")
        print("- A* combina custo real com heurística.")
        print("- Em terreno heterogêneo, menos passos nem sempre significa menor custo.")
        print("\nAbrindo janela...")

        self.draw()
        plt.show()


def main():
    app = MazeSearchApp()
    app.run()


if __name__ == "__main__":
    main()