# Aula 05C - Jogo do Cofre com Alavancas: DFS vs BFS
# Python 3.10+ | Requer: matplotlib
#
# Ideia:
# - há 4 alavancas (A, B, C, D)
# - cada ação altera a própria alavanca e também algumas outras
# - objetivo: chegar à combinação que abre o cofre
#
# Recursos:
# - clique nas alavancas para jogar manualmente
# - resolver com DFS
# - resolver com BFS
# - animar solução encontrada
# - painel com métricas e sequência de ações
#
# Objetivo didático:
# - mostrar espaço de estados sem usar grid
# - comparar DFS e BFS
# - reforçar estado, ação, objetivo e caminho solução

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button


State = Tuple[int, int, int, int]   # A, B, C, D
Action = int                        # índice da alavanca acionada


LEVER_NAMES = ["A", "B", "C", "D"]


@dataclass
class SearchResult:
    name: str
    path: Optional[List[State]]
    actions: Optional[List[Action]]
    visited_count: int
    visit_order: Dict[State, int]
    frontier_sizes: List[int]
    max_frontier_size: int
    execution_time_ms: float


class SafeLeverGame:
    def __init__(self):
        self.initial_state: State = (0, 0, 0, 0)
        self.goal_state: State = (1, 0, 1, 0)

        # Cada alavanca alterna ela mesma e outras.
        # Isso gera um espaço de estados pequeno, mas interessante.
        self.transitions: Dict[int, Tuple[int, ...]] = {
            0: (0, 1),       # A mexe em A e B
            1: (1, 2),       # B mexe em B e C
            2: (2, 3),       # C mexe em C e D
            3: (0, 3),       # D mexe em A e D
        }

    def is_goal(self, state: State) -> bool:
        return state == self.goal_state

    def apply_action(self, state: State, action: Action) -> State:
        values = list(state)
        for idx in self.transitions[action]:
            values[idx] = 1 - values[idx]
        return tuple(values)

    def valid_actions(self, state: State) -> List[Action]:
        # Todas as alavancas sempre podem ser acionadas
        return [0, 1, 2, 3]

    def successors(self, state: State) -> List[Tuple[State, Action]]:
        return [(self.apply_action(state, action), action) for action in self.valid_actions(state)]

    def format_state(self, state: State) -> str:
        return f"A={state[0]}  B={state[1]}  C={state[2]}  D={state[3]}"


def dfs(game: SafeLeverGame) -> SearchResult:
    start_time = time.perf_counter()

    stack = [(game.initial_state, [game.initial_state], [])]
    visited: Set[State] = set()
    visit_order: Dict[State, int] = {}
    frontier_sizes: List[int] = []
    visited_count = 0

    while stack:
        frontier_sizes.append(len(stack))
        current, path, actions = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        visited_count += 1
        visit_order[current] = visited_count

        if game.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                name="DFS",
                path=path,
                actions=actions,
                visited_count=visited_count,
                visit_order=visit_order,
                frontier_sizes=frontier_sizes,
                max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
                execution_time_ms=elapsed,
            )

        for next_state, action in reversed(game.successors(current)):
            if next_state not in visited:
                stack.append((next_state, path + [next_state], actions + [action]))

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        name="DFS",
        path=None,
        actions=None,
        visited_count=visited_count,
        visit_order=visit_order,
        frontier_sizes=frontier_sizes,
        max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
        execution_time_ms=elapsed,
    )


def bfs(game: SafeLeverGame) -> SearchResult:
    start_time = time.perf_counter()

    queue = deque([(game.initial_state, [game.initial_state], [])])
    visited = {game.initial_state}
    visit_order = {game.initial_state: 1}
    frontier_sizes: List[int] = []
    visited_count = 0
    order_counter = 1

    while queue:
        frontier_sizes.append(len(queue))
        current, path, actions = queue.popleft()
        visited_count += 1

        if game.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                name="BFS",
                path=path,
                actions=actions,
                visited_count=visited_count,
                visit_order=visit_order,
                frontier_sizes=frontier_sizes,
                max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
                execution_time_ms=elapsed,
            )

        for next_state, action in game.successors(current):
            if next_state not in visited:
                visited.add(next_state)
                order_counter += 1
                visit_order[next_state] = order_counter
                queue.append((next_state, path + [next_state], actions + [action]))

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        name="BFS",
        path=None,
        actions=None,
        visited_count=visited_count,
        visit_order=visit_order,
        frontier_sizes=frontier_sizes,
        max_frontier_size=max(frontier_sizes) if frontier_sizes else 0,
        execution_time_ms=elapsed,
    )


class SafeApp:
    def __init__(self):
        self.game = SafeLeverGame()

        self.current_state: State = self.game.initial_state
        self.manual_history: List[Action] = []
        self.safe_open = False

        self.dfs_result: Optional[SearchResult] = None
        self.bfs_result: Optional[SearchResult] = None
        self.active_result: Optional[SearchResult] = None

        self.animating = False
        self.anim_index = 0
        self.animation_path: Optional[List[State]] = None
        self.animation_actions: Optional[List[Action]] = None
        self.animation_source_name = ""

        self.status_text = "Clique nas alavancas para tentar abrir o cofre."

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.1, 1.2])

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.info_ax = self.fig.add_subplot(gs[0, 1])

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.reset_ax = self.fig.add_axes([0.08, 0.02, 0.10, 0.06])
        self.dfs_ax = self.fig.add_axes([0.20, 0.02, 0.10, 0.06])
        self.bfs_ax = self.fig.add_axes([0.32, 0.02, 0.10, 0.06])
        self.anim_ax = self.fig.add_axes([0.44, 0.02, 0.10, 0.06])
        self.goal_ax = self.fig.add_axes([0.56, 0.02, 0.12, 0.06])

        self.reset_button = Button(self.reset_ax, "Reiniciar")
        self.dfs_button = Button(self.dfs_ax, "Resolver DFS")
        self.bfs_button = Button(self.bfs_ax, "Resolver BFS")
        self.anim_button = Button(self.anim_ax, "Animar")
        self.goal_button = Button(self.goal_ax, "Ver objetivo")

        self.reset_button.on_clicked(self.on_reset_clicked)
        self.dfs_button.on_clicked(self.on_dfs_clicked)
        self.bfs_button.on_clicked(self.on_bfs_clicked)
        self.anim_button.on_clicked(self.on_anim_clicked)
        self.goal_button.on_clicked(self.on_goal_clicked)

        self.lever_boxes = {
            0: (0.9, 3.2, 1.1, 1.2),
            1: (2.4, 3.2, 1.1, 1.2),
            2: (3.9, 3.2, 1.1, 1.2),
            3: (5.4, 3.2, 1.1, 1.2),
        }

    def reset(self):
        self.current_state = self.game.initial_state
        self.manual_history = []
        self.safe_open = False
        self.dfs_result = None
        self.bfs_result = None
        self.active_result = None
        self.animating = False
        self.anim_index = 0
        self.animation_path = None
        self.animation_actions = None
        self.animation_source_name = ""
        self.status_text = "Clique nas alavancas para tentar abrir o cofre."
        self.draw()

    def apply_manual_action(self, action: Action):
        if self.animating:
            return

        self.current_state = self.game.apply_action(self.current_state, action)
        self.manual_history.append(action)

        if self.game.is_goal(self.current_state):
            self.safe_open = True
            self.status_text = "Cofre aberto! Você acertou a configuração."
        else:
            self.safe_open = False
            self.status_text = f"Você acionou a alavanca {LEVER_NAMES[action]}."

        self.draw()

    def draw_safe(self):
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.axis("off")

        # corpo do cofre
        body = Rectangle((0.7, 1.2), 6.3, 5.8, linewidth=2.5, edgecolor="black", facecolor="#d9d9d9")
        door = Rectangle((1.0, 1.5), 5.7, 5.2, linewidth=2.0, edgecolor="black", facecolor="#cfcfcf")
        self.ax.add_patch(body)
        self.ax.add_patch(door)

        # volante do cofre
        wheel = Circle((6.1, 4.1), 0.55, linewidth=2.0, edgecolor="black", facecolor="#bbbbbb")
        self.ax.add_patch(wheel)
        self.ax.plot([6.1, 6.1], [3.6, 4.6], linewidth=2)
        self.ax.plot([5.6, 6.6], [4.1, 4.1], linewidth=2)

        # indicador de aberto/fechado
        if self.safe_open:
            lock_color = "#2a9d8f"
            lock_text = "ABERTO"
        else:
            lock_color = "#e76f51"
            lock_text = "FECHADO"

        lock = Rectangle((5.2, 5.7), 1.3, 0.5, linewidth=1.8, edgecolor="black", facecolor=lock_color)
        self.ax.add_patch(lock)
        self.ax.text(5.85, 5.95, lock_text, ha="center", va="center", fontsize=10, fontweight="bold")

        # alavancas
        for idx, (x, y, w, h) in self.lever_boxes.items():
            state = self.current_state[idx]
            face = "#90be6d" if state == 1 else "#f94144"

            base = Rectangle((x, y), w, h, linewidth=2.0, edgecolor="black", facecolor="#eeeeee")
            self.ax.add_patch(base)

            self.ax.text(x + w / 2, y + h + 0.22, LEVER_NAMES[idx], ha="center", va="bottom", fontsize=12, fontweight="bold")
            self.ax.text(x + w / 2, y - 0.18, f"estado {state}", ha="center", va="top", fontsize=9)

            # haste da alavanca
            if state == 1:
                x2 = x + 0.78
                y2 = y + 0.95
            else:
                x2 = x + 0.32
                y2 = y + 0.95

            self.ax.plot([x + 0.55, x2], [y + 0.35, y2], linewidth=4)
            knob = Circle((x2, y2), 0.09, facecolor=face, edgecolor="black")
            self.ax.add_patch(knob)

        # título
        self.ax.text(
            3.8, 7.45,
            "Jogo do Cofre com Alavancas — DFS vs BFS",
            ha="center", va="center", fontsize=15, fontweight="bold"
        )

        self.ax.text(
            3.8, 6.95,
            self.status_text,
            ha="center", va="center", fontsize=11
        )

    def draw_info_panel(self):
        self.info_ax.clear()
        self.info_ax.axis("off")

        manual_seq = " → ".join(LEVER_NAMES[a] for a in self.manual_history) if self.manual_history else "-"
        dfs_steps = len(self.dfs_result.path) - 1 if self.dfs_result and self.dfs_result.path else None
        bfs_steps = len(self.bfs_result.path) - 1 if self.bfs_result and self.bfs_result.path else None

        dfs_visit = self.dfs_result.visited_count if self.dfs_result else None
        bfs_visit = self.bfs_result.visited_count if self.bfs_result else None

        dfs_front = self.dfs_result.max_frontier_size if self.dfs_result else None
        bfs_front = self.bfs_result.max_frontier_size if self.bfs_result else None

        dfs_time = self.dfs_result.execution_time_ms if self.dfs_result else None
        bfs_time = self.bfs_result.execution_time_ms if self.bfs_result else None

        active_text = self.active_result.name if self.active_result else "--"

        lines = [
            "PAINEL DIDÁTICO",
            "",
            f"Estado atual:",
            f"{self.game.format_state(self.current_state)}",
            "",
            f"Estado objetivo:",
            f"{self.game.format_state(self.game.goal_state)}",
            "",
            f"Histórico manual:",
            f"{manual_seq}",
            "",
            f"Algoritmo ativo: {active_text}",
            "",
            f"DFS passos: {dfs_steps if dfs_steps is not None else '--'}",
            f"DFS visitados: {dfs_visit if dfs_visit is not None else '--'}",
            f"DFS fronteira máx.: {dfs_front if dfs_front is not None else '--'}",
            f"DFS tempo (ms): {dfs_time:.3f}" if dfs_time is not None else "DFS tempo (ms): --",
            "",
            f"BFS passos: {bfs_steps if bfs_steps is not None else '--'}",
            f"BFS visitados: {bfs_visit if bfs_visit is not None else '--'}",
            f"BFS fronteira máx.: {bfs_front if bfs_front is not None else '--'}",
            f"BFS tempo (ms): {bfs_time:.3f}" if bfs_time is not None else "BFS tempo (ms): --",
            "",
            "REGRAS DAS ALAVANCAS",
            "• A altera A e B",
            "• B altera B e C",
            "• C altera C e D",
            "• D altera A e D",
            "",
            "LEITURA DIDÁTICA",
            "• estado = configuração das",
            "  4 alavancas",
            "• ação = acionar uma alavanca",
            "• BFS busca a sequência mínima",
            "• DFS pode achar solução",
            "  diferente e mais longa",
        ]

        if self.active_result and self.active_result.actions:
            seq = " → ".join(LEVER_NAMES[a] for a in self.active_result.actions)
            lines += [
                "",
                f"SEQUÊNCIA {self.active_result.name}",
                seq
            ]

        self.info_ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)

    def draw(self):
        self.draw_safe()
        self.draw_info_panel()
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.12, wspace=0.18)
        self.fig.canvas.draw_idle()

    def lever_from_event(self, event) -> Optional[Action]:
        if event.inaxes != self.ax:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        x = event.xdata
        y = event.ydata

        for idx, (bx, by, bw, bh) in self.lever_boxes.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return idx
        return None

    def on_click(self, event):
        action = self.lever_from_event(event)
        if action is not None:
            self.apply_manual_action(action)

    def on_reset_clicked(self, event):
        self.reset()

    def on_goal_clicked(self, event):
        self.status_text = f"Objetivo do cofre: {self.game.format_state(self.game.goal_state)}"
        self.draw()

    def on_dfs_clicked(self, event):
        self.dfs_result = dfs(self.game)
        self.active_result = self.dfs_result

        if self.dfs_result.path:
            self.status_text = (
                f"DFS encontrou solução com {len(self.dfs_result.path) - 1} ações "
                f"e visitou {self.dfs_result.visited_count} estados."
            )
            print("\n=== RESOLUÇÃO COM DFS ===")
            print(f"Estados visitados: {self.dfs_result.visited_count}")
            print(f"Passos: {len(self.dfs_result.path) - 1}")
            print(f"Fronteira máxima: {self.dfs_result.max_frontier_size}")
            print(f"Tempo: {self.dfs_result.execution_time_ms:.3f} ms")
            print("Sequência:", " -> ".join(LEVER_NAMES[a] for a in self.dfs_result.actions))
        else:
            self.status_text = "DFS não encontrou solução."
            print("\n=== RESOLUÇÃO COM DFS ===")
            print("Nenhuma solução encontrada.")

        self.draw()

    def on_bfs_clicked(self, event):
        self.bfs_result = bfs(self.game)
        self.active_result = self.bfs_result

        if self.bfs_result.path:
            self.status_text = (
                f"BFS encontrou solução com {len(self.bfs_result.path) - 1} ações "
                f"e visitou {self.bfs_result.visited_count} estados."
            )
            print("\n=== RESOLUÇÃO COM BFS ===")
            print(f"Estados visitados: {self.bfs_result.visited_count}")
            print(f"Passos: {len(self.bfs_result.path) - 1}")
            print(f"Fronteira máxima: {self.bfs_result.max_frontier_size}")
            print(f"Tempo: {self.bfs_result.execution_time_ms:.3f} ms")
            print("Sequência:", " -> ".join(LEVER_NAMES[a] for a in self.bfs_result.actions))
        else:
            self.status_text = "BFS não encontrou solução."
            print("\n=== RESOLUÇÃO COM BFS ===")
            print("Nenhuma solução encontrada.")

        self.draw()

    def on_anim_clicked(self, event):
        if self.active_result is None or self.active_result.path is None or self.active_result.actions is None:
            self.status_text = "Resolva primeiro com DFS ou BFS para animar."
            self.draw()
            return

        self.animation_path = self.active_result.path
        self.animation_actions = self.active_result.actions
        self.animation_source_name = self.active_result.name
        self.anim_index = 0
        self.animating = True
        self.current_state = self.animation_path[0]
        self.safe_open = self.game.is_goal(self.current_state)
        self.status_text = f"Iniciando animação da solução por {self.animation_source_name}."
        self.draw()

        timer = self.fig.canvas.new_timer(interval=900)

        def step():
            if not self.animating or self.animation_path is None:
                timer.stop()
                return

            self.anim_index += 1

            if self.anim_index < len(self.animation_path):
                self.current_state = self.animation_path[self.anim_index]
                self.safe_open = self.game.is_goal(self.current_state)

                if self.anim_index - 1 < len(self.animation_actions):
                    a = self.animation_actions[self.anim_index - 1]
                    self.status_text = (
                        f"{self.animation_source_name}: acionando alavanca {LEVER_NAMES[a]} "
                        f"(passo {self.anim_index}/{len(self.animation_path)-1})"
                    )
                self.draw()
            else:
                self.animating = False
                self.safe_open = self.game.is_goal(self.current_state)
                self.status_text = f"Animação concluída: solução por {self.animation_source_name}."
                self.draw()
                timer.stop()

        timer.add_callback(step)
        timer.start()

    def run(self):
        print("=== AULA 05C - JOGO DO COFRE COM ALAVANCAS ===")
        print("Controles:")
        print("- clique nas alavancas A, B, C, D")
        print("- botão 'Resolver DFS'")
        print("- botão 'Resolver BFS'")
        print("- botão 'Animar'")
        print("- botão 'Ver objetivo'")
        print("\nMensagem-chave:")
        print("- Cada configuração das alavancas é um estado.")
        print("- Cada clique é uma ação.")
        print("- BFS encontra a menor sequência de ações.")
        print("- DFS pode encontrar uma solução válida, mas não mínima.")
        print("\nAbrindo janela...")

        self.draw()
        plt.show()


def main():
    app = SafeApp()
    app.run()


if __name__ == "__main__":
    main()