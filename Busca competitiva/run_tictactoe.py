# Aula 06C - Jogo da Velha com Minimax (versão jogável, visual e didática)
# Python 3.10+ | Requer: matplotlib
#
# Recursos desta versão:
# - aluno joga clicando no tabuleiro
# - IA joga com Minimax + poda alfa-beta
# - linha vencedora destacada
# - botão de reiniciar
# - botão para alternar modo "Humano vs IA" e "IA vs IA"
# - painel lateral com explicação da jogada escolhida
# - coordenadas mostradas no tabuleiro para fins didáticos

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


Board = List[List[str]]
Move = Tuple[int, int]

HUMAN = "X"
AI = "O"
EMPTY = ""


@dataclass
class AIMoveReport:
    chosen_move: Optional[Move]
    chosen_value: int
    evaluated_moves: List[Tuple[Move, int]]
    nodes_visited: int


def create_board() -> Board:
    return [[EMPTY for _ in range(3)] for _ in range(3)]


def available_moves(board: Board) -> List[Move]:
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY]


def check_winner(board: Board) -> Tuple[Optional[str], Optional[List[Move]]]:
    winning_lines = []

    for r in range(3):
        winning_lines.append([(r, 0), (r, 1), (r, 2)])

    for c in range(3):
        winning_lines.append([(0, c), (1, c), (2, c)])

    winning_lines.append([(0, 0), (1, 1), (2, 2)])
    winning_lines.append([(0, 2), (1, 1), (2, 0)])

    for line in winning_lines:
        values = [board[r][c] for r, c in line]
        if values[0] != EMPTY and values[0] == values[1] == values[2]:
            return values[0], line

    return None, None


def is_draw(board: Board) -> bool:
    winner, _ = check_winner(board)
    return winner is None and all(board[r][c] != EMPTY for r in range(3) for c in range(3))


def terminal_state(board: Board) -> bool:
    winner, _ = check_winner(board)
    return winner is not None or is_draw(board)


def evaluate(board: Board) -> int:
    winner, _ = check_winner(board)
    if winner == AI:
        return 1
    if winner == HUMAN:
        return -1
    return 0


def alphabeta(
    board: Board,
    maximizing: bool,
    alpha: int,
    beta: int,
    counter: dict,
) -> int:
    counter["nodes"] += 1

    if terminal_state(board):
        return evaluate(board)

    if maximizing:
        best_value = -999
        for r, c in available_moves(board):
            board[r][c] = AI
            value = alphabeta(board, False, alpha, beta, counter)
            board[r][c] = EMPTY

            best_value = max(best_value, value)
            alpha = max(alpha, best_value)

            if beta <= alpha:
                break

        return best_value

    best_value = 999
    for r, c in available_moves(board):
        board[r][c] = HUMAN
        value = alphabeta(board, True, alpha, beta, counter)
        board[r][c] = EMPTY

        best_value = min(best_value, value)
        beta = min(beta, best_value)

        if beta <= alpha:
            break

    return best_value


def best_ai_move(board: Board, ai_symbol: str, opponent_symbol: str) -> AIMoveReport:
    moves = available_moves(board)
    if not moves:
        return AIMoveReport(None, 0, [], 0)

    evaluated_moves: List[Tuple[Move, int]] = []
    total_nodes = 0

    best_move: Optional[Move] = None

    if ai_symbol == AI:
        best_value = -999
        for r, c in moves:
            board[r][c] = ai_symbol
            counter = {"nodes": 0}
            value = alphabeta(board, False, -999, 999, counter)
            board[r][c] = EMPTY

            evaluated_moves.append(((r, c), value))
            total_nodes += counter["nodes"]

            if value > best_value:
                best_value = value
                best_move = (r, c)
    else:
        best_value = 999
        for r, c in moves:
            board[r][c] = ai_symbol
            counter = {"nodes": 0}
            value = alphabeta_for_symbol(board, True, -999, 999, counter, ai_symbol, opponent_symbol)
            board[r][c] = EMPTY

            evaluated_moves.append(((r, c), value))
            total_nodes += counter["nodes"]

            if value < best_value:
                best_value = value
                best_move = (r, c)

    return AIMoveReport(best_move, best_value, evaluated_moves, total_nodes)


def alphabeta_for_symbol(
    board: Board,
    maximizing_for_opponent: bool,
    alpha: int,
    beta: int,
    counter: dict,
    ai_symbol: str,
    opponent_symbol: str,
) -> int:
    counter["nodes"] += 1

    winner, _ = check_winner(board)
    if winner is not None:
        if winner == ai_symbol:
            return 1
        if winner == opponent_symbol:
            return -1
        return 0

    if is_draw(board):
        return 0

    if maximizing_for_opponent:
        best_value = 999
        for r, c in available_moves(board):
            board[r][c] = opponent_symbol
            value = alphabeta_for_symbol(board, False, alpha, beta, counter, ai_symbol, opponent_symbol)
            board[r][c] = EMPTY

            best_value = min(best_value, value)
            beta = min(beta, best_value)

            if beta <= alpha:
                break
        return best_value

    best_value = -999
    for r, c in available_moves(board):
        board[r][c] = ai_symbol
        value = alphabeta_for_symbol(board, True, alpha, beta, counter, ai_symbol, opponent_symbol)
        board[r][c] = EMPTY

        best_value = max(best_value, value)
        alpha = max(alpha, best_value)

        if beta <= alpha:
            break
    return best_value


class TicTacToeGame:
    def __init__(self):
        self.board = create_board()
        self.current_player = HUMAN
        self.game_over = False
        self.status_text = "Sua vez: jogue com X"
        self.last_report: Optional[AIMoveReport] = None
        self.winning_line: Optional[List[Move]] = None
        self.mode = "human_vs_ai"  # ou "ai_vs_ai"

        self.fig = plt.figure(figsize=(10, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1.5])

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.info_ax = self.fig.add_subplot(gs[0, 1])

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.reset_ax = self.fig.add_axes([0.18, 0.02, 0.16, 0.06])
        self.mode_ax = self.fig.add_axes([0.38, 0.02, 0.22, 0.06])
        self.next_ax = self.fig.add_axes([0.64, 0.02, 0.18, 0.06])

        self.reset_button = Button(self.reset_ax, "Reiniciar")
        self.mode_button = Button(self.mode_ax, "Alternar modo")
        self.next_button = Button(self.next_ax, "Próxima IA")

        self.reset_button.on_clicked(self.on_reset_clicked)
        self.mode_button.on_clicked(self.on_mode_clicked)
        self.next_button.on_clicked(self.on_next_clicked)

    def reset(self):
        self.board = create_board()
        self.current_player = HUMAN
        self.game_over = False
        self.status_text = "Sua vez: jogue com X"
        self.last_report = None
        self.winning_line = None

        if self.mode == "ai_vs_ai":
            self.current_player = HUMAN
            self.status_text = "Modo IA vs IA: clique em 'Próxima IA'"
        self.draw()

    def draw_grid(self):
        self.ax.clear()
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(3, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for i in range(1, 3):
            self.ax.plot([i, i], [0, 3], linewidth=2)
            self.ax.plot([0, 3], [i, i], linewidth=2)

    def draw_coordinates(self):
        for r in range(3):
            for c in range(3):
                self.ax.text(
                    c + 0.08,
                    r + 0.18,
                    f"({r},{c})",
                    fontsize=8,
                    alpha=0.7,
                    ha="left",
                    va="top"
                )

    def draw_marks(self):
        for r in range(3):
            for c in range(3):
                mark = self.board[r][c]
                if mark != EMPTY:
                    self.ax.text(
                        c + 0.5,
                        r + 0.58,
                        mark,
                        ha="center",
                        va="center",
                        fontsize=34,
                        fontweight="bold"
                    )

    def draw_winning_line(self):
        if not self.winning_line:
            return

        (r1, c1), _, (r3, c3) = self.winning_line
        x1, y1 = c1 + 0.5, r1 + 0.5
        x2, y2 = c3 + 0.5, r3 + 0.5

        self.ax.plot([x1, x2], [y1, y2], linewidth=5)

    def draw_status(self):
        mode_name = "Humano vs IA" if self.mode == "human_vs_ai" else "IA vs IA"
        self.ax.set_title(
            "Jogo da Velha com Minimax + poda alfa-beta\n"
            f"Modo: {mode_name}\n"
            f"{self.status_text}",
            fontsize=12,
            fontweight="bold"
        )

    def draw_info_panel(self):
        self.info_ax.clear()
        self.info_ax.axis("off")

        lines = [
            "PAINEL DIDÁTICO",
            "",
            f"Jogador atual: {self.current_player}",
            f"Modo: {'Humano vs IA' if self.mode == 'human_vs_ai' else 'IA vs IA'}",
            "",
        ]

        if self.last_report is None:
            lines += [
                "Ainda não há análise da IA.",
                "",
                "LEITURA DIDÁTICA",
                "• X tenta minimizar",
                "• O tenta maximizar",
                "• utilidade final:",
                "   +1 = vitória da IA",
                "    0 = empate",
                "   -1 = vitória humana",
            ]
        else:
            lines += [
                "ÚLTIMA ANÁLISE DA IA",
                f"Jogada escolhida: {self.last_report.chosen_move}",
                f"Valor escolhido: {self.last_report.chosen_value}",
                f"Nós visitados: {self.last_report.nodes_visited}",
                "",
                "JOGADAS AVALIADAS",
            ]
            for move, value in self.last_report.evaluated_moves:
                marker = " <- escolhida" if move == self.last_report.chosen_move else ""
                lines.append(f"{move}: {value}{marker}")

            lines += [
                "",
                "INTERPRETAÇÃO",
                "• a IA simula desdobramentos",
                "• poda alfa-beta evita buscas inúteis",
                "• a jogada escolhida otimiza",
                "  o resultado contra adversário racional",
            ]

        self.info_ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)

    def draw(self):
        self.draw_grid()
        self.draw_coordinates()
        self.draw_marks()
        self.draw_winning_line()
        self.draw_status()
        self.draw_info_panel()
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        self.fig.canvas.draw_idle()

    def finish_if_needed(self) -> bool:
        winner, line = check_winner(self.board)

        if winner is not None:
            self.winning_line = line
            if winner == HUMAN:
                self.status_text = "X venceu."
            else:
                self.status_text = "O venceu."
            self.game_over = True
            self.draw()
            print(f"\nResultado final: {winner} venceu.")
            return True

        if is_draw(self.board):
            self.status_text = "Empate."
            self.game_over = True
            self.draw()
            print("\nResultado final: empate.")
            return True

        return False

    def human_move(self, row: int, col: int):
        if self.board[row][col] != EMPTY or self.game_over:
            return

        self.board[row][col] = HUMAN
        self.last_report = None
        print(f"\nHumano jogou em {(row, col)}")

        self.status_text = "IA pensando..."
        self.draw()

        if self.finish_if_needed():
            return

        self.current_player = AI
        self.ai_move(AI, HUMAN)

    def ai_move(self, ai_symbol: str, opponent_symbol: str):
        report = self.compute_best_move_for_symbol(ai_symbol, opponent_symbol)
        self.last_report = report

        print("\n=== IA avaliando jogadas ===")
        for move, value in report.evaluated_moves:
            print(f"Jogada candidata {move} -> valor = {value}")
        print(f"Escolha final: {report.chosen_move} -> valor {report.chosen_value}")
        print(f"Nós visitados: {report.nodes_visited}")

        if report.chosen_move is not None:
            r, c = report.chosen_move
            self.board[r][c] = ai_symbol
            print(f"IA ({ai_symbol}) jogou em {(r, c)}")

        if self.finish_if_needed():
            return

        self.current_player = HUMAN if ai_symbol == AI else AI
        if self.mode == "human_vs_ai":
            self.status_text = "Sua vez: jogue com X"
        else:
            self.status_text = f"Modo IA vs IA: próxima jogada de {self.current_player}"
        self.draw()

    def compute_best_move_for_symbol(self, ai_symbol: str, opponent_symbol: str) -> AIMoveReport:
        moves = available_moves(self.board)
        if not moves:
            return AIMoveReport(None, 0, [], 0)

        evaluated_moves: List[Tuple[Move, int]] = []
        total_nodes = 0
        best_value = -999
        best_move = None

        for r, c in moves:
            self.board[r][c] = ai_symbol
            counter = {"nodes": 0}
            value = self.alphabeta_generic(
                maximizing=False,
                alpha=-999,
                beta=999,
                counter=counter,
                max_symbol=ai_symbol,
                min_symbol=opponent_symbol,
            )
            self.board[r][c] = EMPTY

            evaluated_moves.append(((r, c), value))
            total_nodes += counter["nodes"]

            if value > best_value:
                best_value = value
                best_move = (r, c)

        return AIMoveReport(best_move, best_value, evaluated_moves, total_nodes)

    def alphabeta_generic(
        self,
        maximizing: bool,
        alpha: int,
        beta: int,
        counter: dict,
        max_symbol: str,
        min_symbol: str,
    ) -> int:
        counter["nodes"] += 1

        winner, _ = check_winner(self.board)
        if winner is not None:
            if winner == max_symbol:
                return 1
            if winner == min_symbol:
                return -1

        if is_draw(self.board):
            return 0

        if maximizing:
            best_value = -999
            for r, c in available_moves(self.board):
                self.board[r][c] = max_symbol
                value = self.alphabeta_generic(False, alpha, beta, counter, max_symbol, min_symbol)
                self.board[r][c] = EMPTY

                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            return best_value

        best_value = 999
        for r, c in available_moves(self.board):
            self.board[r][c] = min_symbol
            value = self.alphabeta_generic(True, alpha, beta, counter, max_symbol, min_symbol)
            self.board[r][c] = EMPTY

            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if beta <= alpha:
                break
        return best_value

    def on_click(self, event):
        if self.mode != "human_vs_ai":
            return

        if self.game_over:
            return

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if self.current_player != HUMAN:
            return

        col = int(event.xdata)
        row = int(event.ydata)

        if 0 <= row < 3 and 0 <= col < 3:
            self.human_move(row, col)

    def on_reset_clicked(self, event):
        self.reset()

    def on_mode_clicked(self, event):
        self.mode = "ai_vs_ai" if self.mode == "human_vs_ai" else "human_vs_ai"
        self.reset()

    def on_next_clicked(self, event):
        if self.mode != "ai_vs_ai" or self.game_over:
            return

        if self.current_player == HUMAN:
            self.status_text = "IA X pensando..."
            self.draw()
            self.ai_move(HUMAN, AI)
        else:
            self.status_text = "IA O pensando..."
            self.draw()
            self.ai_move(AI, HUMAN)

    def run(self):
        print("=== AULA 06C - JOGO DA VELHA COM MINIMAX ===")
        print("Modo padrão: Humano vs IA")
        print("Você joga com X clicando no tabuleiro.")
        print("A IA joga com O usando Minimax com poda alfa-beta.")
        print("\nRecursos:")
        print("- linha vencedora destacada")
        print("- botão de reiniciar")
        print("- modo IA vs IA")
        print("- painel didático lateral")
        print("- coordenadas visíveis no tabuleiro")
        print("\nInterpretação dos valores:")
        print("  +1 -> vitória do agente avaliado")
        print("   0 -> empate")
        print("  -1 -> derrota do agente avaliado")

        self.draw()
        plt.show()


def main():
    game = TicTacToeGame()
    game.run()


if __name__ == "__main__":
    main()