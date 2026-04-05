# Aula 06B - TSP MUITO mais visual
# Vizinho Mais Próximo com animação passo a passo
# Python 3.10+ | Requer: matplotlib, numpy

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

City = Tuple[float, float]


def dist(a: City, b: City) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_cost(tour: List[int], cities: List[City]) -> float:
    if len(tour) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += dist(cities[tour[i]], cities[tour[i + 1]])
    return cost


def nearest_neighbor_steps(cities: List[City]) -> List[dict]:
    n = len(cities)
    visited = [0]
    current = 0
    steps = []

    while len(visited) < n:
        candidates = []
        for i in range(n):
            if i not in visited:
                d = dist(cities[current], cities[i])
                candidates.append((i, d))

        candidates.sort(key=lambda x: x[1])
        chosen = candidates[0][0]

        steps.append({
            "current": current,
            "visited_before": visited.copy(),
            "candidates": candidates.copy(),
            "chosen": chosen
        })

        visited.append(chosen)
        current = chosen

    steps.append({
        "current": current,
        "visited_before": visited.copy(),
        "candidates": [(0, dist(cities[current], cities[0]))],
        "chosen": 0
    })

    return steps


def build_final_tour_from_steps(steps: List[dict]) -> List[int]:
    route = [0]
    for step in steps[:-1]:
        route.append(step["chosen"])
    return route


def draw_base(ax, cities: List[City]):
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]

    ax.clear()
    ax.scatter(xs, ys, s=220, edgecolors="black", linewidths=1.2, zorder=3)
    ax.set_title("TSP — Heurística do Vizinho Mais Próximo", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    for i, (x, y) in enumerate(cities):
        label = f"{i}"
        if i == 0:
            label += "\nStart"
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)


def draw_step(ax, info_ax, cities: List[City], steps: List[dict], frame: int):
    draw_base(ax, cities)
    info_ax.clear()
    info_ax.axis("off")

    route_so_far = [0]
    for k in range(min(frame, len(steps) - 1)):
        route_so_far.append(steps[k]["chosen"])

    # desenhar rota já consolidada
    for i in range(len(route_so_far) - 1):
        a = cities[route_so_far[i]]
        b = cities[route_so_far[i + 1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=3, zorder=2)

        mx = (a[0] + b[0]) / 2
        my = (a[1] + b[1]) / 2
        ax.text(
            mx, my + 0.2, str(i + 1),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.8),
            zorder=5
        )

    if frame < len(steps):
        step = steps[frame]
        current = step["current"]
        chosen = step["chosen"]
        candidates = step["candidates"]
        visited_before = step["visited_before"]

        current_city = cities[current]
        chosen_city = cities[chosen]

        # cidade atual destacada
        ax.scatter(
            [current_city[0]], [current_city[1]],
            s=420, marker="o", linewidths=2.5, zorder=6
        )

        # candidatos em linhas tracejadas
        for cand, d in candidates:
            cx, cy = cities[cand]
            ax.plot(
                [current_city[0], cx],
                [current_city[1], cy],
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
                zorder=1
            )

            mx = (current_city[0] + cx) / 2
            my = (current_city[1] + cy) / 2
            ax.text(
                mx, my - 0.25, f"{d:.2f}",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", alpha=0.75),
                zorder=5
            )

        # escolha da heurística
        ax.plot(
            [current_city[0], chosen_city[0]],
            [current_city[1], chosen_city[1]],
            linewidth=4.5,
            zorder=7
        )

        ax.scatter(
            [chosen_city[0]], [chosen_city[1]],
            s=300, marker="*", linewidths=1.8, zorder=8
        )

        partial_route = visited_before.copy()
        if chosen != 0 or len(visited_before) == len(cities):
            partial_route = visited_before + [chosen]

        partial_cost = tour_cost(partial_route, cities)

        info_lines = [
            "PASSO ATUAL",
            "",
            f"Cidade atual: {current}",
            f"Próxima escolhida: {chosen}",
            "",
            "CANDIDATAS",
        ]

        for cand, d in candidates:
            marker = " <- escolhida" if cand == chosen else ""
            info_lines.append(f"cidade {cand}: {d:.2f}{marker}")

        info_lines += [
            "",
            f"Rota parcial: {partial_route}",
            f"Custo acumulado: {partial_cost:.2f}",
            "",
            "LEITURA DIDÁTICA",
            "• a heurística olha apenas",
            "  a melhor opção local;",
            "• ela não avalia o circuito",
            "  completo neste momento."
        ]

        info_ax.text(0.0, 1.0, "\n".join(info_lines), va="top", fontsize=11)

    # quadro final: fechar ciclo
    if frame == len(steps):
        final_route = build_final_tour_from_steps(steps)
        closed_route = final_route + [0]

        draw_base(ax, cities)

        for i in range(len(closed_route) - 1):
            a = cities[closed_route[i]]
            b = cities[closed_route[i + 1]]
            ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=3.5, zorder=2)

            mx = (a[0] + b[0]) / 2
            my = (a[1] + b[1]) / 2
            ax.text(
                mx, my + 0.2, str(i + 1),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.8),
                zorder=5
            )

        total_cost = 0.0
        for i in range(len(closed_route) - 1):
            total_cost += dist(cities[closed_route[i]], cities[closed_route[i + 1]])

        info_lines = [
            "ROTA FINAL DA HEURÍSTICA",
            "",
            f"Ordem: {closed_route}",
            f"Custo total: {total_cost:.2f}",
            "",
            "MENSAGEM-CHAVE",
            "• a rota foi construída",
            "  passo a passo;",
            "• cada escolha foi local;",
            "• isso pode ser rápido,",
            "  mas não garante ótimo."
        ]
        info_ax.text(0.0, 1.0, "\n".join(info_lines), va="top", fontsize=11)


def animate_tsp_nearest_neighbor(cities: List[City]):
    steps = nearest_neighbor_steps(cities)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])

    def update(frame):
        draw_step(ax, info_ax, cities, steps, frame)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(steps) + 1,
        interval=1800,
        repeat=True
    )

    plt.tight_layout()
    plt.show()

    return anim


def main():
    cities: List[City] = [
        (0, 0),
        (2, 6),
        (5, 3),
        (6, 8),
        (8, 2),
        (9, 6),
    ]

    print("=== AULA 06B - TSP (ANIMAÇÃO VISUAL) ===")
    print("A animação mostra a heurística escolhendo a próxima cidade.")
    print("Linhas tracejadas = opções candidatas.")
    print("Linha destacada = escolha do passo atual.")

    animate_tsp_nearest_neighbor(cities)

    print("\nMensagem-chave:")
    print("- O vizinho mais próximo toma decisões locais.")
    print("- Isso torna a busca rápida e intuitiva.")
    print("- Mas a melhor escolha imediata pode não levar ao melhor circuito global.")


if __name__ == "__main__":
    main()