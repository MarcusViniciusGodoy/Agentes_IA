# Aula 04 - Sistema Fuzzy Mamdani
# Mini-projeto: Freio assistido com animação no matplotlib
# Python 3.10+ | Requer: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation


# =========================================================
# Funções de pertinência
# =========================================================

def triangular(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    left = (a < x) & (x <= b)
    right = (b < x) & (x < c)

    if b != a:
        y[left] = (x[left] - a) / (b - a)
    if c != b:
        y[right] = (c - x[right]) / (c - b)

    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapezoidal(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    rise = (a < x) & (x < b)
    top = (b <= x) & (x <= c)
    fall = (c < x) & (x < d)

    if b != a:
        y[rise] = (x[rise] - a) / (b - a)
    y[top] = 1.0
    if d != c:
        y[fall] = (d - x[fall]) / (d - c)

    return np.clip(y, 0.0, 1.0)

# =========================================================
# Defuzzificação
# =========================================================

def defuzz_centroid(x, mu):
    denominator = np.sum(mu)
    if denominator == 0:
        return 0.0
    return np.sum(x * mu) / denominator


# =========================================================
# Sistema fuzzy de freio assistido
# =========================================================

def fuzzy_brake(distance_value, speed_value):
    # Universos de discurso
    x_dist = np.linspace(0, 100, 500)
    x_speed = np.linspace(0, 120, 500)
    x_brake = np.linspace(0, 100, 600)

    # -------------------------
    # Conjuntos fuzzy - Distância
    # -------------------------
    dist_perto = trapezoidal(x_dist, 0, 0, 12, 30)
    dist_media = triangular(x_dist, 20, 45, 70)
    dist_longe = trapezoidal(x_dist, 55, 75, 100, 100)

    # -------------------------
    # Conjuntos fuzzy - Velocidade
    # -------------------------
    vel_baixa = trapezoidal(x_speed, 0, 0, 25, 45)
    vel_media = triangular(x_speed, 30, 60, 90)
    vel_alta = trapezoidal(x_speed, 75, 95, 120, 120)

    # -------------------------
    # Conjuntos fuzzy - Saída (Frenagem)
    # -------------------------
    brake_fraca = trapezoidal(x_brake, 0, 0, 15, 30)
    brake_moderada = triangular(x_brake, 20, 40, 60)
    brake_forte = triangular(x_brake, 50, 70, 85)
    brake_emergencia = trapezoidal(x_brake, 80, 90, 100, 100)

    # -------------------------
    # Fuzzificação
    # -------------------------
    mu_dist_perto = trapezoidal([distance_value], 0, 0, 12, 30)[0]
    mu_dist_media = triangular([distance_value], 20, 45, 70)[0]
    mu_dist_longe = trapezoidal([distance_value], 55, 75, 100, 100)[0]

    mu_vel_baixa = trapezoidal([speed_value], 0, 0, 25, 45)[0]
    mu_vel_media = triangular([speed_value], 30, 60, 90)[0]
    mu_vel_alta = trapezoidal([speed_value], 75, 95, 120, 120)[0]

    # -------------------------
    # Regras Mamdani
    # -------------------------
    rules = [
        ("R1", min(mu_dist_longe, mu_vel_baixa), brake_fraca,
         "SE distância longe E velocidade baixa -> frenagem fraca"),

        ("R2", min(mu_dist_longe, mu_vel_media), brake_fraca,
         "SE distância longe E velocidade média -> frenagem fraca"),

        ("R3", min(mu_dist_longe, mu_vel_alta), brake_moderada,
         "SE distância longe E velocidade alta -> frenagem moderada"),

        ("R4", min(mu_dist_media, mu_vel_baixa), brake_fraca,
         "SE distância média E velocidade baixa -> frenagem fraca"),

        ("R5", min(mu_dist_media, mu_vel_media), brake_moderada,
         "SE distância média E velocidade média -> frenagem moderada"),

        ("R6", min(mu_dist_media, mu_vel_alta), brake_forte,
         "SE distância média E velocidade alta -> frenagem forte"),

        ("R7", min(mu_dist_perto, mu_vel_baixa), brake_moderada,
         "SE distância perto E velocidade baixa -> frenagem moderada"),

        ("R8", min(mu_dist_perto, mu_vel_media), brake_forte,
         "SE distância perto E velocidade média -> frenagem forte"),

        ("R9", min(mu_dist_perto, mu_vel_alta), brake_emergencia,
         "SE distância perto E velocidade alta -> frenagem de emergência"),
    ]

    clipped_outputs = [np.minimum(consequent, activation) for _, activation, consequent, _ in rules]
    aggregated = np.maximum.reduce(clipped_outputs)
    brake_value = defuzz_centroid(x_brake, aggregated)

    # Regra mais ativada
    strongest_rule = max(rules, key=lambda item: item[1])

    return {
        "x_dist": x_dist,
        "x_speed": x_speed,
        "x_brake": x_brake,
        "dist_sets": (dist_perto, dist_media, dist_longe),
        "speed_sets": (vel_baixa, vel_media, vel_alta),
        "brake_sets": (brake_fraca, brake_moderada, brake_forte, brake_emergencia),
        "mu_inputs": {
            "dist_perto": mu_dist_perto,
            "dist_media": mu_dist_media,
            "dist_longe": mu_dist_longe,
            "vel_baixa": mu_vel_baixa,
            "vel_media": mu_vel_media,
            "vel_alta": mu_vel_alta,
        },
        "rules": rules,
        "aggregated": aggregated,
        "brake_value": brake_value,
        "strongest_rule_name": strongest_rule[0],
        "strongest_rule_activation": strongest_rule[1],
        "strongest_rule_text": strongest_rule[3],
    }


# =========================================================
# Entrada do usuário
# =========================================================

def read_float(prompt, default):
    raw = input(f"{prompt} [padrão: {default}]: ").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        print(f"Valor inválido. Usando {default}.")
        return float(default)


def read_user_parameters():
    print("=== SIMULAÇÃO DE FREIO ASSISTIDO FUZZY ===\n")
    print("Pressione Enter para usar os valores padrão.\n")

    initial_distance = read_float("Distância inicial ao obstáculo (m)", 60)
    initial_speed = read_float("Velocidade inicial do carro (km/h)", 90)
    track_length = read_float("Comprimento da pista (unidades gráficas)", 120)
    frame_interval = read_float("Intervalo entre frames (ms)", 120)
    max_frames = int(read_float("Número máximo de frames", 220))

    if initial_distance < 0:
        initial_distance = 60.0
    if initial_speed < 0:
        initial_speed = 90.0
    if track_length < 80:
        track_length = 120.0
    if frame_interval < 10:
        frame_interval = 120.0
    if max_frames < 20:
        max_frames = 220

    return {
        "initial_distance": initial_distance,
        "initial_speed": initial_speed,
        "track_length": track_length,
        "frame_interval": frame_interval,
        "max_frames": max_frames,
    }


# =========================================================
# Relatório inicial
# =========================================================

def print_initial_report(initial_distance, initial_speed):
    res = fuzzy_brake(initial_distance, initial_speed)

    print("\n=== ANÁLISE INICIAL DO SISTEMA FUZZY ===\n")
    print(f"Distância inicial : {initial_distance:.2f} m")
    print(f"Velocidade inicial: {initial_speed:.2f} km/h\n")

    print("[Fuzzificação - Distância]")
    print(f"μ(perto) = {res['mu_inputs']['dist_perto']:.3f}")
    print(f"μ(média) = {res['mu_inputs']['dist_media']:.3f}")
    print(f"μ(longe) = {res['mu_inputs']['dist_longe']:.3f}\n")

    print("[Fuzzificação - Velocidade]")
    print(f"μ(baixa) = {res['mu_inputs']['vel_baixa']:.3f}")
    print(f"μ(média) = {res['mu_inputs']['vel_media']:.3f}")
    print(f"μ(alta)  = {res['mu_inputs']['vel_alta']:.3f}\n")

    print("[Ativação das regras]")
    for name, activation, _, description in res["rules"]:
        print(f"{name}: {activation:.3f} -> {description}")

    print("\n[Defuzzificação]")
    print(f"Frenagem inicial recomendada = {res['brake_value']:.2f} %")
    print(f"Regra dominante: {res['strongest_rule_name']} "
          f"(ativação = {res['strongest_rule_activation']:.3f})")
    print(f"Descrição: {res['strongest_rule_text']}\n")


# =========================================================
# Animação
# =========================================================

def animate_fuzzy_brake(initial_distance, initial_speed, track_length=120, frame_interval=120, max_frames=220):
    car_x0 = 8.0
    car_width = 10.0
    obstacle_x = car_x0 + car_width + initial_distance

    # garante que o obstáculo caiba na pista
    margin = 8.0
    if obstacle_x > track_length - margin:
        track_length = obstacle_x + margin

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.28, wspace=0.22)

    ax_scene = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_out = fig.add_subplot(gs[1, 1])

    fig.suptitle("Sistema Fuzzy Mamdani - Freio Assistido com Animação", fontsize=15)

    # -----------------------------------------------------
    # Subplot 1: Cena
    # -----------------------------------------------------
    ax_scene.set_xlim(0, track_length)
    ax_scene.set_ylim(0, 26)
    ax_scene.axis("off")

    # pista
    road = Rectangle((0, 6), track_length, 8, fill=False, linewidth=2)
    ax_scene.add_patch(road)
    ax_scene.plot([0, track_length], [10, 10], linestyle="--", linewidth=1.2)

    # carro
    car_body = Rectangle((car_x0, 7.2), car_width, 3.0, fill=False, linewidth=2)
    wheel1 = Circle((car_x0 + 2, 6.8), 0.6, fill=False, linewidth=2)
    wheel2 = Circle((car_x0 + 8, 6.8), 0.6, fill=False, linewidth=2)
    ax_scene.add_patch(car_body)
    ax_scene.add_patch(wheel1)
    ax_scene.add_patch(wheel2)
    ax_scene.text(car_x0 + 5, 11.2, "CARRO", ha="center", fontsize=10)

    # obstáculo
    obstacle = Rectangle((obstacle_x, 7.0), 4, 4.0, fill=False, linewidth=2)
    ax_scene.add_patch(obstacle)
    ax_scene.text(obstacle_x + 2, 11.2, "OBST.", ha="center", fontsize=10)

    # seta distância
    distance_arrow = ax_scene.annotate(
        "",
        xy=(obstacle_x, 15),
        xytext=(car_x0 + car_width, 15),
        arrowprops=dict(arrowstyle="<->", linewidth=1.5)
    )

    # textos
    txt_speed = ax_scene.text(2, 23.2, "", fontsize=11)
    txt_dist = ax_scene.text(2, 21.0, "", fontsize=11)
    txt_brake = ax_scene.text(2, 18.8, "", fontsize=11)
    txt_status = ax_scene.text(2, 16.6, "", fontsize=11)
    txt_rule = ax_scene.text(2, 3.3, "", fontsize=10)

    # barra de frenagem
    bar_x, bar_y, bar_w, bar_h = track_length * 0.58, 20.4, track_length * 0.22, 2.3
    brake_bar_border = Rectangle((bar_x, bar_y), bar_w, bar_h, fill=False, linewidth=1.5)
    brake_bar_fill = Rectangle((bar_x, bar_y), 0.0, bar_h, fill=True, alpha=0.5)
    ax_scene.add_patch(brake_bar_border)
    ax_scene.add_patch(brake_bar_fill)
    ax_scene.text(bar_x + bar_w / 2, bar_y + 3.0, "Intensidade de frenagem", ha="center", fontsize=10)

    # -----------------------------------------------------
    # Subplot 2: Entrada fuzzy - distância
    # -----------------------------------------------------
    x_dist = np.linspace(0, 100, 500)
    dist_perto = trapezoidal(x_dist, 0, 0, 12, 30)
    dist_media = triangular(x_dist, 20, 45, 70)
    dist_longe = trapezoidal(x_dist, 55, 75, 100, 100)

    ax_dist.plot(x_dist, dist_perto, label="Perto", linewidth=2)
    ax_dist.plot(x_dist, dist_media, label="Média", linewidth=2)
    ax_dist.plot(x_dist, dist_longe, label="Longe", linewidth=2)
    dist_line = ax_dist.axvline(initial_distance, linestyle="--", linewidth=2, label="Distância atual")
    ax_dist.set_title("Entrada Fuzzy - Distância")
    ax_dist.set_xlabel("Distância ao obstáculo (m)")
    ax_dist.set_ylabel("μ(x)")
    ax_dist.set_ylim(-0.05, 1.05)
    ax_dist.grid(True)
    ax_dist.legend()

    # -----------------------------------------------------
    # Subplot 3: Saída fuzzy
    # -----------------------------------------------------
    x_brake = np.linspace(0, 100, 600)
    brake_fraca = trapezoidal(x_brake, 0, 0, 15, 30)
    brake_moderada = triangular(x_brake, 20, 40, 60)
    brake_forte = triangular(x_brake, 50, 70, 85)
    brake_emergencia = trapezoidal(x_brake, 80, 90, 100, 100)

    ax_out.plot(x_brake, brake_fraca, "--", linewidth=1.5, label="Fraca")
    ax_out.plot(x_brake, brake_moderada, "--", linewidth=1.5, label="Moderada")
    ax_out.plot(x_brake, brake_forte, "--", linewidth=1.5, label="Forte")
    ax_out.plot(x_brake, brake_emergencia, "--", linewidth=1.5, label="Emergência")

    initial_res = fuzzy_brake(initial_distance, initial_speed)
    aggregated_fill = ax_out.fill_between(
        initial_res["x_brake"], 0, initial_res["aggregated"], alpha=0.55, label="Agregação"
    )
    centroid_line = ax_out.axvline(initial_res["brake_value"], linewidth=2, label="Centróide")

    ax_out.set_title("Saída Fuzzy - Frenagem")
    ax_out.set_xlabel("Intensidade de frenagem (%)")
    ax_out.set_ylabel("μ(x)")
    ax_out.set_ylim(-0.05, 1.05)
    ax_out.grid(True)
    ax_out.legend()

    # -----------------------------------------------------
    # Estado do sistema
    # -----------------------------------------------------
    state = {
        "car_x": car_x0,
        "speed_kmh": initial_speed,
        "stopped": False,
        "collided": False,
    }

    def update(frame):
        nonlocal aggregated_fill, distance_arrow

        if state["stopped"] or state["collided"]:
            return (
                car_body, wheel1, wheel2, brake_bar_fill,
                txt_speed, txt_dist, txt_brake, txt_status, txt_rule,
                dist_line, centroid_line
            )

        car_front = state["car_x"] + car_width
        distance = max(obstacle_x - car_front, 0.0)

        fuzzy = fuzzy_brake(distance, state["speed_kmh"])
        brake = fuzzy["brake_value"]

        # desaceleração didática
        deceleration = 0.030 * brake
        state["speed_kmh"] = max(state["speed_kmh"] - deceleration, 0.0)

        # conversão para deslocamento visual
        speed_plot = state["speed_kmh"] / 90.0
        state["car_x"] += speed_plot

        # colisão / parada
        if state["car_x"] + car_width >= obstacle_x:
            state["car_x"] = obstacle_x - car_width
            if state["speed_kmh"] > 0.5:
                state["collided"] = True
            else:
                state["stopped"] = True

        if state["speed_kmh"] <= 0.5 and distance <= 0.5:
            state["stopped"] = True

        # atualizar carro
        car_body.set_x(state["car_x"])
        wheel1.center = (state["car_x"] + 2, 6.8)
        wheel2.center = (state["car_x"] + 8, 6.8)

        # atualizar seta
        distance_arrow.remove()
        distance_arrow = ax_scene.annotate(
            "",
            xy=(obstacle_x, 15),
            xytext=(state["car_x"] + car_width, 15),
            arrowprops=dict(arrowstyle="<->", linewidth=1.5)
        )

        # atualizar barra
        brake_bar_fill.set_width((brake / 100.0) * bar_w)

        # atualizar textos
        txt_speed.set_text(f"Velocidade: {state['speed_kmh']:.1f} km/h")
        txt_dist.set_text(f"Distância ao obstáculo: {distance:.1f} m")
        txt_brake.set_text(f"Frenagem fuzzy: {brake:.1f} %")

        if state["collided"]:
            txt_status.set_text("Status: colisão")
        elif state["stopped"]:
            txt_status.set_text("Status: veículo parou")
        elif distance < 12 and state["speed_kmh"] > 50:
            txt_status.set_text("Status: risco elevado")
        elif distance < 25 and state["speed_kmh"] > 20:
            txt_status.set_text("Status: frenagem ativa")
        else:
            txt_status.set_text("Status: controle ativo")

        txt_rule.set_text(
            f"Regra dominante: {fuzzy['strongest_rule_name']} | "
            f"ativação = {fuzzy['strongest_rule_activation']:.3f} | "
            f"{fuzzy['strongest_rule_text']}"
        )

        # atualizar linha de distância
        dist_line.set_xdata([distance, distance])

        # atualizar saída fuzzy
        aggregated_fill.remove()
        aggregated_fill = ax_out.fill_between(
            fuzzy["x_brake"], 0, fuzzy["aggregated"], alpha=0.55
        )
        centroid_line.set_xdata([brake, brake])

        return (
            car_body, wheel1, wheel2, brake_bar_fill,
            txt_speed, txt_dist, txt_brake, txt_status, txt_rule,
            dist_line, centroid_line
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=frame_interval,
        repeat=False,
        blit=False
    )

    plt.show()
    return anim


# =========================================================
# Programa principal
# =========================================================

def main():
    params = read_user_parameters()

    print_initial_report(
        params["initial_distance"],
        params["initial_speed"]
    )

    print("Iniciando animação...\n")

    animate_fuzzy_brake(
        initial_distance=params["initial_distance"],
        initial_speed=params["initial_speed"],
        track_length=params["track_length"],
        frame_interval=params["frame_interval"],
        max_frames=params["max_frames"],
    )

    print("Mensagem-chave da aula:")
    print("- Em lógica fuzzy, múltiplas regras podem atuar simultaneamente.")
    print("- A decisão final depende da agregação dos consequentes.")
    print("- Defuzzificação transforma uma decisão difusa em uma ação numérica.")
    print("- Distância e velocidade interagem ao mesmo tempo no freio assistido.")


if __name__ == "__main__":
    main()