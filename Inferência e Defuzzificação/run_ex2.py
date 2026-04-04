# Aula 04 - Sistema Fuzzy Mamdani
# Mini-projeto: Controle de potência de um drone com interferência manual
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
# Sistema fuzzy do drone
# Entradas:
#   - airspeed_value   -> velocidade do ar medida pelo Pitot
#   - descent_value    -> taxa de descida (m/s, positiva = descendo)
# Saída:
#   - thrust_value     -> potência do motor (%)
# =========================================================

def fuzzy_drone(airspeed_value, descent_value):
    # Universos de discurso
    x_air = np.linspace(0, 30, 500)       # m/s
    x_desc = np.linspace(0, 8, 500)       # m/s
    x_thrust = np.linspace(0, 100, 600)   # %

    # -------------------------
    # Conjuntos fuzzy - Velocidade do ar
    # -------------------------
    air_low = trapezoidal(x_air, 0, 0, 8, 13)
    air_ok = triangular(x_air, 10, 16, 22)
    air_high = trapezoidal(x_air, 19, 24, 30, 30)

    # -------------------------
    # Conjuntos fuzzy - Taxa de descida
    # -------------------------
    desc_small = trapezoidal(x_desc, 0, 0, 1.2, 2.5)
    desc_medium = triangular(x_desc, 1.8, 3.5, 5.2)
    desc_critical = trapezoidal(x_desc, 4.5, 6.0, 8.0, 8.0)

    # -------------------------
    # Conjuntos fuzzy - Saída (Potência)
    # -------------------------
    thrust_low = trapezoidal(x_thrust, 0, 0, 18, 32)
    thrust_medium = triangular(x_thrust, 25, 45, 65)
    thrust_high = triangular(x_thrust, 55, 72, 88)
    thrust_max = trapezoidal(x_thrust, 82, 92, 100, 100)

    # -------------------------
    # Fuzzificação
    # -------------------------
    mu_air_low = trapezoidal([airspeed_value], 0, 0, 8, 13)[0]
    mu_air_ok = triangular([airspeed_value], 10, 16, 22)[0]
    mu_air_high = trapezoidal([airspeed_value], 19, 24, 30, 30)[0]

    mu_desc_small = trapezoidal([descent_value], 0, 0, 1.2, 2.5)[0]
    mu_desc_medium = triangular([descent_value], 1.8, 3.5, 5.2)[0]
    mu_desc_critical = trapezoidal([descent_value], 4.5, 6.0, 8.0, 8.0)[0]

    # -------------------------
    # Regras Mamdani
    # -------------------------
    rules = [
        ("R1", min(mu_air_low, mu_desc_small), thrust_high,
         "SE velocidade do ar baixa E descida pequena -> potência alta"),

        ("R2", min(mu_air_low, mu_desc_medium), thrust_high,
         "SE velocidade do ar baixa E descida moderada -> potência alta"),

        ("R3", min(mu_air_low, mu_desc_critical), thrust_max,
         "SE velocidade do ar baixa E descida crítica -> potência máxima"),

        ("R4", min(mu_air_ok, mu_desc_small), thrust_medium,
         "SE velocidade do ar ideal E descida pequena -> potência média"),

        ("R5", min(mu_air_ok, mu_desc_medium), thrust_medium,
         "SE velocidade do ar ideal E descida moderada -> potência média"),

        ("R6", min(mu_air_ok, mu_desc_critical), thrust_high,
         "SE velocidade do ar ideal E descida crítica -> potência alta"),

        ("R7", min(mu_air_high, mu_desc_small), thrust_low,
         "SE velocidade do ar alta E descida pequena -> potência baixa"),

        ("R8", min(mu_air_high, mu_desc_medium), thrust_medium,
         "SE velocidade do ar alta E descida moderada -> potência média"),

        ("R9", min(mu_air_high, mu_desc_critical), thrust_medium,
         "SE velocidade do ar alta E descida crítica -> potência média"),
    ]

    clipped_outputs = [np.minimum(consequent, activation) for _, activation, consequent, _ in rules]
    aggregated = np.maximum.reduce(clipped_outputs)
    thrust_value = defuzz_centroid(x_thrust, aggregated)

    strongest_rule = max(rules, key=lambda item: item[1])

    return {
        "x_air": x_air,
        "x_desc": x_desc,
        "x_thrust": x_thrust,
        "air_sets": (air_low, air_ok, air_high),
        "desc_sets": (desc_small, desc_medium, desc_critical),
        "thrust_sets": (thrust_low, thrust_medium, thrust_high, thrust_max),
        "mu_inputs": {
            "air_low": mu_air_low,
            "air_ok": mu_air_ok,
            "air_high": mu_air_high,
            "desc_small": mu_desc_small,
            "desc_medium": mu_desc_medium,
            "desc_critical": mu_desc_critical,
        },
        "rules": rules,
        "aggregated": aggregated,
        "thrust_value": thrust_value,
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
    print("=== SIMULAÇÃO DE DRONE COM CONTROLE FUZZY ===\n")
    print("Pressione Enter para usar os valores padrão.\n")

    initial_altitude = read_float("Altitude inicial do drone (m)", 55)
    initial_airspeed = read_float("Velocidade inicial do ar medida pelo Pitot (m/s)", 16)
    initial_descent = read_float("Taxa inicial de descida (m/s)", 1.2)
    map_length = read_float("Comprimento do mapa (unidades gráficas)", 140)
    frame_interval = read_float("Intervalo entre frames (ms)", 120)
    max_frames = int(read_float("Número máximo de frames", 260))

    if initial_altitude < 10:
        initial_altitude = 55.0
    if initial_airspeed < 0:
        initial_airspeed = 16.0
    if initial_descent < 0:
        initial_descent = 1.2
    if map_length < 80:
        map_length = 140.0
    if frame_interval < 10:
        frame_interval = 120.0
    if max_frames < 20:
        max_frames = 260

    return {
        "initial_altitude": initial_altitude,
        "initial_airspeed": initial_airspeed,
        "initial_descent": initial_descent,
        "map_length": map_length,
        "frame_interval": frame_interval,
        "max_frames": max_frames,
    }


# =========================================================
# Relatório inicial
# =========================================================

def print_initial_report(initial_airspeed, initial_descent):
    res = fuzzy_drone(initial_airspeed, initial_descent)

    print("\n=== ANÁLISE INICIAL DO SISTEMA FUZZY ===\n")
    print(f"Velocidade do ar inicial : {initial_airspeed:.2f} m/s")
    print(f"Taxa de descida inicial  : {initial_descent:.2f} m/s\n")

    print("[Fuzzificação - Velocidade do ar]")
    print(f"μ(baixa) = {res['mu_inputs']['air_low']:.3f}")
    print(f"μ(ideal) = {res['mu_inputs']['air_ok']:.3f}")
    print(f"μ(alta)  = {res['mu_inputs']['air_high']:.3f}\n")

    print("[Fuzzificação - Taxa de descida]")
    print(f"μ(pequena)  = {res['mu_inputs']['desc_small']:.3f}")
    print(f"μ(moderada) = {res['mu_inputs']['desc_medium']:.3f}")
    print(f"μ(crítica)  = {res['mu_inputs']['desc_critical']:.3f}\n")

    print("[Ativação das regras]")
    for name, activation, _, description in res["rules"]:
        print(f"{name}: {activation:.3f} -> {description}")

    print("\n[Defuzzificação]")
    print(f"Potência inicial recomendada = {res['thrust_value']:.2f} %")
    print(f"Regra dominante: {res['strongest_rule_name']} "
          f"(ativação = {res['strongest_rule_activation']:.3f})")
    print(f"Descrição: {res['strongest_rule_text']}\n")


# =========================================================
# Mapa editável
# Os alunos podem alterar essas zonas
# headwind  -> reduz a velocidade do ar
# downdraft -> aumenta a taxa de descida
# =========================================================

WIND_ZONES = [
    {"x_start": 20, "x_end": 40, "headwind": 3.0, "downdraft": 0.6},
    {"x_start": 55, "x_end": 75, "headwind": 6.0, "downdraft": 1.2},
    {"x_start": 95, "x_end": 115, "headwind": 2.0, "downdraft": 2.2},
]


def get_environment_effects(x_position):
    headwind = 0.0
    downdraft = 0.0

    for zone in WIND_ZONES:
        if zone["x_start"] <= x_position <= zone["x_end"]:
            headwind += zone["headwind"]
            downdraft += zone["downdraft"]

    return headwind, downdraft


# =========================================================
# Animação
# =========================================================

def animate_fuzzy_drone(initial_altitude, initial_airspeed, initial_descent,
                        map_length=140, frame_interval=120, max_frames=260):

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.28, wspace=0.22)

    ax_scene = fig.add_subplot(gs[0, :])
    ax_air = fig.add_subplot(gs[1, 0])
    ax_out = fig.add_subplot(gs[1, 1])

    fig.suptitle("Sistema Fuzzy Mamdani - Controle de Drone com Interferência Manual", fontsize=15)

    # -----------------------------------------------------
    # Subplot 1: Cena
    # -----------------------------------------------------
    ax_scene.set_xlim(0, map_length)
    ax_scene.set_ylim(0, 80)
    ax_scene.set_xlabel("Posição horizontal")
    ax_scene.set_ylabel("Altitude")
    ax_scene.grid(True, alpha=0.25)

    # solo
    ax_scene.plot([0, map_length], [0, 0], linewidth=3)

    # desenha zonas de vento
    for zone in WIND_ZONES:
        rect = Rectangle(
            (zone["x_start"], 0),
            zone["x_end"] - zone["x_start"],
            80,
            alpha=0.12
        )
        ax_scene.add_patch(rect)
        ax_scene.text(
            (zone["x_start"] + zone["x_end"]) / 2,
            74,
            f"V={zone['headwind']:.1f}\nD={zone['downdraft']:.1f}",
            ha="center",
            va="top",
            fontsize=9
        )

    # drone
    drone_body = Rectangle((5, initial_altitude), 6, 2.5, fill=False, linewidth=2)
    rotor_left = Circle((6.2, initial_altitude + 3.0), 0.8, fill=False, linewidth=1.8)
    rotor_right = Circle((9.8, initial_altitude + 3.0), 0.8, fill=False, linewidth=1.8)
    ax_scene.add_patch(drone_body)
    ax_scene.add_patch(rotor_left)
    ax_scene.add_patch(rotor_right)
    ax_scene.text(8, initial_altitude + 5.4, "DRONE", ha="center", fontsize=10)

    # trilha
    trail_line, = ax_scene.plot([], [], linestyle="--", linewidth=1.2)

    # textos
    txt_airspeed = ax_scene.text(2, 76.5, "", fontsize=10)
    txt_descent = ax_scene.text(2, 72.5, "", fontsize=10)
    txt_altitude = ax_scene.text(2, 68.5, "", fontsize=10)
    txt_thrust = ax_scene.text(2, 64.5, "", fontsize=10)
    txt_env = ax_scene.text(2, 60.5, "", fontsize=10)
    txt_manual = ax_scene.text(2, 56.5, "", fontsize=10)
    txt_status = ax_scene.text(2, 52.5, "", fontsize=10)
    txt_rule = ax_scene.text(2, 4.0, "", fontsize=9)

    # barra de potência
    bar_x, bar_y, bar_w, bar_h = map_length * 0.72, 69.0, map_length * 0.18, 3.2
    thrust_bar_border = Rectangle((bar_x, bar_y), bar_w, bar_h, fill=False, linewidth=1.5)
    thrust_bar_fill = Rectangle((bar_x, bar_y), 0.0, bar_h, fill=True, alpha=0.5)
    ax_scene.add_patch(thrust_bar_border)
    ax_scene.add_patch(thrust_bar_fill)
    ax_scene.text(bar_x + bar_w / 2, bar_y + 4.5, "Potência do motor", ha="center", fontsize=10)

    # -----------------------------------------------------
    # Subplot 2: Entrada fuzzy - velocidade do ar
    # -----------------------------------------------------
    x_air = np.linspace(0, 30, 500)
    air_low = trapezoidal(x_air, 0, 0, 8, 13)
    air_ok = triangular(x_air, 10, 16, 22)
    air_high = trapezoidal(x_air, 19, 24, 30, 30)

    ax_air.plot(x_air, air_low, label="Baixa", linewidth=2)
    ax_air.plot(x_air, air_ok, label="Ideal", linewidth=2)
    ax_air.plot(x_air, air_high, label="Alta", linewidth=2)
    air_line = ax_air.axvline(initial_airspeed, linestyle="--", linewidth=2, label="Leitura Pitot")
    ax_air.set_title("Entrada Fuzzy - Velocidade do ar (Pitot)")
    ax_air.set_xlabel("Velocidade do ar (m/s)")
    ax_air.set_ylabel("μ(x)")
    ax_air.set_ylim(-0.05, 1.05)
    ax_air.grid(True)
    ax_air.legend()

    # -----------------------------------------------------
    # Subplot 3: Saída fuzzy - potência
    # -----------------------------------------------------
    x_thrust = np.linspace(0, 100, 600)
    thrust_low = trapezoidal(x_thrust, 0, 0, 18, 32)
    thrust_medium = triangular(x_thrust, 25, 45, 65)
    thrust_high = triangular(x_thrust, 55, 72, 88)
    thrust_max = trapezoidal(x_thrust, 82, 92, 100, 100)

    ax_out.plot(x_thrust, thrust_low, "--", linewidth=1.5, label="Baixa")
    ax_out.plot(x_thrust, thrust_medium, "--", linewidth=1.5, label="Média")
    ax_out.plot(x_thrust, thrust_high, "--", linewidth=1.5, label="Alta")
    ax_out.plot(x_thrust, thrust_max, "--", linewidth=1.5, label="Máxima")

    initial_res = fuzzy_drone(initial_airspeed, initial_descent)
    aggregated_fill = ax_out.fill_between(
        initial_res["x_thrust"], 0, initial_res["aggregated"], alpha=0.55, label="Agregação"
    )
    centroid_line = ax_out.axvline(initial_res["thrust_value"], linewidth=2, label="Centróide")

    ax_out.set_title("Saída Fuzzy - Potência do motor")
    ax_out.set_xlabel("Potência (%)")
    ax_out.set_ylabel("μ(x)")
    ax_out.set_ylim(-0.05, 1.05)
    ax_out.grid(True)
    ax_out.legend()

    # -----------------------------------------------------
    # Estado do sistema
    # -----------------------------------------------------
    state = {
        "x": 5.0,
        "y": initial_altitude,
        "base_airspeed": initial_airspeed,
        "vertical_speed": initial_descent,   # positiva = descendo
        "stopped": False,
        "crashed": False,
        "finished": False,
        "history_x": [],
        "history_y": [],
    }

    # Perturbações manuais
    manual = {
        "extra_headwind": 0.0,
        "extra_downdraft": 0.0,
        "pitot_bias": 0.0,
    }

    # -----------------------------------------------------
    # Controle por teclado
    # -----------------------------------------------------
    def on_key_press(event):
        key = event.key.lower() if event.key else ""

        if key == "a":
            manual["extra_headwind"] += 1.5
        elif key == "d":
            manual["extra_headwind"] -= 1.5
        elif key == "w":
            manual["extra_downdraft"] -= 0.5   # corrente ascendente
        elif key == "s":
            manual["extra_downdraft"] += 0.5   # corrente descendente
        elif key == "p":
            manual["pitot_bias"] -= 1.5        # Pitot lê menos do que o real
        elif key == "r":
            manual["extra_headwind"] = 0.0
            manual["extra_downdraft"] = 0.0
            manual["pitot_bias"] = 0.0

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # -----------------------------------------------------
    # Atualização
    # -----------------------------------------------------
    def update(frame):
        nonlocal aggregated_fill

        if state["crashed"] or state["finished"]:
            return (
                drone_body, rotor_left, rotor_right, thrust_bar_fill,
                txt_airspeed, txt_descent, txt_altitude, txt_thrust,
                txt_env, txt_manual, txt_status, txt_rule,
                air_line, centroid_line, trail_line
            )

        # ambiente local
        zone_headwind, zone_downdraft = get_environment_effects(state["x"])

        total_headwind = zone_headwind + manual["extra_headwind"]
        total_downdraft = zone_downdraft + manual["extra_downdraft"]

        # velocidade do ar real
        real_airspeed = max(state["base_airspeed"] - total_headwind, 0.0)

        # leitura do Pitot com possível erro
        measured_airspeed = max(real_airspeed + manual["pitot_bias"], 0.0)

        # taxa de descida medida
        descent_rate = max(state["vertical_speed"] + total_downdraft, 0.0)

        # fuzzy
        fuzzy = fuzzy_drone(measured_airspeed, descent_rate)
        thrust = fuzzy["thrust_value"]

        # dinâmica simplificada/didática
        # maior potência reduz a descida
        lift_effect = 0.055 * thrust
        gravity_effect = 2.6
        state["vertical_speed"] = gravity_effect + total_downdraft - lift_effect
        state["vertical_speed"] = np.clip(state["vertical_speed"], -2.5, 7.0)

        # avanço horizontal simplificado
        horizontal_step = max(real_airspeed * 0.28, 0.4)

        # atualiza posição
        state["x"] += horizontal_step
        state["y"] -= state["vertical_speed"] * 0.28

        # limites
        if state["y"] <= 0:
            state["y"] = 0
            state["crashed"] = True

        if state["x"] >= map_length - 8:
            state["finished"] = True

        # atualiza histórico
        state["history_x"].append(state["x"] + 3)
        state["history_y"].append(state["y"] + 1.2)
        trail_line.set_data(state["history_x"], state["history_y"])

        # atualiza drone
        drone_body.set_x(state["x"])
        drone_body.set_y(state["y"])
        rotor_left.center = (state["x"] + 1.2, state["y"] + 3.0)
        rotor_right.center = (state["x"] + 4.8, state["y"] + 3.0)

        # atualiza barra de potência
        thrust_bar_fill.set_width((thrust / 100.0) * bar_w)

        # atualiza textos
        txt_airspeed.set_text(
            f"Velocidade do ar (Pitot): {measured_airspeed:.2f} m/s | real: {real_airspeed:.2f} m/s"
        )
        txt_descent.set_text(f"Taxa de descida: {descent_rate:.2f} m/s")
        txt_altitude.set_text(f"Altitude: {state['y']:.2f} m")
        txt_thrust.set_text(f"Potência fuzzy: {thrust:.2f} %")
        txt_env.set_text(
            f"Ambiente local -> vento contrário: {zone_headwind:.2f} | corrente descendente: {zone_downdraft:.2f}"
        )
        txt_manual.set_text(
            f"Manual -> A/D vento: {manual['extra_headwind']:.2f} | W/S corrente: {manual['extra_downdraft']:.2f} | P Pitot: {manual['pitot_bias']:.2f}"
        )

        if state["crashed"]:
            txt_status.set_text("Status: colisão com o solo")
        elif state["finished"]:
            txt_status.set_text("Status: percurso concluído")
        elif state["y"] < 12:
            txt_status.set_text("Status: altitude crítica")
        elif descent_rate > 5:
            txt_status.set_text("Status: descida severa")
        elif measured_airspeed < 8:
            txt_status.set_text("Status: risco de baixa velocidade")
        else:
            txt_status.set_text("Status: controle ativo")

        txt_rule.set_text(
            f"Regra dominante: {fuzzy['strongest_rule_name']} | "
            f"ativação = {fuzzy['strongest_rule_activation']:.3f} | "
            f"{fuzzy['strongest_rule_text']}"
        )

        # atualiza linha da entrada fuzzy
        air_line.set_xdata([measured_airspeed, measured_airspeed])

        # atualiza saída fuzzy
        aggregated_fill.remove()
        aggregated_fill = ax_out.fill_between(
            fuzzy["x_thrust"], 0, fuzzy["aggregated"], alpha=0.55
        )
        centroid_line.set_xdata([thrust, thrust])

        return (
            drone_body, rotor_left, rotor_right, thrust_bar_fill,
            txt_airspeed, txt_descent, txt_altitude, txt_thrust,
            txt_env, txt_manual, txt_status, txt_rule,
            air_line, centroid_line, trail_line
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=frame_interval,
        repeat=False,
        blit=False
    )

    print("Controles durante a simulação:")
    print("  A -> aumenta vento contrário")
    print("  D -> aumenta vento favorável")
    print("  W -> corrente ascendente")
    print("  S -> corrente descendente")
    print("  P -> falha parcial no Pitot")
    print("  R -> resetar perturbações manuais\n")

    plt.show()
    return anim


# =========================================================
# Programa principal
# =========================================================

def main():
    params = read_user_parameters()

    print_initial_report(
        params["initial_airspeed"],
        params["initial_descent"]
    )

    print("Iniciando animação...\n")

    animate_fuzzy_drone(
        initial_altitude=params["initial_altitude"],
        initial_airspeed=params["initial_airspeed"],
        initial_descent=params["initial_descent"],
        map_length=params["map_length"],
        frame_interval=params["frame_interval"],
        max_frames=params["max_frames"],
    )

    print("Mensagem-chave da aula:")
    print("- Em lógica fuzzy, múltiplas regras podem atuar simultaneamente.")
    print("- A saída final depende da agregação dos consequentes.")
    print("- Sensores e perturbações externas alteram diretamente a inferência.")
    print("- Um sistema fuzzy pode ser robusto, mas não é mágico: sensor ruim e ambiente hostil cobram a conta.")


if __name__ == "__main__":
    main()