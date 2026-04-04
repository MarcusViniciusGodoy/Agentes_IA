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