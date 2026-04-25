# Aula 08 - Simulador de Decisão com Incerteza
# Agente probabilístico em ambiente 2D
# Python 3.10+ | Requer: matplotlib, numpy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


GRID_SIZE = 10


class ProbabilisticAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = (np.random.randint(GRID_SIZE),
                       np.random.randint(GRID_SIZE))
        self.position = (np.random.randint(GRID_SIZE),
                         np.random.randint(GRID_SIZE))

        # distribuição inicial uniforme
        self.belief = np.ones((GRID_SIZE, GRID_SIZE))
        self.belief /= self.belief.sum()

    def sense(self, noise):
        """
        Sensor retorna distância ao alvo com ruído
        """
        true_dist = abs(self.position[0] - self.target[0]) + \
            abs(self.position[1] - self.target[1])

        if np.random.rand() < noise:
            return true_dist + np.random.randint(-2, 3)
        return true_dist

    def update_belief(self, observed_dist):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                dist = abs(self.position[0] - i) + abs(self.position[1] - j)

                likelihood = np.exp(-abs(dist - observed_dist))
                self.belief[i, j] *= likelihood

        self.belief += 1e-6
        self.belief /= self.belief.sum()

    def move(self):
        # move para região de maior probabilidade
        target_estimate = np.unravel_index(
            np.argmax(self.belief), self.belief.shape)

        dx = np.sign(target_estimate[0] - self.position[0])
        dy = np.sign(target_estimate[1] - self.position[1])

        self.position = (
            np.clip(self.position[0] + dx, 0, GRID_SIZE - 1),
            np.clip(self.position[1] + dy, 0, GRID_SIZE - 1),
        )


class SimulationUI:
    def __init__(self):
        self.agent = ProbabilisticAgent()

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25)

        # sliders
        self.ax_noise = plt.axes([0.15, 0.1, 0.65, 0.03])
        self.ax_step = plt.axes([0.15, 0.05, 0.3, 0.04])
        self.ax_reset = plt.axes([0.6, 0.05, 0.2, 0.04])

        self.slider_noise = Slider(
            self.ax_noise, "Ruído", 0.0, 0.9, valinit=0.2)
        self.btn_step = Button(self.ax_step, "Passo")
        self.btn_reset = Button(self.ax_reset, "Reset")

        self.btn_step.on_clicked(self.step)
        self.btn_reset.on_clicked(self.reset)

        self.draw()

    def draw(self):
        self.ax.clear()

        self.ax.imshow(self.agent.belief, cmap="viridis")

        x, y = self.agent.position
        tx, ty = self.agent.target

        self.ax.scatter(y, x, c="red", label="Agente")
        self.ax.scatter(ty, tx, c="white", marker="x", label="Alvo")

        self.ax.set_title("Mapa de crença (probabilidade)")
        self.ax.legend()

        self.fig.canvas.draw_idle()

    def step(self, event):
        noise = self.slider_noise.val

        observed = self.agent.sense(noise)
        self.agent.update_belief(observed)
        self.agent.move()

        self.draw()

    def reset(self, event):
        self.agent.reset()
        self.draw()

    def run(self):
        plt.show()


if __name__ == "__main__":
    ui = SimulationUI()
    ui.run()
