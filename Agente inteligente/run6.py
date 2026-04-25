# Aula 08 - Agente Inteligente para Controle de Trânsito
# Python 3.10+ | Requer: pygame
#
# Instalação:
# pip install pygame

import pygame
import random
from dataclasses import dataclass

pygame.init()

WIDTH, HEIGHT = 1100, 720
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Aula 08 - Agente Inteligente de Trânsito")

CLOCK = pygame.time.Clock()
FPS = 60

FONT = pygame.font.SysFont("consolas", 18)
BIG_FONT = pygame.font.SysFont("consolas", 28, bold=True)

BG = (25, 28, 35)
ROAD = (55, 55, 60)
ROAD_LINE = (220, 220, 220)
GREEN = (80, 220, 120)
RED = (230, 80, 80)
YELLOW = (240, 210, 80)
WHITE = (245, 245, 245)
BLUE = (80, 160, 255)
ORANGE = (255, 165, 80)
GRAY = (140, 140, 140)

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
ROAD_W = 180


@dataclass
class Car:
    lane: str
    x: float
    y: float
    speed: float
    emergency: bool = False
    wait_time: float = 0.0

    def color(self):
        return ORANGE if self.emergency else BLUE


class TrafficSimulation:
    def __init__(self):
        self.cars = []
        self.green_axis = "NS"
        self.timer = 0
        self.spawn_rate = 0.035
        self.ai_enabled = True
        self.paused = False
        self.rain = False
        self.sensor_noise = 0.15
        self.total_passed = 0
        self.emergency_count = 0

        self.rain_particles = [
            [random.randint(0, WIDTH), random.randint(0, HEIGHT)]
            for _ in range(130)
        ]

    def axis_of_lane(self, lane):
        return "NS" if lane in ["N", "S"] else "EW"

    def spawn_car(self, lane=None, emergency=False):
        if lane is None:
            lane = random.choice(["N", "S", "E", "W"])

        if self.spawn_blocked(lane):
            return

        speed = random.uniform(1.8, 2.8)

        if emergency:
            speed = 3.5
            self.emergency_count += 1

        if lane == "N":
            car = Car(lane, CENTER_X - 45, -40, speed, emergency)
        elif lane == "S":
            car = Car(lane, CENTER_X + 45, HEIGHT + 40, speed, emergency)
        elif lane == "E":
            car = Car(lane, WIDTH + 40, CENTER_Y - 45, speed, emergency)
        else:
            car = Car(lane, -40, CENTER_Y + 45, speed, emergency)

        self.cars.append(car)

    def spawn_blocked(self, lane, min_dist=85):
        for car in self.cars:
            if car.lane != lane:
                continue

            if lane == "N" and car.y < min_dist:
                return True
            if lane == "S" and car.y > HEIGHT - min_dist:
                return True
            if lane == "E" and car.x > WIDTH - min_dist:
                return True
            if lane == "W" and car.x < min_dist:
                return True

        return False

    def inside_intersection(self, car):
        return (
            CENTER_X - ROAD_W // 2 <= car.x <= CENTER_X + ROAD_W // 2 and
            CENTER_Y - ROAD_W // 2 <= car.y <= CENTER_Y + ROAD_W // 2
        )

    def has_crossed_stop_line(self, car):
        if car.lane == "N":
            return car.y > CENTER_Y - ROAD_W // 2
        if car.lane == "S":
            return car.y < CENTER_Y + ROAD_W // 2
        if car.lane == "E":
            return car.x < CENTER_X + ROAD_W // 2
        if car.lane == "W":
            return car.x > CENTER_X - ROAD_W // 2
        return False

    def stop_line_reached(self, car):
        if car.lane == "N":
            return car.y >= CENTER_Y - ROAD_W // 2 - 35
        if car.lane == "S":
            return car.y <= CENTER_Y + ROAD_W // 2 + 35
        if car.lane == "E":
            return car.x <= CENTER_X + ROAD_W // 2 + 35
        if car.lane == "W":
            return car.x >= CENTER_X - ROAD_W // 2 - 35
        return False

    def passed_intersection(self, car):
        margin = 80
        return (
            car.x < -margin or car.x > WIDTH + margin or
            car.y < -margin or car.y > HEIGHT + margin
        )

    def is_car_ahead(self, car, min_dist=55):
        # Se já está no cruzamento, ele precisa liberar a área.
        if self.inside_intersection(car):
            return False

        for other in self.cars:
            if other is car:
                continue

            if other.lane != car.lane:
                continue

            if car.lane == "N" and other.y > car.y:
                if abs(other.y - car.y) < min_dist:
                    return True

            elif car.lane == "S" and other.y < car.y:
                if abs(other.y - car.y) < min_dist:
                    return True

            elif car.lane == "E" and other.x < car.x:
                if abs(other.x - car.x) < min_dist:
                    return True

            elif car.lane == "W" and other.x > car.x:
                if abs(other.x - car.x) < min_dist:
                    return True

        return False

    def can_move(self, car):
        axis = self.axis_of_lane(car.lane)

        # Depois que entrou no cruzamento, não pode parar no meio.
        if self.inside_intersection(car) or self.has_crossed_stop_line(car):
            return True

        # Ambulância pode furar parcialmente o sinal.
        if car.emergency:
            return random.random() < 0.65

        # Sinal verde para o eixo.
        if axis == self.green_axis:
            return True

        # Sinal vermelho antes da linha de parada.
        if self.stop_line_reached(car):
            return False

        return True

    def update_cars(self):
        for car in self.cars:
            blocked_by_car = self.is_car_ahead(car)
            allowed_by_light = self.can_move(car)

            if allowed_by_light and not blocked_by_car:
                if car.lane == "N":
                    car.y += car.speed
                elif car.lane == "S":
                    car.y -= car.speed
                elif car.lane == "E":
                    car.x -= car.speed
                elif car.lane == "W":
                    car.x += car.speed
            else:
                car.wait_time += 1 / FPS

        before = len(self.cars)
        self.cars = [c for c in self.cars if not self.passed_intersection(c)]
        self.total_passed += before - len(self.cars)

    def get_lane_stats(self):
        stats = {
            "NS": {"queue": 0, "wait": 0, "emergency": 0},
            "EW": {"queue": 0, "wait": 0, "emergency": 0},
        }

        for car in self.cars:
            axis = self.axis_of_lane(car.lane)

            if self.stop_line_reached(car) and not self.has_crossed_stop_line(car):
                stats[axis]["queue"] += 1
                stats[axis]["wait"] += car.wait_time

            if car.emergency:
                stats[axis]["emergency"] += 1

        return stats

    def noisy_measurement(self, value):
        if random.random() < self.sensor_noise:
            return value * random.uniform(0.6, 1.4)
        return value

    def ai_decision(self):
        stats = self.get_lane_stats()

        ns_queue = self.noisy_measurement(stats["NS"]["queue"])
        ew_queue = self.noisy_measurement(stats["EW"]["queue"])

        ns_wait = self.noisy_measurement(stats["NS"]["wait"])
        ew_wait = self.noisy_measurement(stats["EW"]["wait"])

        ns_emergency = stats["NS"]["emergency"]
        ew_emergency = stats["EW"]["emergency"]

        rain_factor = 1.4 if self.rain else 1.0

        score_ns = (
            2.0 * ns_queue +
            0.25 * ns_wait +
            8.0 * ns_emergency
        ) * rain_factor

        score_ew = (
            2.0 * ew_queue +
            0.25 * ew_wait +
            8.0 * ew_emergency
        ) * rain_factor

        if abs(score_ns - score_ew) < 1.0:
            return self.green_axis

        return "NS" if score_ns > score_ew else "EW"

    def update_lights(self):
        self.timer += 1

        if self.ai_enabled:
            if self.timer > FPS * 4:
                self.green_axis = self.ai_decision()
                self.timer = 0
        else:
            if self.timer > FPS * 6:
                self.green_axis = "EW" if self.green_axis == "NS" else "NS"
                self.timer = 0

    def update(self):
        if self.paused:
            return

        if random.random() < self.spawn_rate:
            self.spawn_car()

        self.update_lights()
        self.update_cars()

    def draw_roads(self):
        SCREEN.fill(BG)

        pygame.draw.rect(
            SCREEN, ROAD,
            (CENTER_X - ROAD_W // 2, 0, ROAD_W, HEIGHT)
        )
        pygame.draw.rect(
            SCREEN, ROAD,
            (0, CENTER_Y - ROAD_W // 2, WIDTH, ROAD_W)
        )

        for y in range(0, HEIGHT, 45):
            pygame.draw.rect(SCREEN, ROAD_LINE, (CENTER_X - 3, y, 6, 25))

        for x in range(0, WIDTH, 45):
            pygame.draw.rect(SCREEN, ROAD_LINE, (x, CENTER_Y - 3, 25, 6))

        pygame.draw.rect(
            SCREEN,
            (35, 35, 38),
            (CENTER_X - ROAD_W // 2, CENTER_Y - ROAD_W // 2, ROAD_W, ROAD_W)
        )

    def draw_lights(self):
        ns_color = GREEN if self.green_axis == "NS" else RED
        ew_color = GREEN if self.green_axis == "EW" else RED

        pygame.draw.circle(
            SCREEN, ns_color, (CENTER_X - 120, CENTER_Y - 120), 16)
        pygame.draw.circle(
            SCREEN, ns_color, (CENTER_X + 120, CENTER_Y + 120), 16)

        pygame.draw.circle(
            SCREEN, ew_color, (CENTER_X + 120, CENTER_Y - 120), 16)
        pygame.draw.circle(
            SCREEN, ew_color, (CENTER_X - 120, CENTER_Y + 120), 16)

    def draw_cars(self):
        for car in self.cars:
            if car.lane in ["N", "S"]:
                rect = pygame.Rect(car.x - 13, car.y - 22, 26, 44)
            else:
                rect = pygame.Rect(car.x - 22, car.y - 13, 44, 26)

            pygame.draw.rect(SCREEN, car.color(), rect, border_radius=6)

            if car.emergency:
                pygame.draw.circle(SCREEN, WHITE, rect.center, 5)

    def draw_rain(self):
        if not self.rain:
            return

        for p in self.rain_particles:
            pygame.draw.line(SCREEN, (120, 160, 255), p,
                             (p[0] - 4, p[1] + 12), 1)
            p[0] -= 2
            p[1] += 8

            if p[1] > HEIGHT:
                p[0] = random.randint(0, WIDTH)
                p[1] = random.randint(-60, 0)

    def draw_dashboard(self):
        panel = pygame.Rect(780, 20, 295, 245)
        pygame.draw.rect(SCREEN, (35, 38, 48), panel, border_radius=12)
        pygame.draw.rect(SCREEN, (80, 85, 100), panel, 2, border_radius=12)

        stats = self.get_lane_stats()

        title = BIG_FONT.render("Agente de Trânsito", True, WHITE)
        SCREEN.blit(title, (800, 35))

        lines = [
            f"Modo IA: {'ON' if self.ai_enabled else 'OFF'}",
            f"Semáforo verde: {self.green_axis}",
            f"Chuva: {'SIM' if self.rain else 'NAO'}",
            f"Fluxo: {self.spawn_rate:.3f}",
            f"Carros ativos: {len(self.cars)}",
            f"Carros liberados: {self.total_passed}",
            f"Ambulâncias: {self.emergency_count}",
            "",
            f"Fila NS: {stats['NS']['queue']} | Espera: {stats['NS']['wait']:.1f}",
            f"Fila EW: {stats['EW']['queue']} | Espera: {stats['EW']['wait']:.1f}",
        ]

        y = 78
        for line in lines:
            txt = FONT.render(line, True, WHITE)
            SCREEN.blit(txt, (800, y))
            y += 20

        help_lines = [
            "A: IA on/off",
            "E: ambulância",
            "R: chuva",
            "+/-: fluxo",
            "ESPAÇO: pausar",
        ]

        y = 300
        for line in help_lines:
            txt = FONT.render(line, True, (210, 210, 210))
            SCREEN.blit(txt, (800, y))
            y += 23

    def draw_scores(self):
        stats = self.get_lane_stats()

        ns_score = (
            2 * stats["NS"]["queue"] +
            0.25 * stats["NS"]["wait"] +
            8 * stats["NS"]["emergency"]
        )

        ew_score = (
            2 * stats["EW"]["queue"] +
            0.25 * stats["EW"]["wait"] +
            8 * stats["EW"]["emergency"]
        )

        max_score = max(ns_score, ew_score, 1)

        x, y = 800, 450

        label = FONT.render("Score de decisão do agente", True, WHITE)
        SCREEN.blit(label, (x, y))

        pygame.draw.rect(SCREEN, GRAY, (x, y + 35, 240, 22), border_radius=6)
        pygame.draw.rect(
            SCREEN,
            GREEN,
            (x, y + 35, int(240 * ns_score / max_score), 22),
            border_radius=6
        )
        SCREEN.blit(FONT.render(
            f"NS: {ns_score:.1f}", True, WHITE), (x, y + 62))

        pygame.draw.rect(SCREEN, GRAY, (x, y + 95, 240, 22), border_radius=6)
        pygame.draw.rect(
            SCREEN,
            YELLOW,
            (x, y + 95, int(240 * ew_score / max_score), 22),
            border_radius=6
        )
        SCREEN.blit(FONT.render(
            f"EW: {ew_score:.1f}", True, WHITE), (x, y + 122))

    def draw(self):
        self.draw_roads()
        self.draw_lights()
        self.draw_cars()
        self.draw_rain()
        self.draw_dashboard()
        self.draw_scores()

        if self.paused:
            txt = BIG_FONT.render("PAUSADO", True, WHITE)
            SCREEN.blit(txt, (CENTER_X - 60, 40))

        pygame.display.flip()

    def handle_key(self, key):
        if key == pygame.K_SPACE:
            self.paused = not self.paused

        elif key == pygame.K_a:
            self.ai_enabled = not self.ai_enabled

        elif key == pygame.K_e:
            self.spawn_car(emergency=True)

        elif key == pygame.K_r:
            self.rain = not self.rain
            self.sensor_noise = 0.35 if self.rain else 0.15

        elif key in [pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS]:
            self.spawn_rate = min(0.12, self.spawn_rate + 0.01)

        elif key in [pygame.K_MINUS, pygame.K_KP_MINUS]:
            self.spawn_rate = max(0.005, self.spawn_rate - 0.01)


def main():
    sim = TrafficSimulation()
    running = True

    while running:
        CLOCK.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                sim.handle_key(event.key)

        sim.update()
        sim.draw()

    pygame.quit()


if __name__ == "__main__":
    main()
