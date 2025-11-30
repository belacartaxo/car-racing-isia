import numpy as np
import pygame.gfxdraw
from gymnasium.envs.box2d.car_racing import CarRacing
from Box2D.b2 import b2CircleShape, b2FixtureDef, b2BodyDef, b2_staticBody


class CustomCarRacingEnv(CarRacing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.obstacle_radius = 2.0      # raio do cone no mundo Box2D
        self.obstacle_positions = []
        self.obstacle_bodies = []

    # -------------------------------  
    # GERA OBSTÁCULOS NA BORDA DA PISTA
    # -------------------------------
    def generate_obstacles_near_edge(self, num_obstacles=10):
        self.obstacle_positions = []

        for _ in range(num_obstacles):
            tile = self.track[np.random.randint(0, len(self.track))]
            _, _, cx, cy = tile  # track item = (alpha, beta, x, y)

            width = 40 / 6.0  # TRACK_WIDTH / SCALE (valor da pista normal)

            edge_dist = np.random.uniform(0.6, 0.9) * width
            side = np.random.choice([-1, 1])

            ox = cx + side * edge_dist
            oy = cy

            self.obstacle_positions.append((ox, oy))

    # -------------------------------  
    # RECRIAR PISTA E CRIAR CORPOS BOX2D
    # -------------------------------
    def _recreate_track(self, **args):
        super()._recreate_track(**args)

        for body in self.obstacle_bodies:
            self.world.DestroyBody(body)
        self.obstacle_bodies.clear()

        self.generate_obstacles_near_edge(12)

        for (x, y) in self.obstacle_positions:
            body_def = b2BodyDef(type=b2_staticBody, position=(x, y))
            body = self.world.CreateBody(body_def)

            fixture = b2FixtureDef(
                shape=b2CircleShape(radius=self.obstacle_radius),
                density=1.0,
                friction=0.9
            )
            body.CreateFixture(fixture)
            self.obstacle_bodies.append(body)

    # -------------------------------  
    # DESENHAR CONES NA TELA (USANDO A MESMA CAMÊRA DO JOGO)
    # -------------------------------
    def _render_cones(self, zoom, translation, angle):
        for (ox, oy) in self.obstacle_positions:
            pos = pygame.math.Vector2((ox, oy)).rotate_rad(angle)

            px = pos[0] * zoom + translation[0]
            py = pos[1] * zoom + translation[1]

            # tamanho do cone na tela
            radius_px = int(self.obstacle_radius * zoom)

            # clipping simples
            if px < -50 or px > 1050 or py < -50 or py > 850:
                continue

            pygame.gfxdraw.filled_circle(self.surf, int(px), int(py), radius_px, (255, 128, 0))
            pygame.gfxdraw.aacircle(self.surf, int(px), int(py), radius_px, (255, 128, 0))

    # -------------------------------  
    # RENDER PRINCIPAL (CHAMADO APÓS DESENHAR O CARRO/PISTA)
    # -------------------------------
    def _render(self, mode):
        result = super()._render(mode)

        if mode != "human":
            return result

        # Extração dos parâmetros da câmera
        zoom = self.zoom * self.SCALE if hasattr(self, "zoom") else 2.7 * self.SCALE

        # Valores usados no draw oficial
        angle = -self.car.hull.angle
        zoom = 0.1 * self.SCALE * max(1 - self.t, 0) + 2.7 * self.SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom

        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        translation = (1000 / 2 + trans[0], 800 / 4 + trans[1])

        # DESENHA OS CONES AQUI
        self._render_cones(zoom, translation, angle)

        return result

    # -------------------------------  
    # COLISÃO COM OBSTÁCULOS
    # -------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        car_x, car_y = self.car.hull.position
        for (ox, oy) in self.obstacle_positions:
            dist = np.sqrt((car_x - ox) ** 2 + (car_y - oy) ** 2)
            if dist < self.obstacle_radius + 1.0:
                print("Collision with obstacle!")
                reward -= 20
                terminated = True
                break

        return obs, reward, terminated, truncated, info
