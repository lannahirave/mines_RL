"""
Pygame renderer for game visualization.
"""

import pygame
import numpy as np
from typing import Tuple, List, Optional

from .snake import Snake, Direction
from .game_objects import GameObject, ObjectType


class Renderer:
    """Game visualization using Pygame."""

    # Colors
    COLORS = {
        "background": (30, 30, 40),
        "grid": (50, 50, 60),
        "snake_head": (0, 200, 100),
        "snake_body": (0, 150, 80),
        "apple": (220, 50, 50),
        "golden": (255, 215, 0),
        "poison": (20, 20, 20),
        "poison_skull": (200, 200, 200),
        "sour": (255, 255, 100),
        "rotten": (139, 90, 43),
        "obstacle": (100, 100, 120),
        "text": (255, 255, 255),
    }

    def __init__(
        self,
        grid_size: Tuple[int, int],
        cell_size: int = 30,
        render_mode: str = "human"
    ):
        """
        Args:
            grid_size: field size (width, height)
            cell_size: cell size in pixels
            render_mode: "human" or "rgb_array"
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode

        # Window dimensions
        self.window_width = grid_size[0] * cell_size
        self.window_height = grid_size[1] * cell_size + 50  # +50 for info panel

        pygame.init()
        pygame.display.set_caption("Snake+ RL")

        if render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        else:
            self.screen = pygame.Surface(
                (self.window_width, self.window_height)
            )

        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

    def render(
        self,
        snake: Snake,
        objects: List[GameObject],
        obstacles: List[GameObject],
        score: int,
        steps: int
    ) -> Optional[np.ndarray]:
        """
        Renders current game state.

        Returns:
            RGB array if render_mode == "rgb_array", otherwise None
        """
        # Clear screen
        self.screen.fill(self.COLORS["background"])

        # Draw grid
        self._draw_grid()

        # Draw obstacles
        for obs in obstacles:
            self._draw_cell(obs.x, obs.y, self.COLORS["obstacle"])

        # Draw objects
        for obj in objects:
            self._draw_object(obj)

        # Draw snake
        self._draw_snake(snake)

        # Draw info panel
        self._draw_info(score, steps, snake.length)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(10)  # 10 FPS
            return None
        else:
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                (1, 0, 2)
            )

    def _draw_grid(self):
        """Draws grid."""
        for x in range(self.grid_size[0] + 1):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size[1] * self.cell_size)
            )

        for y in range(self.grid_size[1] + 1):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (0, y * self.cell_size),
                (self.window_width, y * self.cell_size)
            )

    def _draw_cell(self, x: int, y: int, color: Tuple[int, int, int],
                   margin: int = 2):
        """Draws a filled cell."""
        rect = pygame.Rect(
            x * self.cell_size + margin,
            y * self.cell_size + margin,
            self.cell_size - 2 * margin,
            self.cell_size - 2 * margin
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=5)

    def _draw_snake(self, snake: Snake):
        """Draws snake."""
        # Body
        for i, (x, y) in enumerate(snake.body):
            if i == 0:
                # Head
                self._draw_cell(x, y, self.COLORS["snake_head"])
                # Eyes
                self._draw_eyes(x, y, snake.direction)
            else:
                # Body with gradient
                ratio = i / len(snake.body)
                color = self._interpolate_color(
                    self.COLORS["snake_body"],
                    (0, 100, 50),
                    ratio
                )
                self._draw_cell(x, y, color)

    def _draw_eyes(self, x: int, y: int, direction):
        """Draws snake eyes."""
        cx = x * self.cell_size + self.cell_size // 2
        cy = y * self.cell_size + self.cell_size // 2

        eye_offsets = {
            Direction.UP: [(-5, -3), (5, -3)],
            Direction.DOWN: [(-5, 3), (5, 3)],
            Direction.LEFT: [(-3, -5), (-3, 5)],
            Direction.RIGHT: [(3, -5), (3, 5)],
        }

        for ox, oy in eye_offsets[direction]:
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (cx + ox, cy + oy),
                3
            )
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (cx + ox, cy + oy),
                1
            )

    def _draw_object(self, obj: GameObject):
        """Draws a game object."""
        x, y = obj.x, obj.y

        color_map = {
            ObjectType.APPLE: self.COLORS["apple"],
            ObjectType.GOLDEN: self.COLORS["golden"],
            ObjectType.POISON: self.COLORS["poison"],
            ObjectType.SOUR: self.COLORS["sour"],
            ObjectType.ROTTEN: self.COLORS["rotten"],
        }

        color = color_map.get(obj.object_type, (255, 255, 255))

        # Draw circle for apples
        cx = x * self.cell_size + self.cell_size // 2
        cy = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 4

        pygame.draw.circle(self.screen, color, (cx, cy), radius)

        # Extra details for poison (skull)
        if obj.object_type == ObjectType.POISON:
            # Simple skull
            pygame.draw.circle(
                self.screen,
                self.COLORS["poison_skull"],
                (cx, cy - 2),
                radius // 2
            )
            # Eyes
            pygame.draw.circle(self.screen, (0, 0, 0), (cx - 3, cy - 3), 2)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx + 3, cy - 3), 2)

        # Shine for golden
        if obj.object_type == ObjectType.GOLDEN:
            pygame.draw.circle(
                self.screen,
                (255, 255, 200),
                (cx - 3, cy - 3),
                3
            )

    def _draw_info(self, score: int, steps: int, length: int):
        """Draws info panel."""
        y = self.grid_size[1] * self.cell_size + 10

        texts = [
            f"Score: {score}",
            f"Steps: {steps}",
            f"Length: {length}",
        ]

        x_positions = [10, 150, 290]

        for text, x in zip(texts, x_positions):
            surface = self.font.render(text, True, self.COLORS["text"])
            self.screen.blit(surface, (x, y))

    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        ratio: float
    ) -> Tuple[int, int, int]:
        """Interpolates between two colors."""
        return tuple(
            int(c1 + (c2 - c1) * ratio)
            for c1, c2 in zip(color1, color2)
        )

    def close(self):
        """Closes pygame."""
        pygame.quit()
