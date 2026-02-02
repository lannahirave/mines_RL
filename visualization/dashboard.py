"""
Interactive Pygame dashboard for Snake+ RL.

Two modes:
- Observe: watch a trained agent play
- Play: control the snake manually

Usage:
    python -m visualization.dashboard --model results/runs/<timestamp>/model_final.pt
    python -m visualization.dashboard --mode play
    python -m visualization.dashboard --model model.pt --config configs/training.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

import pygame
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from env.snake_env import SnakePlusEnv
from env.snake import Direction, Action
from env.game_objects import ObjectType
from agent.dqn_agent import DQNAgent


# Dashboard colors (extend renderer palette)
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
    "panel_bg": (20, 20, 30),
    "panel_border": (60, 60, 80),
    "button": (50, 120, 200),
    "button_hover": (70, 140, 220),
    "button_text": (255, 255, 255),
    "mode_observe": (0, 180, 100),
    "mode_play": (200, 100, 0),
    "paused": (200, 50, 50),
}


class Dashboard:
    """
    Interactive Pygame dashboard for observing agents or playing manually.

    Args:
        config: environment configuration dict
        model_path: path to trained model checkpoint (None for play mode)
        mode: initial mode ("observe" or "play")
    """

    PANEL_WIDTH = 220
    CELL_SIZE = 30
    FPS_OPTIONS = [5, 10, 15, 20, 30, 60]

    def __init__(
        self,
        config: dict,
        model_path: str = None,
        mode: str = "observe",
    ):
        self.config = config
        self.model_path = model_path
        self.mode = mode

        # Grid dimensions
        self.grid_size = tuple(config["env"]["grid_size"])
        self.grid_w = self.grid_size[0] * self.CELL_SIZE
        self.grid_h = self.grid_size[1] * self.CELL_SIZE

        # Window dimensions
        self.window_w = self.grid_w + self.PANEL_WIDTH
        self.window_h = self.grid_h

        # Game state
        self.paused = False
        self.fps_index = 1  # default 10 FPS
        self.episode_count = 0
        self.best_score = 0
        self.total_score = 0
        self.scores_history: list = []

        # Agent (loaded later if needed)
        self.agent: DQNAgent = None

        # Pending manual action
        self._pending_action: int = None

        # Create environment
        self.env = SnakePlusEnv(
            grid_size=self.grid_size,
            spawn_probs=config["env"]["spawn_probs"],
            max_objects=config["env"]["max_objects"],
            obstacle_decay=config["env"].get("obstacle_decay"),
            max_steps=config["env"]["max_steps"],
            observation_type=config["env"]["observation_type"],
        )

        # Load agent if in observe mode
        if model_path and mode == "observe":
            self._load_agent()

    def _load_agent(self) -> None:
        """Loads a trained DQN agent from checkpoint."""
        self.agent = DQNAgent(
            observation_type=self.config["env"]["observation_type"],
            n_actions=3,
            learning_rate=self.config["agent"]["learning_rate"],
            discount_factor=self.config["agent"]["discount_factor"],
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            buffer_size=1000,
            batch_size=64,
            target_update_freq=1000,
            use_double_dqn=self.config["agent"]["use_double_dqn"],
            use_dueling=self.config["agent"]["use_dueling"],
            use_prioritized_replay=self.config["agent"]["use_prioritized_replay"],
        )
        self.agent.load(self.model_path)
        self.agent.epsilon = 0.0  # greedy
        print(f"Loaded model from {self.model_path}")

    def run(self) -> None:
        """Main dashboard loop."""
        pygame.init()
        pygame.display.set_caption("Snake+ RL Dashboard")
        screen = pygame.display.set_mode((self.window_w, self.window_h))
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 22)
        font_large = pygame.font.Font(None, 28)

        # Start first episode
        state, info = self.env.reset()
        done = False
        current_score = 0
        current_steps = 0

        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    running, state, info, done, current_score, current_steps = (
                        self._handle_keydown(
                            event, state, info, done,
                            current_score, current_steps
                        )
                    )

            # Step the game (if not paused and not done)
            if not self.paused and not done:
                action = self._get_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                current_score = info["score"]
                current_steps = info["steps"]

                if done:
                    self._end_episode(current_score)

            # Draw
            screen.fill(COLORS["background"])
            self._draw_grid(screen)
            self._draw_game(screen)
            self._draw_panel(screen, font, font_large, current_score, current_steps, done)

            pygame.display.flip()
            clock.tick(self.FPS_OPTIONS[self.fps_index])

        pygame.quit()

    def _handle_keydown(self, event, state, info, done, current_score, current_steps):
        """Handles keyboard input. Returns updated state tuple."""
        running = True

        if event.key == pygame.K_ESCAPE:
            running = False

        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused

        elif event.key == pygame.K_r:
            # Reset episode
            state, info = self.env.reset()
            done = False
            current_score = 0
            current_steps = 0

        elif event.key == pygame.K_m:
            # Toggle mode
            if self.mode == "observe" and self.agent is not None:
                self.mode = "play"
            elif self.mode == "play" and self.agent is not None:
                self.mode = "observe"

        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
            self.fps_index = min(self.fps_index + 1, len(self.FPS_OPTIONS) - 1)

        elif event.key == pygame.K_MINUS:
            self.fps_index = max(self.fps_index - 1, 0)

        # Manual controls (play mode)
        elif self.mode == "play":
            if event.key in (pygame.K_UP, pygame.K_w):
                self._pending_action = self._direction_to_action(Direction.UP)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._pending_action = self._direction_to_action(Direction.DOWN)
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self._pending_action = self._direction_to_action(Direction.LEFT)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self._pending_action = self._direction_to_action(Direction.RIGHT)

        return running, state, info, done, current_score, current_steps

    def _direction_to_action(self, target_dir: Direction) -> int:
        """Converts an absolute direction to a relative action."""
        if self.env.snake is None:
            return Action.FORWARD.value

        current = self.env.snake.direction
        if target_dir == current:
            return Action.FORWARD.value

        from env.snake import Snake
        if Snake.TURN_LEFT_MAP[current] == target_dir:
            return Action.TURN_LEFT.value
        elif Snake.TURN_RIGHT_MAP[current] == target_dir:
            return Action.TURN_RIGHT.value

        # 180-degree turn: just go forward (can't reverse)
        return Action.FORWARD.value

    def _get_action(self, state) -> int:
        """Gets the next action based on current mode."""
        if self.mode == "observe" and self.agent is not None:
            return self.agent.select_action(state, training=False)
        elif self.mode == "play":
            if self._pending_action is not None:
                action = self._pending_action
                self._pending_action = None
                return action
            return Action.FORWARD.value
        else:
            return Action.FORWARD.value

    def _end_episode(self, score: int) -> None:
        """Updates stats at end of episode."""
        self.episode_count += 1
        self.total_score += score
        self.best_score = max(self.best_score, score)
        self.scores_history.append(score)

    def _draw_grid(self, screen: pygame.Surface) -> None:
        """Draws the game grid lines."""
        for x in range(self.grid_size[0] + 1):
            pygame.draw.line(
                screen, COLORS["grid"],
                (x * self.CELL_SIZE, 0),
                (x * self.CELL_SIZE, self.grid_h)
            )
        for y in range(self.grid_size[1] + 1):
            pygame.draw.line(
                screen, COLORS["grid"],
                (0, y * self.CELL_SIZE),
                (self.grid_w, y * self.CELL_SIZE)
            )

    def _draw_game(self, screen: pygame.Surface) -> None:
        """Draws snake, objects, and obstacles."""
        if self.env.snake is None:
            return

        # Obstacles
        for obs in self.env.obstacles:
            self._draw_cell(screen, obs.x, obs.y, COLORS["obstacle"])

        # Objects
        color_map = {
            ObjectType.APPLE: COLORS["apple"],
            ObjectType.GOLDEN: COLORS["golden"],
            ObjectType.POISON: COLORS["poison"],
            ObjectType.SOUR: COLORS["sour"],
            ObjectType.ROTTEN: COLORS["rotten"],
        }
        for obj in self.env.objects:
            cx = obj.x * self.CELL_SIZE + self.CELL_SIZE // 2
            cy = obj.y * self.CELL_SIZE + self.CELL_SIZE // 2
            radius = self.CELL_SIZE // 2 - 4
            color = color_map.get(obj.object_type, (255, 255, 255))
            pygame.draw.circle(screen, color, (cx, cy), radius)

            if obj.object_type == ObjectType.GOLDEN:
                pygame.draw.circle(screen, (255, 255, 200), (cx - 3, cy - 3), 3)

        # Snake body
        for i, (x, y) in enumerate(self.env.snake.body):
            if i == 0:
                self._draw_cell(screen, x, y, COLORS["snake_head"])
            else:
                ratio = i / max(len(self.env.snake.body), 1)
                color = self._lerp_color(COLORS["snake_body"], (0, 100, 50), ratio)
                self._draw_cell(screen, x, y, color)

    def _draw_cell(self, screen, x, y, color, margin=2):
        """Draws a single grid cell."""
        rect = pygame.Rect(
            x * self.CELL_SIZE + margin,
            y * self.CELL_SIZE + margin,
            self.CELL_SIZE - 2 * margin,
            self.CELL_SIZE - 2 * margin,
        )
        pygame.draw.rect(screen, color, rect, border_radius=5)

    def _draw_panel(self, screen, font, font_large, score, steps, done):
        """Draws the right-side statistics panel."""
        panel_x = self.grid_w
        panel_rect = pygame.Rect(panel_x, 0, self.PANEL_WIDTH, self.window_h)
        pygame.draw.rect(screen, COLORS["panel_bg"], panel_rect)
        pygame.draw.line(
            screen, COLORS["panel_border"],
            (panel_x, 0), (panel_x, self.window_h), 2
        )

        y = 15
        pad = 10

        # Mode indicator
        mode_color = COLORS["mode_observe"] if self.mode == "observe" else COLORS["mode_play"]
        mode_text = f"Mode: {self.mode.upper()}"
        surf = font_large.render(mode_text, True, mode_color)
        screen.blit(surf, (panel_x + pad, y))
        y += 35

        # Paused indicator
        if self.paused:
            surf = font_large.render("PAUSED", True, COLORS["paused"])
            screen.blit(surf, (panel_x + pad, y))
        y += 30

        # Divider
        pygame.draw.line(
            screen, COLORS["panel_border"],
            (panel_x + pad, y), (panel_x + self.PANEL_WIDTH - pad, y)
        )
        y += 15

        # Current episode stats
        lines = [
            ("Score:", str(score)),
            ("Steps:", str(steps)),
            ("Length:", str(self.env.snake.length if self.env.snake else 0)),
            ("Obstacles:", str(len(self.env.obstacles))),
        ]
        for label, val in lines:
            surf = font.render(f"{label} {val}", True, COLORS["text"])
            screen.blit(surf, (panel_x + pad, y))
            y += 22

        if done:
            surf = font.render("GAME OVER", True, COLORS["paused"])
            screen.blit(surf, (panel_x + pad, y))
        y += 30

        # Divider
        pygame.draw.line(
            screen, COLORS["panel_border"],
            (panel_x + pad, y), (panel_x + self.PANEL_WIDTH - pad, y)
        )
        y += 15

        # Session stats
        avg_score = self.total_score / max(self.episode_count, 1)
        session_lines = [
            ("Episodes:", str(self.episode_count)),
            ("Best Score:", str(self.best_score)),
            ("Avg Score:", f"{avg_score:.1f}"),
        ]
        for label, val in session_lines:
            surf = font.render(f"{label} {val}", True, COLORS["text"])
            screen.blit(surf, (panel_x + pad, y))
            y += 22

        y += 15

        # Speed
        fps = self.FPS_OPTIONS[self.fps_index]
        surf = font.render(f"Speed: {fps} FPS", True, COLORS["text"])
        screen.blit(surf, (panel_x + pad, y))
        y += 30

        # Divider
        pygame.draw.line(
            screen, COLORS["panel_border"],
            (panel_x + pad, y), (panel_x + self.PANEL_WIDTH - pad, y)
        )
        y += 15

        # Controls
        controls = [
            "Controls:",
            "SPACE - Pause",
            "+/- - Speed",
            "R - Reset",
            "M - Toggle mode",
            "ESC - Quit",
        ]
        if self.mode == "play":
            controls += [
                "",
                "Arrow keys / WASD",
                "to move snake",
            ]

        for line in controls:
            surf = font.render(line, True, COLORS["text"])
            screen.blit(surf, (panel_x + pad, y))
            y += 20

    @staticmethod
    def _lerp_color(c1, c2, t):
        """Linearly interpolates between two colors."""
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def main():
    parser = argparse.ArgumentParser(description="Snake+ RL Dashboard")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/training.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--mode", type=str, default=None, choices=["observe", "play"],
        help="Initial mode (default: observe if model provided, play otherwise)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mode = args.mode
    if mode is None:
        mode = "observe" if args.model else "play"

    dashboard = Dashboard(config, model_path=args.model, mode=mode)
    dashboard.run()


if __name__ == "__main__":
    main()
