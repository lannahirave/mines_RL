"""
Module with the Snake class.
"""

from enum import Enum, auto
from typing import List, Tuple, Optional
from collections import deque


class Direction(Enum):
    """Movement directions."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Action(Enum):
    """Agent actions (relative)."""
    FORWARD = 0     # Move forward
    TURN_LEFT = 1   # Turn left
    TURN_RIGHT = 2  # Turn right


class Snake:
    """Snake class."""

    # Turn mapping: current direction -> new direction on turn
    TURN_LEFT_MAP = {
        Direction.UP: Direction.LEFT,
        Direction.LEFT: Direction.DOWN,
        Direction.DOWN: Direction.RIGHT,
        Direction.RIGHT: Direction.UP,
    }

    TURN_RIGHT_MAP = {
        Direction.UP: Direction.RIGHT,
        Direction.RIGHT: Direction.DOWN,
        Direction.DOWN: Direction.LEFT,
        Direction.LEFT: Direction.UP,
    }

    def __init__(self, start_pos: Tuple[int, int], start_length: int = 3,
                 start_direction: Direction = Direction.RIGHT):
        """
        Args:
            start_pos: initial head position (x, y)
            start_length: initial length
            start_direction: initial direction
        """
        self.direction = start_direction
        self.grow_pending = 0  # how many segments to add

        # Body as deque: [head, ..., tail]
        self.body: deque = deque()

        # Initialize body
        x, y = start_pos
        dx, dy = start_direction.value

        for i in range(start_length):
            self.body.append((x - i * dx, y - i * dy))

    @property
    def head(self) -> Tuple[int, int]:
        """Head position."""
        return self.body[0]

    @property
    def tail(self) -> Tuple[int, int]:
        """Tail position."""
        return self.body[-1]

    @property
    def length(self) -> int:
        """Snake length."""
        return len(self.body)

    def get_body_set(self) -> set:
        """Returns set of body positions (for fast collision checks)."""
        return set(self.body)

    def apply_action(self, action: Action) -> None:
        """Changes direction according to action."""
        if action == Action.TURN_LEFT:
            self.direction = self.TURN_LEFT_MAP[self.direction]
        elif action == Action.TURN_RIGHT:
            self.direction = self.TURN_RIGHT_MAP[self.direction]
        # FORWARD - direction doesn't change

    def move(self) -> Tuple[int, int]:
        """
        Moves the snake one step.

        Returns:
            New head position
        """
        # Calculate new head position
        hx, hy = self.head
        dx, dy = self.direction.value
        new_head = (hx + dx, hy + dy)

        # Add new head
        self.body.appendleft(new_head)

        # Remove tail (unless growing)
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()

        return new_head

    def grow(self, amount: int = 1) -> None:
        """Increases length by amount segments."""
        self.grow_pending += amount

    def shrink(self, amount: int) -> None:
        """
        Decreases length by amount segments.
        Minimum length = 1 (head only).
        """
        for _ in range(min(amount, len(self.body) - 1)):
            self.body.pop()

    def detach_tail(self, amount: int) -> List[Tuple[int, int]]:
        """
        Detaches amount tail segments.

        Returns:
            List of positions of detached segments (will become obstacles)
        """
        detached = []
        for _ in range(min(amount, len(self.body) - 1)):
            pos = self.body.pop()
            detached.append(pos)
        return detached

    def check_self_collision(self) -> bool:
        """Checks if head collided with body."""
        return self.head in list(self.body)[1:]
