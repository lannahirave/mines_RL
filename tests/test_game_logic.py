"""
Unit tests for game logic (game_objects and snake).
"""

import pytest
import numpy as np
from env.game_objects import ObjectType, GameObject, ObjectFactory, RewardCalculator
from env.snake import Snake, Direction, Action


class TestGameObject:
    def test_position_property(self):
        obj = GameObject(x=5, y=10, object_type=ObjectType.APPLE)
        assert obj.position == (5, 10)

    def test_default_lifetime(self):
        obj = GameObject(x=0, y=0, object_type=ObjectType.APPLE)
        assert obj.lifetime == -1

    def test_custom_lifetime(self):
        obj = GameObject(x=0, y=0, object_type=ObjectType.OBSTACLE, lifetime=50)
        assert obj.lifetime == 50


class TestObjectFactory:
    def setup_method(self):
        self.spawn_probs = {
            "apple": 0.50,
            "golden": 0.10,
            "poison": 0.15,
            "sour": 0.15,
            "rotten": 0.10,
        }
        self.factory = ObjectFactory(self.spawn_probs, grid_size=(15, 15))

    def test_create_object_in_free_position(self):
        obj = self.factory.create_random_object(occupied_positions=set())
        assert obj is not None
        assert 0 <= obj.x < 15
        assert 0 <= obj.y < 15
        assert obj.object_type in [
            ObjectType.APPLE, ObjectType.GOLDEN, ObjectType.POISON,
            ObjectType.SOUR, ObjectType.ROTTEN,
        ]

    def test_create_object_no_space(self):
        occupied = {(x, y) for x in range(15) for y in range(15)}
        obj = self.factory.create_random_object(occupied)
        assert obj is None

    def test_object_not_in_occupied(self):
        occupied = {(0, 0), (1, 1), (2, 2)}
        obj = self.factory.create_random_object(occupied)
        assert obj is not None
        assert obj.position not in occupied

    def test_normalized_probs_sum_to_one(self):
        total = sum(self.factory.normalized_probs.values())
        assert abs(total - 1.0) < 1e-6


class TestRewardCalculator:
    def test_apple_reward(self):
        reward = RewardCalculator.get_reward(ObjectType.APPLE)
        assert reward == 10

    def test_golden_reward_range(self):
        rewards = [RewardCalculator.get_reward(ObjectType.GOLDEN) for _ in range(100)]
        assert all(30 <= r <= 70 for r in rewards)

    def test_poison_reward(self):
        assert RewardCalculator.get_reward(ObjectType.POISON) == -1000

    def test_sour_reward(self):
        assert RewardCalculator.get_reward(ObjectType.SOUR) == -5

    def test_rotten_reward(self):
        assert RewardCalculator.get_reward(ObjectType.ROTTEN) == -20

    def test_length_change_apple(self):
        assert RewardCalculator.get_length_change(ObjectType.APPLE) == 1

    def test_length_change_golden(self):
        assert RewardCalculator.get_length_change(ObjectType.GOLDEN) == 3

    def test_length_change_sour(self):
        change = RewardCalculator.get_length_change(ObjectType.SOUR)
        assert -3 <= change <= -1


class TestDirection:
    def test_direction_values(self):
        assert Direction.UP.value == (0, -1)
        assert Direction.DOWN.value == (0, 1)
        assert Direction.LEFT.value == (-1, 0)
        assert Direction.RIGHT.value == (1, 0)


class TestSnake:
    def test_initial_position(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        assert snake.head == (7, 7)
        assert snake.length == 3
        assert list(snake.body) == [(7, 7), (6, 7), (5, 7)]

    def test_initial_direction(self):
        snake = Snake(start_pos=(7, 7), start_direction=Direction.UP)
        assert snake.direction == Direction.UP

    def test_move_right(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        new_head = snake.move()
        assert new_head == (8, 7)
        assert snake.head == (8, 7)
        assert snake.length == 3

    def test_move_up(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.UP)
        new_head = snake.move()
        assert new_head == (7, 6)

    def test_apply_action_forward(self):
        snake = Snake(start_pos=(7, 7), start_direction=Direction.RIGHT)
        snake.apply_action(Action.FORWARD)
        assert snake.direction == Direction.RIGHT

    def test_apply_action_turn_left(self):
        snake = Snake(start_pos=(7, 7), start_direction=Direction.RIGHT)
        snake.apply_action(Action.TURN_LEFT)
        assert snake.direction == Direction.UP

    def test_apply_action_turn_right(self):
        snake = Snake(start_pos=(7, 7), start_direction=Direction.RIGHT)
        snake.apply_action(Action.TURN_RIGHT)
        assert snake.direction == Direction.DOWN

    def test_grow(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        snake.grow(1)
        snake.move()
        assert snake.length == 4

    def test_shrink(self):
        snake = Snake(start_pos=(7, 7), start_length=5, start_direction=Direction.RIGHT)
        snake.shrink(2)
        assert snake.length == 3

    def test_shrink_minimum_length(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        snake.shrink(10)
        assert snake.length == 1

    def test_detach_tail(self):
        snake = Snake(start_pos=(7, 7), start_length=5, start_direction=Direction.RIGHT)
        detached = snake.detach_tail(2)
        assert len(detached) == 2
        assert snake.length == 3

    def test_detach_tail_max(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        detached = snake.detach_tail(10)
        assert len(detached) == 2  # can't detach more than length - 1
        assert snake.length == 1

    def test_self_collision_false(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        assert not snake.check_self_collision()

    def test_self_collision_true(self):
        # Create a snake that loops back on itself
        snake = Snake(start_pos=(7, 7), start_length=5, start_direction=Direction.RIGHT)
        # Manually create a self-collision scenario
        snake.body = deque([(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)])
        assert snake.check_self_collision()

    def test_get_body_set(self):
        snake = Snake(start_pos=(7, 7), start_length=3, start_direction=Direction.RIGHT)
        body_set = snake.get_body_set()
        assert body_set == {(7, 7), (6, 7), (5, 7)}

    def test_full_turn_cycle(self):
        """Test that 4 left turns return to original direction."""
        snake = Snake(start_pos=(7, 7), start_direction=Direction.RIGHT)
        for _ in range(4):
            snake.apply_action(Action.TURN_LEFT)
        assert snake.direction == Direction.RIGHT


# Need deque import for self collision test
from collections import deque
