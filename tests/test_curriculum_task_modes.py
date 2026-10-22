"""A curriculum stage switches the env's task modes, e.g. from random obstacles to group scenarios at a fixed tick."""

from __future__ import annotations

import threading

import pydantic
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosnav_rl.utils.stable_baselines3.tick_callbacks import TickScheduledCurriculum
from task_generator_msgs.srv import QueueEpisode

from arena_training.arena_rosnav_rl.cfg.arena_cfg.task import StagedCfg, TaskCfg
from arena_training.arena_rosnav_rl.stable_baselines3.eval_callbacks.initialization import curriculum_stage_lists

NODE = "/curriculum_modes_test/env_{i}/task_generator_node"
RANDOM_STAGE = {"task.random.dynamic.n": [1, 3], "task.random.dynamic.models": ["Common/Human/arenian"], "tm_obstacles": "random", "tm_robots": "random"}
SCENARIO_STAGE = {"tm_obstacles": "scenario", "tm_robots": "scenario"}


@pytest.fixture
def queued_modes():
    """A task_generator node answering config/queue_episode, and the (tm_obstacles, tm_robots) of every request carrying a mode."""
    if not rclpy.ok():
        rclpy.init()
    target = Node("task_generator_node", namespace="/curriculum_modes_test/env_0", allow_undeclared_parameters=True)
    trainer = Node("curriculum_modes_test_trainer")
    modes: list[tuple[str, str]] = []

    def queue(request: QueueEpisode.Request, response: QueueEpisode.Response) -> QueueEpisode.Response:
        if request.tm_obstacles or request.tm_robots:
            modes.append((request.tm_obstacles, request.tm_robots))
        response.success = True
        return response

    target.create_service(QueueEpisode, "~/config/queue_episode", queue)
    executor = MultiThreadedExecutor()
    executor.add_node(target)
    executor.add_node(trainer)
    threading.Thread(target=executor.spin, daemon=True).start()
    yield trainer, modes
    executor.shutdown()
    trainer.destroy_node()
    target.destroy_node()


def test_stage_ticks_switch_the_env_from_random_to_scenario(queued_modes):
    trainer, modes = queued_modes
    task = TaskCfg(staged=StagedCfg(curriculum_definition=[RANDOM_STAGE, SCENARIO_STAGE], stage_ticks=[0, 1000]))
    curriculum = TickScheduledCurriculum(
        node=trainer,
        train_stages=curriculum_stage_lists(task),
        stage_ticks=[0, 1000],
        ticks_per_step=1,
        num_envs=1,
        parameter_node_template=NODE,
        timeout=5.0,
    )
    assert modes[-1] == ("random", "random")

    curriculum.advance_curriculum()
    assert modes[-1] == ("scenario", "scenario")


def test_a_task_mode_must_be_set_on_every_stage():
    with pytest.raises(pydantic.ValidationError, match=r"tm_obstacles is set on some curriculum stages but not on stages \[0\]"):
        StagedCfg(curriculum_definition=[{"task.random.dynamic.n": [1, 3]}, {"tm_obstacles": "scenario"}])
