"""Record successful A-to-B walks from the scripted Unitree A1 expert.

Usage:
    python -m dog_sim.record_auto --dataset.repo_id=<hf-user>/<name> [--dataset.* flags of lerobot-record]
"""

import logging
from dataclasses import dataclass, field
import numpy as np

from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

from dog_sim.expert import Expert, GOAL_RADIUS, SETTLE_TIME, distance_to_goal, has_fallen
from dog_sim.robot import DogSim, DogSimConfig, native_to_scaled, q_to_dict

logger = logging.getLogger(__name__)

START_JITTER_RADIUS = 0.5  # meters
ROUTE_DISTANCE_MIN = 0.6
ROUTE_DISTANCE_MAX = 2.5


@dataclass
class RecordAutoConfig:
    dataset: DatasetRecordConfig = field(default_factory=lambda: DatasetRecordConfig(
        single_task="Walk to point B", push_to_hub=False, episode_time_s=20
    ))
    start_x: float = 0.0
    start_y: float = 0.0
    goal_x: float = 1.0
    goal_y: float = 0.0
    # Uniformly sample B inside a disk of this radius around the configured goal.
    goal_jitter: float = 0.0
    # Sample the start position, initial heading, and goal relative to the start.
    diverse_routes: bool = False
    seed: int | None = None
    camera: bool = True
    legacy_observations: bool = False
    display_data: bool = False


def sample_direction(rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(angle), np.sin(angle)])


@parser.wrap()
def record_auto(cfg: RecordAutoConfig) -> LeRobotDataset:
    init_logging()
    start = np.array([cfg.start_x, cfg.start_y])
    center = np.array([cfg.goal_x, cfg.goal_y])
    if not np.isfinite([*start, *center, cfg.goal_jitter]).all() or cfg.goal_jitter < 0:
        raise ValueError("Use finite coordinates and a nonnegative goal_jitter.")
    if not cfg.diverse_routes and np.linalg.norm(center - start) <= GOAL_RADIUS + cfg.goal_jitter:
        raise ValueError("The start must be outside the goal radius for every sampled goal.")
    duration = cfg.dataset.episode_time_s
    if (cfg.dataset.fps < 1 or cfg.dataset.num_episodes < 1
            or not np.isfinite(duration) or duration <= SETTLE_TIME):
        raise ValueError("Use positive fps and num_episodes, and episode_time_s above 0.4.")
    max_frames = int(duration * cfg.dataset.fps)
    rng = np.random.default_rng(cfg.seed)
    robot = DogSim(DogSimConfig(
        fps=cfg.dataset.fps,
        camera=cfg.camera,
        legacy_observations=cfg.legacy_observations,
        start_x=cfg.start_x,
        start_y=cfg.start_y,
        goal_x=cfg.goal_x,
        goal_y=cfg.goal_y,
    ))
    features = {
        **hw_to_dataset_features({**robot.action_features}, ACTION, cfg.dataset.video),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, cfg.dataset.video),
    }

    cfg.dataset.stamp_repo_id()
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        rgb_encoder=cfg.dataset.rgb_encoder,
        depth_encoder=cfg.dataset.depth_encoder,
        encoder_threads=cfg.dataset.encoder_threads,
        streaming_encoding=cfg.dataset.streaming_encoding,
        # An offline simulation can render faster than the encoder consumes frames.
        encoder_queue_maxsize=max(cfg.dataset.encoder_queue_maxsize, max_frames),
    )

    wanted = cfg.dataset.num_episodes
    attempts = 0
    visualization_started = False
    settle_frames = int(np.ceil(SETTLE_TIME * cfg.dataset.fps))
    try:
        with VideoEncodingManager(dataset):
            robot.connect()
            expert = Expert(robot.model)
            if cfg.display_data:
                init_visualization("rerun", session_name="recording")
                visualization_started = True
            while dataset.num_episodes < wanted and attempts < 2 * wanted:
                attempts += 1
                direction = sample_direction(rng)
                if cfg.diverse_routes:
                    spawn = start + START_JITTER_RADIUS * np.sqrt(rng.random()) * direction
                    goal = spawn + sample_direction(rng) * rng.uniform(
                        ROUTE_DISTANCE_MIN, ROUTE_DISTANCE_MAX
                    )
                    robot.reset(goal, start=spawn, yaw=rng.uniform(-np.pi, np.pi))
                else:
                    robot.reset(center + cfg.goal_jitter * np.sqrt(rng.random()) * direction)
                settled = 0
                reason = "time limit"
                for _ in range(max_frames):
                    observation = robot.get_observation()
                    target = expert.action(robot.data, robot.goal)
                    action = robot.send_action(q_to_dict(native_to_scaled(robot.model, target)))
                    frame = {
                        **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                        **build_dataset_frame(dataset.features, action, prefix=ACTION),
                        "task": cfg.dataset.single_task,
                    }
                    dataset.add_frame(frame)
                    if cfg.display_data:
                        log_visualization_data("rerun", observation=observation, action=action)
                    if has_fallen(robot.data):
                        reason = "fall"
                        break
                    settled = settled + 1 if expert.at_rest_near_goal(robot.data, robot.goal) else 0
                    if settled >= settle_frames:
                        break

                distance = distance_to_goal(robot.data, robot.goal)
                if settled >= settle_frames:
                    dataset.save_episode()
                    logger.info("Attempt %d: saved %d/%d, %.2f m from B.",
                                attempts, dataset.num_episodes, wanted, distance)
                else:
                    dataset.clear_episode_buffer()
                    logger.warning("Attempt %d discarded (%s), %.2f m from B.",
                                   attempts, reason, distance)
    finally:
        robot.disconnect()
        if visualization_started:
            shutdown_visualization("rerun")

    if dataset.num_episodes < wanted:
        raise RuntimeError(f"Saved {dataset.num_episodes}/{wanted} episodes after {attempts} attempts.")
    if cfg.dataset.push_to_hub and dataset.num_episodes > 0:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
    return dataset


def main():
    record_auto()  # ty: ignore[missing-argument]  (parser.wrap reads cfg from the command line)


if __name__ == "__main__":
    main()
