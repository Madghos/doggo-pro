"""Record pick-and-place episodes with the scripted expert, without a human.

In each frame the loop does what lerobot-record does: it reads the observation,
sends the action, and adds both to the dataset (lerobot/scripts/lerobot_record.py).
It does not wait between frames, because the simulation does not need real time.
After each attempt it tests whether the cube lies on the disk. It saves a
correct episode and discards a failed one, until the dataset has
--dataset.num_episodes episodes.

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

from dog_sim.expert import Expert, is_success
from dog_sim.robot import DogSim, DogSimConfig, q_to_dict

logger = logging.getLogger(__name__)


@dataclass
class RecordAutoConfig:
    # The same --dataset.* flags as lerobot-record. --dataset.fps also sets the
    # control rate of the simulation. episode_time_s and reset_time_s have no effect.
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)
    # Seed of the random cube positions. None gives new positions at every run.
    seed: int | None = None
    # Show the cameras and the joints in Rerun. This makes the run slower.
    display_data: bool = False


@parser.wrap()
def record_auto(cfg: RecordAutoConfig) -> LeRobotDataset:
    init_logging()
    robot = DogSim(DogSimConfig(fps=cfg.dataset.fps, seed=cfg.seed))
    features = {
        **hw_to_dataset_features({**robot.action_features}, ACTION, cfg.dataset.video),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, cfg.dataset.video),
    }

    # The same arguments as in lerobot-record.
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
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
    )

    robot.connect()
    expert = Expert(robot.model, cfg.dataset.fps)
    if cfg.display_data:
        init_visualization("rerun", session_name="recording")

    wanted = cfg.dataset.num_episodes
    attempts = 0
    try:
        # On exit, VideoEncodingManager finishes the videos and calls dataset.finalize().
        with VideoEncodingManager(dataset):
            while dataset.num_episodes < wanted and attempts < 2 * wanted:
                attempts += 1
                robot.reset()
                for target in expert.plan(robot.data):
                    observation = robot.get_observation()
                    action = q_to_dict(target)
                    robot.send_action(action)
                    frame = {
                        **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
                        **build_dataset_frame(dataset.features, action, prefix=ACTION),
                        "task": cfg.dataset.single_task,
                    }
                    dataset.add_frame(frame)
                    if cfg.display_data:
                        log_visualization_data("rerun", observation=observation, action=action)

                distance = float(np.linalg.norm(robot.data.body("dog").xpos[:2] - robot.model.geom("target").pos[:2])) * 100
                if is_success(robot.model, robot.data):
                    dataset.save_episode()
                    logger.info(
                        f"Attempt {attempts}: saved episode {dataset.num_episodes} of {wanted}. "
                        f"The dog is {distance:.1f} cm from the disk center."
                    )
                else:
                    dataset.clear_episode_buffer()
                    logger.warning(
                        f"Attempt {attempts}: discarded. The dog is {distance:.1f} cm from the disk center."
                    )
    finally:
        robot.disconnect()
        if cfg.display_data:
            shutdown_visualization("rerun")

    if dataset.num_episodes < wanted:
        logger.warning(f"Only {dataset.num_episodes} of {wanted} episodes after {attempts} attempts.")
    if cfg.dataset.push_to_hub and dataset.num_episodes > 0:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
    return dataset


def main():
    record_auto()  # ty: ignore[missing-argument]  (parser.wrap reads cfg from the command line)


if __name__ == "__main__":
    main()
