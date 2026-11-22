"""Physical Unitree A1 MuJoCo simulation."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mujoco
import numpy as np
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig

# Actuator names from the official Unitree A1 MJCF model.
JOINTS = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
SCENE = Path(__file__).resolve().parent / "scene.xml"


def limits(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.actuator(name).ctrlrange for name in JOINTS])


def scaled_to_native(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    lo, hi = limits(model).T
    return lo + (np.asarray(q) + 100) / 200 * (hi - lo)


def native_to_scaled(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    lo, hi = limits(model).T
    return 200 * (np.asarray(q) - lo) / (hi - lo) - 100


def q_to_dict(q: np.ndarray) -> dict[str, float]:
    return {f"{joint}.pos": float(value) for joint, value in zip(JOINTS, q, strict=True)}


@RobotConfig.register_subclass("dog_sim")
@dataclass
class DogSimConfig(RobotConfig):
    scene_path: str = str(SCENE)
    width: int = 256
    height: int = 256
    fps: int = 50


class DogSim(Robot):
    """Free-floating, contact-driven Unitree A1 controlled at its 12 joints."""

    config_class = DogSimConfig
    name = "dog_sim"
    cameras: ClassVar[dict[str, str]] = {"front": "front"}

    def __init__(self, config: DogSimConfig):
        super().__init__(config)
        self.config = config
        self.connected = False

    @property
    def is_connected(self):
        return self.connected

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    @property
    def action_features(self):
        return {f"{joint}.pos": float for joint in JOINTS}

    @property
    def observation_features(self):
        return {
            **self.action_features,
            **{name: (self.config.height, self.config.width, 3) for name in self.cameras},
        }

    def connect(self, calibrate=True):
        self.model = mujoco.MjModel.from_xml_path(self.config.scene_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.config.height, width=self.config.width)
        self.idx = np.array([self.model.joint(f"{joint}_joint").qposadr[0] for joint in JOINTS])
        self.steps_per_action = max(1, round(1 / self.config.fps / self.model.opt.timestep))
        self.model.opt.timestep = 1 / self.config.fps / self.steps_per_action
        self.connected = True
        self.reset()

    def reset(self):
        # The original model's keyframe holds a stable standing posture.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def get_observation(self) -> RobotObservation:
        observation = q_to_dict(native_to_scaled(self.model, self.data.qpos[self.idx]))
        for name, camera in self.cameras.items():
            self.renderer.update_scene(self.data, camera=camera)
            observation[name] = self.renderer.render()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        if action.get("reset"):
            self.reset()
        target = np.asarray([action[f"{joint}.pos"] for joint in JOINTS])
        self.data.ctrl[:] = scaled_to_native(self.model, np.clip(target, -100, 100))
        for _ in range(self.steps_per_action):
            mujoco.mj_step(self.model, self.data)
        return q_to_dict(target)

    def disconnect(self):
        if self.connected:
            self.renderer.close()
            self.connected = False
