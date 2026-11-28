"""Physical Unitree A1 MuJoCo simulation."""
import logging
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig

from dog_sim.expert import GOAL_RADIUS, JOINTS, distance_to_goal, goal_in_body_frame, has_fallen

logger = logging.getLogger(__name__)
SCENE = Path(__file__).resolve().parent / "scene.xml"
GOAL_FEATURES = ("goal.forward_m", "goal.left_m")


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
    camera: bool = True
    legacy_observations: bool = False
    start_x: float = 0.0
    start_y: float = 0.0
    goal_x: float = 1.0
    goal_y: float = 0.0
    report_goal_result: bool = False


class DogSim(Robot):
    """Free-floating, contact-driven Unitree A1 controlled at its 12 joints."""

    config_class = DogSimConfig
    name = "dog_sim"

    def __init__(self, config: DogSimConfig):
        super().__init__(config)
        self.config = config
        self.connected = False
        self.goal = np.array([config.goal_x, config.goal_y])
        self.cameras = {}
        if config.camera:
            self.cameras = {"front": "front"} if config.legacy_observations else {
                "front": "onboard", "overhead": "overhead"
            }

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
        goal_features = {name: float for name in GOAL_FEATURES} if self.config.legacy_observations else {}
        return {
            **self.action_features,
            **goal_features,
            **{name: (self.config.height, self.config.width, 3) for name in self.cameras},
        }

    def connect(self, calibrate=True):
        spec = mujoco.MjSpec.from_file(self.config.scene_path)
        if self.cameras and not self.config.legacy_observations:
            spec.worldbody.add_camera(
                name="overhead",
                pos=[self.config.start_x, self.config.start_y, 7.0],
                fovy=55.0,
            )
            spec.body("trunk").add_camera(
                name="onboard",
                pos=[0.275, 0.0, 0.065],
                xyaxes=[0.0, -1.0, 0.0, 0.34, 0.0, 0.94],
                fovy=60.0,
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.idx = np.array([self.model.joint(f"{joint}_joint").qposadr[0] for joint in JOINTS])
        self.actuators = np.array([self.model.actuator(name).id for name in JOINTS])
        trunk = self.model.body("trunk")
        self.root_qpos = self.model.jnt_qposadr[trunk.jntadr[0]]
        self.goal_mocap = self.model.body("goal_marker").mocapid[0]
        self.steps_per_action = max(1, round(1 / self.config.fps / self.model.opt.timestep))
        self.model.opt.timestep = 1 / self.config.fps / self.steps_per_action
        self.reset()
        self.renderer = None
        if self.cameras:
            self.renderer = mujoco.Renderer(
                self.model, height=self.config.height, width=self.config.width
            )
            if self.config.legacy_observations:
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                camera.trackbodyid = trunk.id
                camera.distance = 2.1
                camera.azimuth = 120
                camera.elevation = -22
                self.cameras["front"] = camera
        self.connected = True

    def reset(self, goal=None, *, start=None, yaw=None):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        root = self.data.qpos[self.root_qpos : self.root_qpos + 7]
        if start is None:
            start = [self.config.start_x, self.config.start_y]
        root[:2] = start
        if yaw is not None:
            root[3:] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        if goal is None:
            goal = [self.config.goal_x, self.config.goal_y]
        self.goal = np.asarray(goal, dtype=float).copy()
        self.data.mocap_pos[self.goal_mocap, :2] = self.goal
        mujoco.mj_forward(self.model, self.data)
        if self.config.report_goal_result:
            self.initial_goal_distance = distance_to_goal(self.data, self.goal)
            self.closest_goal_distance = self.initial_goal_distance
            self.fell_during_rollout = False

    def get_observation(self) -> RobotObservation:
        observation = q_to_dict(native_to_scaled(self.model, self.data.qpos[self.idx]))
        if self.config.legacy_observations:
            goal_local = goal_in_body_frame(self.data, self.goal)
            observation.update(zip(GOAL_FEATURES, map(float, goal_local), strict=True))
        for name, camera in self.cameras.items():
            self.renderer.update_scene(self.data, camera=camera)
            observation[name] = self.renderer.render()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        if action.get("reset"):
            self.reset()
        target = np.asarray([action[f"{joint}.pos"] for joint in JOINTS])
        target = np.clip(target, -100, 100)
        self.data.ctrl[self.actuators] = scaled_to_native(self.model, target)
        mujoco.mj_step(self.model, self.data, nstep=self.steps_per_action)
        mujoco.mj_forward(self.model, self.data)
        if self.config.report_goal_result:
            self.closest_goal_distance = min(self.closest_goal_distance, distance_to_goal(self.data, self.goal))
            self.fell_during_rollout |= has_fallen(self.data)
        return q_to_dict(target)

    def disconnect(self):
        if self.connected:
            if self.renderer is not None:
                self.renderer.close()
            self.connected = False
            if self.config.report_goal_result:
                reached = self.closest_goal_distance <= GOAL_RADIUS and not self.fell_during_rollout
                logger.info(
                    "Goal result: %s | A-to-B: %.3f m | closest: %.3f m | final: %.3f m | fall: %s",
                    "PASS" if reached else "MISS",
                    self.initial_goal_distance,
                    self.closest_goal_distance,
                    distance_to_goal(self.data, self.goal),
                    "yes" if self.fell_during_rollout else "no",
                )
