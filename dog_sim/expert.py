"""Feedback-controlled trot for the physical Unitree A1 in MuJoCo."""

import mujoco
import numpy as np


LEGS = ("FR", "FL", "RR", "RL")
JOINTS = [f"{leg}_{joint}" for leg in LEGS for joint in ("hip", "thigh", "calf")]
GOAL_RADIUS = 0.18
SETTLE_TIME = 0.4


class Expert:
    def __init__(self, model: mujoco.MjModel):
        home = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, home, model.key("home").id)
        mujoco.mj_forward(model, home)
        actuators = [model.actuator(name).id for name in JOINTS]
        self.home = home.ctrl[actuators].copy()
        root_joint = model.body("trunk").jntadr[0]
        self.root_dof = model.jnt_dofadr[root_joint]

        self.hips = np.array([model.body(f"{leg}_hip").pos for leg in LEGS])
        self.hip_offsets = np.array([model.body(f"{leg}_thigh").pos[1] for leg in LEGS])
        self.thigh_length = abs(model.body("FR_calf").pos[2])
        self.calf_length = abs(self._foot_geom(model, "FR").pos[2])
        foot_ids = [self._foot_geom(model, leg).id for leg in LEGS]
        self.feet = home.geom_xpos[foot_ids] - home.body("trunk").xpos
        self.limits = model.actuator_ctrlrange[actuators].T

    @staticmethod
    def _foot_geom(model, leg):
        calf = model.body(f"{leg}_calf")
        return next(
            model.geom(i)
            for i in range(calf.geomadr[0], calf.geomadr[0] + calf.geomnum[0])
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        )

    def _joint_targets(self, feet):
        x, y, z = (feet - self.hips).T
        offsets = self.hip_offsets
        z_sagittal = -np.sqrt(np.maximum(y**2 + z**2 - offsets**2, 1e-6))
        hip = np.arctan2(y, -z) - np.arctan2(offsets, -z_sagittal)

        thigh_length, calf_length = self.thigh_length, self.calf_length
        knee_cos = (
            x**2 + z_sagittal**2 - thigh_length**2 - calf_length**2
        ) / (2 * thigh_length * calf_length)
        knee = -np.arccos(np.clip(knee_cos, -1.0, 1.0))
        thigh = np.arctan2(-x, -z_sagittal) - np.arctan2(
            calf_length * np.sin(knee), thigh_length + calf_length * np.cos(knee)
        )
        return np.clip(np.column_stack((hip, thigh, knee)).ravel(), *self.limits)

    def action(self, data: mujoco.MjData, goal: np.ndarray) -> np.ndarray:
        """Return the 12 joint position targets in radians, ordered by JOINTS."""
        local_goal = goal_in_body_frame(data, goal)
        distance = np.linalg.norm(local_goal)
        if distance <= GOAL_RADIUS * 0.7:
            return self.home.copy()

        heading = np.arctan2(local_goal[1], local_goal[0])
        speed = np.clip(0.9 * distance, 0.12, 0.55)
        forward = speed * max(0.0, np.cos(heading))
        lateral = 0.35 * speed * np.sin(heading)
        turn = np.clip(2.0 * heading, -1.35, 1.35)

        # In stance, push back along the desired body twist. In swing, return
        # each foot to the front of its step while clearing the ground.
        foot_velocity = np.column_stack((
            forward - turn * self.feet[:, 1],
            lateral + turn * self.feet[:, 0],
        ))
        period = 0.36
        stance = 0.57
        phase = (data.time / period + np.array([0.0, 0.5, 0.5, 0.0])) % 1.0
        swing = np.clip((phase - stance) / (1.0 - stance), 0.0, 1.0)
        stride = np.where(
            phase < stance, 0.5 - phase / stance, -0.5 * np.cos(np.pi * swing)
        )

        feet = self.feet.copy()
        feet[:, :2] += period * stance * stride[:, None] * foot_velocity
        feet[:, 2] += 0.065 * np.sin(np.pi * swing)**2
        return self._joint_targets(feet)

    def at_rest_near_goal(self, data: mujoco.MjData, goal: np.ndarray) -> bool:
        speed = np.linalg.norm(data.qvel[self.root_dof : self.root_dof + 2])
        return bool(distance_to_goal(data, goal) <= GOAL_RADIUS and speed <= 0.10)


def distance_to_goal(data: mujoco.MjData, goal: np.ndarray) -> float:
    return float(np.linalg.norm(data.body("trunk").xpos[:2] - goal))


def goal_in_body_frame(data: mujoco.MjData, goal: np.ndarray) -> np.ndarray:
    trunk = data.body("trunk")
    x, y = goal - trunk.xpos[:2]
    yaw = np.arctan2(trunk.xmat[3], trunk.xmat[0])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return np.array([cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y])


def has_fallen(data: mujoco.MjData) -> bool:
    trunk = data.body("trunk")
    return bool(trunk.xpos[2] < 0.16 or trunk.xmat[8] < 0.45)
