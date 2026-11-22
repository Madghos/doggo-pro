"""Scripted guided movement to the green disk. Not learned or physical legged walking."""
import numpy as np
from dog_sim.robot import JOINTS, native_to_scaled


def is_success(model, data):
    dog = data.body('dog').xpos[:2]
    target = model.geom('target').pos[:2]
    return float(np.linalg.norm(dog - target)) < float(model.geom('target').size[0])


class Expert:
    def __init__(self, model, fps):
        self.model, self.fps = model, fps

    def plan(self, data):
        target = self.model.geom('target').pos[:2]
        # Local slide coordinates are offsets from the initial dog base at (-.55,0).
        dx = target[0] + .55
        dy = target[1]
        # ACT sees 11 actuator targets, compatible with the manual recorder.
        start = np.zeros(len(JOINTS))
        current = start.copy()
        frames = []
        duration = 8.0
        count = round(duration * self.fps)
        for i in range(count):
            f = (i + 1) / count
            smooth = 10 * f**3 - 15 * f**4 + 6 * f**5
            phase = 2 * np.pi * 2 * f * duration
            native = np.zeros(len(JOINTS))
            native[0] = dx * smooth
            native[1] = dy * smooth
            swing = .24 * np.sin(phase) * np.sin(np.pi * f)**2
            bend = -.14 * (1 - np.cos(phase)) * np.sin(np.pi * f)**2
            native[3:] = [swing, bend, -swing, bend, -swing, bend, swing, bend]
            frames.append(native_to_scaled(self.model, native))
        last = frames[-1].copy()
        frames.extend(last.copy() for _ in range(round(self.fps * 1.0)))
        return np.asarray(frames)
