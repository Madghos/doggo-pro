from pathlib import Path

import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dog_sim.env import DogGoEnv


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"


def main():
    base_env = DummyVecEnv([lambda: DogGoEnv()])

    env = VecNormalize.load(
        str(CHECKPOINTS / "vecnormalize.pkl"),
        base_env,
    )
    env.training = False
    env.norm_reward = False

    model = PPO.load(
        str(CHECKPOINTS / "a1_goal_ppo"),
        env=env,
    )

    robot_env = env.venv.envs[0]
    obs = env.reset()

    print("Uruchomiono wytrenowanego Unitree A1.")
    print("Zamknij okno MuJoCo, aby zakończyć.")

    with mujoco.viewer.launch_passive(
        robot_env.model,
        robot_env.data,
    ) as viewer:
        while viewer.is_running():
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, rewards, dones, infos = env.step(action)

            viewer.sync()

            if dones[0]:
                info = infos[0]
                print(
                    f"Episode zakończony. "
                    f"Cel: {info.get('success')}, "
                    f"przewrócenie: {info.get('fallen')}, "
                    f"dystans: {info.get('distance', 0):.2f} m"
                )

    env.close()


if __name__ == "__main__":
    main()
