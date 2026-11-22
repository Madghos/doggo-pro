from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from dog_sim.env import DogGoEnv


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

N_ENVS = 4
TOTAL_TIMESTEPS = 5_000_000


def main():
    env = make_vec_env(
        DogGoEnv,
        n_envs=N_ENVS,
        seed=42,
    )

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path=str(CHECKPOINTS),
        name_prefix="a1_goal",
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": torch.nn.Tanh,
        },
        verbose=1,
        tensorboard_log=str(ROOT / "runs"),
        device="auto",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            tb_log_name="a1_goal_ppo",
        )
    finally:
        model.save(str(CHECKPOINTS / "a1_goal_ppo"))
        env.save(str(CHECKPOINTS / "vecnormalize.pkl"))
        env.close()

    print("Trening zakończony. Model zapisany w checkpoints/.")


if __name__ == "__main__":
    main()
