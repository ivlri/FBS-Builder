#--------- Training / Validation---------
if __name__ == "__main__":
    from src.builder.fbs_builder import FBSBuilderEnv, EpisodeRewardCallback
    from src.builder.structures import  WallInstance

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env import VecNormalize
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse

    grid_step = 20

    MIN_LENGTH = 1200
    MAX_LENGTH = 6000
    MIN_HEIGHT = 1200
    MAX_HEIGHT = 3000

    def mask_fn(env):
        while hasattr(env, 'env'):
            if hasattr(env, 'get_action_mask'):
                return env.get_action_mask()
            env = env.env
        return env.get_action_mask()

    def make_env():
        env = FBSBuilderEnv(
            randomize=True,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            min_height=MIN_HEIGHT,
            max_height=MAX_HEIGHT,
            grid_step=grid_step,
            render_mode=None,
            max_steps=500,
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env

    def make_vec_env(n_envs=4):
        vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=50.0,
            gamma=0.99,
            norm_obs_keys=["grid", "blocked_mask"],
        )
        return vec_env

    def model_train(total_timesteps=200_000, vec_normalize=True):
        if vec_normalize:
            env = make_vec_env(n_envs=4)
        else:
            env = make_env()

        model = MaskablePPO(
            policy=MaskableMultiInputActorCriticPolicy,
            env=env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=512,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.1,
            clip_range=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=3,
            target_kl=0.01,
            tensorboard_log="./fbs_tensorboard/",
            device='cuda',
        )

        callback = EpisodeRewardCallback(verbose=1)
        model.learn(callback=callback, total_timesteps=total_timesteps)
        model.save("src/builder/data/ppo_fbs_builder")

        if vec_normalize:
            env.save("src/builder/data/vec_normalize.pkl")

        if callback.episode_rewards:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(callback.episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("PPO Stage 3 — Rewards")
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.plot(callback.episode_lengths)
            plt.xlabel("Episode")
            plt.ylabel("Episode Length")
            plt.title("PPO Stage 3 — Length")
            plt.grid()

            plt.tight_layout()
            plt.savefig("src/builder/data/training_progress.png")
            plt.show()

        return model

    def validation(model_path="src/builder/data/ppo_fbs_builder"):
        wall = WallInstance(id=0, length=3000, height=1800, weight=300, grid_step=grid_step)
        env = FBSBuilderEnv(
            wall_instance=wall,
            randomize=False,
            max_length=MAX_LENGTH,
            max_height=MAX_HEIGHT,
            render_mode="terminal_human",
            max_steps=500,
        )

        model = MaskablePPO.load(model_path)

        obs, _ = env.reset()
        done = False

        while not done:
            action_masks = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated

        print(f"\nEpisode finished: {info}")
        print(f"Total reward: {env.total_reward:.1f}")

    def manual_testing():
        """Random policy on randomized walls — quick sanity check."""
        env = FBSBuilderEnv(
            randomize=True,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            min_height=MIN_HEIGHT,
            max_height=MAX_HEIGHT,
            grid_step=grid_step,
            render_mode="terminal_human",
            max_steps=500,
        )

        for episode in range(3):
            obs, _ = env.reset()
            wall = env.num_cells * env.grid_step
            height = env.num_rows * 300
            print(f"\n{'='*60}")
            print(f"Episode {episode+1}: {wall}mm x {height}mm "
                  f"({env.num_cells}c x {env.num_rows}R / {env.num_layers}L)")
            print(f"{'='*60}")

            while True:
                mask = obs["action_mask"]
                legal_moves = np.where(mask == 1)[0]

                if len(legal_moves) == 0:
                    print("No legal moves remain.")
                    break

                action = np.random.choice(legal_moves)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                print(f"Step reward: {reward:.2f}")

                if terminated or truncated:
                    print(f"Episode finished: {info}")
                    break

            print(f"Total reward: {env.total_reward:.1f}")

    #Main 
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--mode", '-m',default='v', help="start mode (t or v)")
    args = parser.parse_args()

    if args.mode == 'v':
        print("\nStarting validation...")
        validation()
    else:
        print("Starting training (openings + randomization)...")
        model = model_train(total_timesteps=200000)