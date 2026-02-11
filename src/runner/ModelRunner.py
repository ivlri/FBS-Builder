import os
import numpy as np
from typing import Optional, Dict, Any, List

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.wrappers import ActionMasker

from src.builder.fbs_builder import FBSBuilderEnv
from src.builder.structures import WallInstance, Opening,  GRID_STEP

from src.contextbuilder.contextbuilder import ContextBuilder


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        vec_norm_patch: str = "src/builder/data/vec_normalize.pkl",
        deterministic:bool = True,
        max_steps: int = 500,
        min_length: int = 1200,
        max_length: int = 6000,
        min_height: int = 1200,
        max_height: int = 3000,
    ):
        self.model_path = model_path
        self.vec_norm_path = vec_norm_patch
        self.deterministic = deterministic

        self.max_steps = max_steps
        self.min_length = min_length
        self.max_length = max_length
        self.min_height = min_height
        self.max_height = max_height

        self.model: Optional[MaskablePPO] = None

    @staticmethod
    def _mask_fn(env):
        while hasattr(env, "env"):
            if hasattr(env, "get_action_mask"):
                return env.get_action_mask()
            env = env.env
        return env.get_action_mask()
    
    @staticmethod
    def _get_base_env(vec_env):
        env = vec_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env
    
    def _load_model(self):
        if self.model is None:
            self.model = MaskablePPO.load(self.model_path)

    def _make_env(
            self, 
            wall: WallInstance, 
            openings: Optional[Opening] = None,
            context_builder: Optional[ContextBuilder] = None,
            context_data: dict = None,
    ):
        def make_env():
            env = FBSBuilderEnv(
                wall_instance=wall,
                context_builder=context_builder,
                context_data=context_data,
                openings=openings,
                render_mode=None,
                max_steps=self.max_steps,
                min_length=self.min_length,
                max_length=self.max_length,
                min_height=self.min_height,
                max_height=self.max_height,
                grid_step=GRID_STEP,
            )
            env = ActionMasker(env, ModelRunner._mask_fn)
            return env

        vec_env = DummyVecEnv([make_env])

        # Load VecNormalize
        if os.path.exists(self.vec_norm_path):
            try:
                vec_env = VecNormalize.load(self.vec_norm_path, vec_env)
            except AssertionError:
                vec_env = VecNormalize(
                    vec_env,
                    norm_obs=True,
                    norm_reward=False,
                    norm_obs_keys=["grid", "blocked_mask"],
                )
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
                norm_obs_keys=["grid", "blocked_mask"],
            )

        vec_env.training = False
        vec_env.norm_reward = False

        return vec_env
    
    def run(
            self, 
            wall: WallInstance, 
            openings: Optional[Opening]=None, 
            context_builder: Optional[ContextBuilder] = None,
            context_data: dict = None,
    ):
        """Single run inference episode"""

        self._load_model()
        vec_env = self._make_env(wall, openings, context_builder, context_data)

        obs = vec_env.reset()
        done = False
        steps = 0
        base_env = ModelRunner._get_base_env(vec_env)

        while not done:
            masks = base_env.get_action_mask()

            action, _ = self.model.predict(
                obs,
                deterministic=self.deterministic,
                action_masks=masks,
            )

            obs, rewards, dones, infos = vec_env.step(action)
            steps += 1
            done = dones[0]
            info = infos[0]

        result = {
            "reward": info.get("total_reward", 0.0),
            "steps": steps,
            "reason": info.get("reason", "?"),
            "grid": info.get("terminal_grid"),
            "instances": info.get("terminal_instances"),
            "wall": wall,
        }

        vec_env.close()
        return result

    def render(self, result):
        """
        Render previously computed result.
        """
        vec_env = self._make_env(result["wall"])
        base_env = ModelRunner._get_base_env(vec_env)

        base_env.grid_human = result["grid"]
        base_env.inst = result["instances"]
        base_env.render()

        vec_env.close()