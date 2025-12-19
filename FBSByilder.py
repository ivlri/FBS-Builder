# fbs_builder_env.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

# === Block & Wall dataclasses / helpers ===

@dataclass(frozen=True)
class BlockType:
    id: int         # integer id stored in grid (>=2 to reserve 0=empty, 1=monolith)
    length_mm: int  # real length in mm
    name: str

    def num_cells(self, grid_step: int) -> int:
        """How many discrete grid cells the block occupies."""
        return self.length_mm // grid_step


@dataclass(frozen=True)
class WallInstance:
    id: int
    length_mm: int
    height_mm: int
    weight: int
    grid_step: int

    @property
    def num_cells(self) -> int:
        return self.length_mm // self.grid_step

    @property
    def num_layers(self) -> int:
        # block height always 600 mm (one "layer")
        return self.height_mm // 600


# === Default block types ===
BLOCK_TYPES = [
    BlockType(id=1, length_mm=20, name="Монолит"),
    BlockType(id=2, length_mm=2380, name="ФБС 24"),
    BlockType(id=3, length_mm=1180, name="ФБС 12"),
    BlockType(id=4, length_mm=880,  name="ФБС 9"),
    BlockType(id=5, length_mm=780,  name="ФБС 8"),
    BlockType(id=6, length_mm=580,  name="ФБС 6"),
]


# === Environment ===
class FBSBuilderEnv(gym.Env):
    """
    Gymnasium environment for FBS block placement.
    Action: Tuple(block_type_index, start_cell)
      - block_type_index: index into BLOCK_TYPES list (0..n-1)
      - start_cell: integer 0..(num_cells-1) representing the start cell for placement
    Observation: Dict with
      - 'grid': array shape (layers, num_cells), dtype=int8, 0=empty, 1=monolith, >=2 block ids
      - 'current_layer': scalar (which layer agent should fill next; 0 = bottom)
      - 'action_mask': boolean array (n_block_types, num_cells) marking legal (type, start) combos
    Reward shaping:
      - valid placement: +1
      - completing a layer without holes: +50
      - completing full wall: +200
      - invalid placement: -10
      - placement that creates single-cell gap (< min_block_cells): -2 (discourage tiny gaps)
    Episode ends when all layers filled OR no legal moves left OR max_steps reached.
    """
    metadata = {"render_modes": ["human", "terminal"], "render_fps": 1}

    def __init__(
        self,
        wall_instance: WallInstance = None,
        block_types: List[BlockType] = None,
        render_mode: str = None,
        max_steps: int = 1000
    ):
        super().__init__()
        self.reward = 0

        self.block_types = block_types or BLOCK_TYPES
        self.render_mode = render_mode
        self.max_steps = max_steps

        if wall_instance is None:
            wall_instance = WallInstance(
                id=0, length_mm=3000, height_mm=1800, weight=300, grid_step=20
            )

        self.wall_instance = wall_instance

        self.num_cells = self.wall_instance.num_cells
        self.num_layers = self.wall_instance.num_layers
        self.n_types = len(self.block_types)

        # internal state
        self.grid = None  # shape (layers, cells)
        self.grid_human = None  # shape (layers, cells)
        self.current_layer_numb = None
        self.step_count = 0

        #block instance
        self.instance_count = 1
        self.instance = {}

        # Precompute block lengths in cells
        self.block_cells = [
           bt.num_cells(self.wall_instance.grid_step) for bt in self.block_types 
        ]

        # action: (index into block_types, start cell 0..num_cells-1 for place block)
        self.action_space = spaces.MultiDiscrete([self.n_types, self.num_cells])

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, 
                               high=255, 
                               shape=(self.num_layers, self.num_cells), 
                               dtype=np.int16
                            ),
            "current_layer": spaces.Discrete(self.num_layers + 1),  # which layer is being filled next (0..num_layers)
            "action_mask": spaces.Box(low=0, 
                                      high=1, 
                                      shape=(self.n_types, self.num_cells), 
                                      dtype=np.int8
                                    )
        })

        self.reset()

    # ========= Core block placement ============
    def _intersects(self, layer: int, start: int, end: int) -> bool:
        """
        True if block placement intersects any non-empty cells on the same layer
        """
        try:
            out = np.any(self.grid[layer, start:end] != 0)
        except IndexError as ex:
            return False
        return out

    def _fits_bounds(self, start: int, block_cells: int) -> bool:
        return 0 <= start and (start + block_cells) <= self.num_cells

    def _check_bonding(self, layer: int, start: int, end: int) -> bool:
        """
        Bonding rule: each block must have at least one-cell support 
        overlap with blocks in the layer below.
        """
        if layer == 0:
            return True
        
        below = self.grid[layer - 1, start:end]

        return np.any(below != 0)

    def _vertical_suture_check(self, layer: int, start: int, end: int) -> int: 
        """
        Checking the "vertical_suture(перевязку швов)" between the blocks
        """
        if layer == 0:
            return 0
        

    # ========= Core penalty ============
    def _bonding_block_penalty(self, layer: int, heighth: int, step: int) -> int:
        if layer == 0:
            return 0

        min_mm = 0.4 * heighth
        min_cells = int(min_mm // step)

        cur = self.grid[layer]
        below = self.grid[layer - 1]

        seams_cur = self._find_boundaries(cur)
        seams_below = self._find_boundaries(below)

        if len(seams_cur) == 0 or len(seams_below) == 0:
            return 0

        penalty = 0
        for s in seams_cur:
            d = np.min(np.abs(seams_below - s))
            if d < min_cells:
                penalty += 5  

        return penalty
    
    def _big_mon_penalty(self, layer: int) -> int:
        """
        Checking for excessive monolithic sections (more than min(block_cells)).
        Total reward -50 for each similar plot
        """
        current_layer = self.grid[layer]        
        reward = 0
        gaps = []
        i = 0
        n = len(current_layer)
        while i < n:
            if current_layer[i] == 1:
                j = i
                while j < n and current_layer[j] == 1:
                    j += 1
                gaps.append(j - i)
                i = j
            else:
                i += 1

        # penalty
        min_block_cells = min(self.block_cells)
        for g in gaps:
            if g > min_block_cells:
                reward += 100
        return reward
    
    def _find_boundaries(self, row: np.ndarray) -> np.ndarray:
        bounds = []
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i+1] != 0 and row[i] != row[i+1]:
                bounds.append(i)
        return np.array(bounds, dtype=np.int32)

    # =========- Action mask / legal actions =========-
    def compute_action_mask(self) -> np.ndarray:
        """
        Return boolean mask shape (n_types, num_cells) 
        where True means action(type_idx, start) is a priori legal 
        (fits in bounds and does not intersect current layer occupied cells),
        AND satisfies bonding wrt layer below (if any).
        Note: we consider only current_layer for placement
        """
        mask = np.zeros((self.n_types, self.num_cells), dtype=np.int8)
        layer = self.current_layer_numb

        for t_idx in range(self.n_types):
            b_cells = self.block_cells[t_idx]

            # possible starts 0..num_cells - b_cells
            for s in range(0, self.num_cells - b_cells + 1):
                if not self._fits_bounds(s, b_cells):
                    continue
                if self._intersects(layer, s, s + b_cells):
                    continue
                if not self._check_bonding(layer, s, s + b_cells):
                    continue
                mask[t_idx, s] = 1
                
        return mask

    # ========= Gym-base =========
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.num_layers, self.num_cells), dtype=np.int32)
        self.grid_human = np.zeros((self.num_layers, self.num_cells), dtype=np.int32)

        self.instance_counter = 1
        self.instances = {}

        self.current_layer_numb = 0  # Нужно помнить - рендер будет в обратную сторону 
        self.step_count = 0
        self.reward = 0

        obs = self._get_obs()

        return obs, {}
    
    def step(self, action: Tuple[int, int]):
        """
        action: (type_index, start_cell)
        """
        self.step_count += 1
        info = {}
        terminated = False
        truncated = False

        t_idx, start = map(int, action)

        def fail(penalty=10.0):
            """Common failure handler"""
            self.reward -= penalty
            return self._get_obs(), float(self.reward), False, False, info

        # ========= Validate action =========
        if not (0 <= t_idx < self.n_types):
            return fail()

        if not (0 <= start < self.num_cells):
            return fail()

        b_cells = self.block_cells[t_idx]
        end = start + b_cells
        layer = self.current_layer_numb

        # ========= Placement checks =========
        if not self._fits_bounds(start, b_cells):
            return fail()

        if self._intersects(layer, start, end):
            return fail()

        if not self._check_bonding(layer, start, end):
            return fail()

        # ========= Place block =========
        instance_id = self.instance_counter
        self.instance_counter += 1

        block_type_id = self.block_types[t_idx].id

        self.instances[instance_id] = {
            "type_id": block_type_id,
            "layer": layer,
            "start": start,
            "end": end,
        }

        self.grid[layer, start:end] = instance_id
        self.grid_human[layer, start:end] = block_type_id
        # mark separator if not monolith
  
        self.reward -= self._bonding_block_penalty(layer=layer,
                                                    heighth=600,
                                                    step=self.wall_instance.grid_step)

        # ========= Layer completion =========
        if np.all(self.grid[layer] != 0):
            # self.reward += 50.0

            # penalties 
            self.reward -= self._big_mon_penalty(layer)
            self.current_layer_numb += 1

        # ========= Termination checks =========
        if self.current_layer_numb >= self.num_layers:
            self.reward += 200.0
            terminated = True
            info["reason"] = "all_layers_completed"

        elif self.step_count >= self.max_steps:
            truncated = True
            info["reason"] = "max_steps"

        return self._get_obs(), float(self.reward), terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        mask = self.compute_action_mask()
        type_grid = np.zeros_like(self.grid, dtype=np.int16)

        for inst_id, meta in self.instances.items():
            layer = meta["layer"]
            start = meta["start"]
            end = meta["end"]
            type_grid[layer, start:end] = meta["type_id"]

        obs = {
            "grid": type_grid,
            "current_layer": np.int64(self.current_layer_numb),
            "action_mask": mask
}
        return obs
    
    def get_action_mask(self):
        """
        Mask for MultiDiscrete([n_types, num_cells])
        Shape: (n_types + num_cells,)
        """
        
        joint_mask = self.compute_action_mask()  # (n_types, num_cells)

        type_mask = np.any(joint_mask, axis=1).astype(np.int8)  # (n_types,)

        start_mask = np.any(joint_mask, axis=0).astype(np.int8)  # (num_cells,)

        return np.concatenate([type_mask, start_mask])
    
    def render(self):
        if "terminal" in self.render_mode:
            if "human" in self.render_mode:
                grid = self.grid_human
            else:
                grid = self.grid


            print(f"\n=== Wall state: === \nCurrent_layer:{self.current_layer_numb}, step: {self.step_count}, reward{self.reward}")
            for layer in range(self.num_layers-1, -1, -1):
                row = grid[layer]
                print(f"L{layer} | " + "".join(f"{int(x)}" for x in row))

    def close(self):
        pass


# register environment so you can gym.make("FBSBuilder-v0")
register(
    id="FBSBuilder-v0",
    entry_point="FBSBuilder:FBSBuilderEnv",
)


# ========= Example usage =========
from stable_baselines3.common.callbacks import BaseCallback
class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True

if __name__ == "__main__":
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker
    import matplotlib.pyplot as plt

    grid_step = 20

    def mask_fn(env):
        return env.get_action_mask()
        
    def model_train(model, env):
        env = ActionMasker(env, mask_fn)

        obs, _ = env.reset()

        reward_callback = EpisodeRewardCallback()

        model.learn(callback=reward_callback, total_timesteps=100_000)

        model.save("ppo_fbs_builder")

        plt.figure(figsize=(10, 4))
        plt.plot(reward_callback.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("PPO training progress")
        plt.grid()
        plt.show()

    def validation(model, wall_type):
        env = FBSBuilderEnv(wall_instance=wall_type, render_mode="terminal", max_steps=500)
        env.render_mode = "terminal"
        model.load("ppo_fbs_builder")

        obs, _ = env.reset()
        done = False

        while not done:
            action_masks = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated

        print("Episode finished:", info)

    def manual_testing():
        grid_step = 20
        wall = WallInstance(id=0, length_mm=3000, height_mm=1800, weight=300, grid_step=grid_step)
        env = FBSBuilderEnv(wall_instance=wall, render_mode="terminal_human", max_steps=500)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while True:
            mask = obs["action_mask"]
            legal_next_move = np.argwhere(mask == 1)

            # if legal_next_move.shape[0] == 0:
            #     print("No legal moves remain.")
            #     break

            choice = legal_next_move[np.random.choice(len(legal_next_move))]
            action = (int(choice[0]), int(choice[1]))
            obs, reward, terminated, trunc, info = env.step(action)
            total_reward += reward
            env.render()
            if terminated or trunc:
                print("Episode finished:", info)
                break

        print("Total reward:", total_reward)

        
    wall = WallInstance(id=0, length_mm=3000, height_mm=1800, weight=300, grid_step=grid_step)
    env = FBSBuilderEnv(wall_instance=wall, render_mode=None, max_steps=500)
    model = MaskablePPO(
            policy=MaskableMultiInputActorCriticPolicy,
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            tensorboard_log="./fbs_tensorboard/"
        )
    
    #manual testing
    manual_testing()
    # Train
    # model_train(model=model,env=env)

    # Validation
    # validation(model=model, wall_type=wall)