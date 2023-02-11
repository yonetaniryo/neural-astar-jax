from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, dataclass


@dataclass
class MazeDataLoader:
    filename: str
    split: str
    batch_size: int
    # shuffle: bool

    def __post_init__(self):
        data = np.load(self.filename)
        if self.split == "train":
            i = 0
        elif self.split == "val":
            i = 4
        else:
            i = 8

        self.map_designs = jnp.array(data[f"arr_{i}"])
        self.goal_maps = jnp.array(data[f"arr_{i + 1}"])[:, 0]
        self.opt_policies = jnp.array(data[f"arr_{i + 2}"])[:, :, 0]
        self.opt_dists = jnp.array(data[f"arr_{i + 3}"])[:, 0]

        self.sample_batch = self._build_sample_batch()

    def _build_sample_batch(self):
        def sample_batch(key):
            indices = jax.random.randint(
                key, (self.batch_size,), 0, len(self.map_designs)
            )
            map_designs = self.map_designs[indices]
            goal_maps = self.goal_maps[indices]
            opt_policies = self.opt_policies[indices]
            opt_dists = self.opt_dists[indices]
            key_array = jax.random.split(key, self.batch_size)
            start_maps = jax.vmap(sample_start)(key_array, opt_dists)
            path_maps = jax.vmap(get_opt_path_map)(start_maps, goal_maps, opt_policies)

            return map_designs, start_maps, self.goal_maps, path_maps

        return jax.jit(sample_batch)


@jax.jit
def get_opt_path_map(start_map, goal_map, opt_policy):
    goal_idx = jnp.argmax(goal_map)

    def next_loc(one_hot_action):
        actions = jnp.array(
            [[-1, 0], [0, 1], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, 1], [1, -1]]
        )
        return one_hot_action.dot(actions).astype(int)

    class Carry(NamedTuple):
        path_map: Array
        idx: Array
        t: int

    def cond(carry):
        return (carry.t < carry.path_map.size) & (carry.idx != goal_idx)

    def body(carry):
        action = next_loc(opt_policy.reshape(8, -1).T[carry.idx])
        idx = jnp.ravel_multi_index(
            jnp.array(jnp.unravel_index(carry.idx, start_map.shape)) + action,
            start_map.shape,
            mode="clip",
        )
        path_map = carry.path_map.at[idx].set(1)
        return Carry(path_map=path_map, idx=idx, t=carry.t + 1)

    path_map = jax.lax.while_loop(
        cond, body, Carry(path_map=start_map.flatten(), idx=jnp.argmax(start_map), t=0)
    ).path_map.reshape(start_map.shape)
    return path_map


@partial(jax.jit, static_argnames="pct")
def sample_start(key, opt_dist, pct=45):
    opt_dist_nan = opt_dist + (opt_dist == opt_dist.min()) * jnp.nan
    th = jnp.nanpercentile(opt_dist_nan, pct)
    start_cand = (
        jax.random.permutation(key, opt_dist.size).reshape(opt_dist.shape)
        + jnp.nan * (opt_dist > th)
        + (opt_dist == opt_dist.min()) * jnp.nan
    )
    start_idx = jnp.unravel_index(jnp.nanargmax(start_cand), opt_dist.shape)
    start_map = jnp.zeros(opt_dist.shape).at[start_idx].set(1)
    return start_map
