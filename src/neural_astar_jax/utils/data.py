from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, dataclass


class Instance(NamedTuple):
    map_design: Array
    start_map: Array
    goal_map: Array
    path_map: Array


@dataclass
class MazeDataLoader:
    filename: str
    split: str
    batch_size: int

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

        self.sample_instance = self._build_sample_instance()
        self.sample_batch = self._build_sample_batch()

        self.N = len(self.map_designs)

    def _build_sample_batch(self):
        def sample_batch(key: PRNGKey) -> Instance:
            indices = jax.random.randint(key, (self.batch_size,), 0, self.N)
            key_array = jax.random.split(key, self.batch_size)

            return jax.vmap(self.sample_instance)(key_array, indices)

        return jax.jit(sample_batch)

    def _build_sample_instance(self):
        def sample_instance(key: PRNGKey, index: int) -> Instance:
            map_design = self.map_designs[index]
            goal_map = self.goal_maps[index]
            opt_policy = self.opt_policies[index]
            opt_dist = self.opt_dists[index]
            start_map = sample_start(key, opt_dist)
            path_map = get_opt_path_map(start_map, goal_map, opt_policy)

            return Instance(
                map_design=map_design,
                start_map=start_map,
                goal_map=self.goal_maps[index],
                path_map=path_map,
            )

        return jax.jit(sample_instance)

    def load_all_instances(self, key: PRNGKey) -> Instance:
        key_array = jax.random.split(key, self.N)
        return jax.vmap(self.sample_instance)(key_array, jnp.arange(self.N))


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
