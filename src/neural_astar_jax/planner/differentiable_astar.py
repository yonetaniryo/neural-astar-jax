from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass


@partial(jax.jit, static_argnames="tb_factor")
def get_heuristic_map(goal_map, tb_factor=0.001):
    H, W = goal_map.shape
    goal_idx = jnp.argmax(goal_map)
    goal_y, goal_x = jnp.unravel_index(goal_idx, goal_map.shape)
    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    x, y = x.flatten(), y.flatten()
    euc = jnp.sqrt((goal_y - y) ** 2 + (goal_x - x) ** 2)
    cheb = (
        jnp.abs(goal_y - y)
        + jnp.abs(goal_x - x)
        - jnp.minimum(jnp.abs(goal_y - y), jnp.abs(goal_x - x))
    )
    h = (cheb + tb_factor * euc).reshape(goal_map.shape)
    return h


@jax.jit
def st_softmax_noexp(val):
    val_ = val.flatten()
    y = val_ / val_.sum()
    idx = jnp.argmax(val_)
    y_hard = jnp.zeros_like(val_)
    y_hard = y_hard.at[idx].set(1)
    y_hard = y_hard.reshape(val.shape)
    y = y.reshape(val.shape)
    zero = y - jax.lax.stop_gradient(y)
    return zero + jax.lax.stop_gradient(y_hard)


@jax.jit
def expand(idx_map):

    padded_map = jnp.zeros((idx_map.shape[0] + 2, idx_map.shape[1] + 2))
    padded_map = jax.lax.dynamic_update_slice(padded_map, idx_map, (1, 1))
    idx = jnp.argmax(padded_map)
    idx_y, idx_x = jnp.unravel_index(idx, padded_map.shape)
    neighbor = jnp.ones((3, 3)).at[1, 1].set(0)
    neighbor_map = jax.lax.dynamic_update_slice(
        padded_map, neighbor, (idx_y - 1, idx_x - 1)
    )
    return neighbor_map[1:-1, 1:-1]


def backtrack(goal_map, parents, t):
    if jnp.any(jnp.isnan(parents)):
        return goal_map
    path_map = goal_map.flatten()
    next_idx = int(jnp.argmax(goal_map.flatten()))

    for _ in range(t + 1):
        path_map = path_map.at[next_idx].set(1)
        next_idx = int(parents[next_idx])
    return path_map.reshape(goal_map.shape)


@dataclass
class DifferentiableAstar:
    Tmax: float = 1.0
    g_ratio: float = 0.5

    def __post_init__(self):
        self.forward = self.build_forward()

    def __call__(self, cost_map, start_map, goal_map, obstacles_map):
        return self.forward(cost_map, start_map, goal_map, obstacles_map)

    def build_forward(self):
        def forward(cost_map, start_map, goal_map, obstacles_map):
            H, W = cost_map.shape

            open_map = start_map
            history = jnp.zeros_like(start_map)
            parents = jnp.ones_like(start_map).flatten() * jnp.argmax(
                goal_map.flatten()
            )

            h = get_heuristic_map(goal_map) + cost_map
            g = jnp.zeros_like(start_map)

            T = start_map.size * self.Tmax

            class Carry(NamedTuple):
                g: Array
                idx_map: Array
                parents: Array
                open_map: Array
                history: Array
                t: int

            def cond(carry: Carry) -> bool:
                return ~(jnp.allclose(carry.idx_map, goal_map) | (carry.t > T))

            def body(carry: Carry) -> Carry:
                f = self.g_ratio * carry.g + (1 - self.g_ratio) * h
                f_exp = jnp.exp(-1 * f / jnp.sqrt(W))
                f_exp = f_exp * carry.open_map
                idx_map = st_softmax_noexp(f_exp)
                idx = jnp.argmax(idx_map)
                idx_y, idx_x = jnp.unravel_index(idx, obstacles_map.shape)

                history = jnp.clip(carry.history + idx_map, a_max=1)
                open_map = jnp.clip(carry.open_map - idx_map, a_min=0, a_max=1)
                neighbor_map = expand(idx_map) * obstacles_map

                g2 = (carry.g + cost_map) * neighbor_map
                neighbor_map = (
                    (1 - open_map) * (1 - history) + open_map * (carry.g > g2)
                ) * neighbor_map
                g = jax.lax.stop_gradient(
                    g2 * neighbor_map + carry.g * (1 - neighbor_map)
                )
                open_map = jax.lax.stop_gradient(
                    jnp.clip(open_map + neighbor_map, a_max=1)
                )
                parents = idx * neighbor_map.flatten() + carry.parents * (
                    1 - neighbor_map.flatten()
                )

                return Carry(
                    g=g,
                    idx_map=idx_map,
                    parents=parents,
                    open_map=open_map,
                    history=history,
                    t=carry.t + 1,
                )

            carry = jax.lax.while_loop(
                cond,
                body,
                Carry(
                    g=g,
                    idx_map=start_map,
                    parents=parents,
                    open_map=open_map,
                    history=history,
                    t=0,
                ),
            )

            return carry

        return jax.jit(forward)
