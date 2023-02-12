from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass


class AstarOutput(NamedTuple):
    path_map: Array
    history: Array


@partial(jax.jit, static_argnames="tb_factor")
def _get_heuristic_map(goal_map: Array, tb_factor: float = 0.001) -> Array:
    """
    Compute Chebyshev heuristic

    Args:
        goal_map (Array): one-hot goal map
        tb_factor (float, optional): tie-breaking factor. Defaults to 0.001.

    Returns:
        Array: heuristics map
    """
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
def _st_softmax_noexp(val: Array) -> Array:
    """
    straight-through soft-max function taking exp(-f) as input

    Args:
        val (Array): exp(-f)

    Returns:
        Array: one-hot map for selected index
    """

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
def _expand(idx_map: Array) -> Array:
    """
    Expand eight neighbors of the selected index

    Args:
        idx_map (Array): One-hot map for selected index

    Returns:
        Array: Binary map for neighbors
    """

    padded_map = jnp.zeros((idx_map.shape[0] + 2, idx_map.shape[1] + 2))
    padded_map = jax.lax.dynamic_update_slice(padded_map, idx_map, (1, 1))
    idx = jnp.argmax(padded_map)
    idx_y, idx_x = jnp.unravel_index(idx, padded_map.shape)
    neighbor = jnp.ones((3, 3)).at[1, 1].set(0)
    neighbor_map = jax.lax.dynamic_update_slice(
        padded_map, neighbor, (idx_y - 1, idx_x - 1)
    )
    return neighbor_map[1:-1, 1:-1]


@jax.jit
def _backtrack(parents: Array, start_map: Array, goal_map: Array) -> Array:
    """
    Backtracking operation to produce path

    Args:
        parents (Array): Array indicating parent indices
        start_map (Array): one-hot start map
        goal_map (Array): one-hot goal map

    Returns:
        Array: Path map
    """

    path_map = goal_map.flatten()
    start_idx = jnp.argmax(start_map.flatten())
    next_idx = jnp.argmax(path_map).astype(int)

    class Carry(NamedTuple):
        path_map: Array
        next_idx: Array
        t: int

    def cond(carry):
        return (carry.t < start_map.size) & (carry.path_map[start_idx] == 0)

    def body(carry):
        path_map = carry.path_map.at[carry.next_idx].set(1)
        next_idx = parents[carry.next_idx].astype(int)
        return Carry(path_map=path_map, next_idx=next_idx, t=carry.t + 1)

    carry = jax.lax.while_loop(
        cond, body, Carry(path_map=path_map, next_idx=next_idx, t=0)
    )
    return carry.path_map.reshape(goal_map.shape)


@dataclass
class DifferentiableAstar:
    """
    Differentiable A* module

    Returns:
        g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
        search_step_ratio (float, optional): how much of the map the planner explores during training. Defaults to 1.0.
        is_training (bool, optional): if reverse-mode differentiation is enabled over loop. Defaults to False.
    """

    g_ratio: float = 0.5
    search_step_ratio: float = 1.0
    is_training: bool = False

    def __post_init__(self):
        self.forward = self.build_forward()

    def __call__(
        self, cost_map: Array, start_map: Array, goal_map: Array, obstacles_map: Array
    ) -> AstarOutput:
        """
        Perform differentiable A*

        Args:
            cost_map (Array): cost map
            start_map (Array): one-hot start map
            goal_map (Array): one-hot goal map
            obstacles_map (Array): binary obstalces map indicating 1 for passable and 0 for otherwise

        Returns:
            AstarOutput: namedtuple of path_map and history
        """

        return self.forward(cost_map, start_map, goal_map, obstacles_map)

    def build_forward(self):
        class Carry(NamedTuple):
            g: Array
            idx_map: Array
            parents: Array
            open_map: Array
            history: Array
            t: int

        def forward(
            cost_map: Array, start_map: Array, goal_map: Array, obstacles_map: Array
        ) -> AstarOutput:
            """
            Perform differentiable A*
            Reverse-mode differentiation over loop is implemented using:
            https://github.com/google/jax/discussions/3850#discussioncomment-45954

            Args:
                cost_map (Array): cost map
                start_map (Array): one-hot start map
                goal_map (Array): one-hot goal map
                obstacles_map (Array): binary obstalces map indicating 1 for passable and 0 for otherwise

            Returns:
                AstarOutput: namedtuple of path_map and history
            """

            size = cost_map.shape[-1]

            open_map = start_map
            history = jnp.zeros_like(start_map)
            parents = jnp.ones_like(start_map).flatten() * jnp.argmax(
                goal_map.flatten()
            )

            h = _get_heuristic_map(goal_map) + cost_map
            g = jnp.zeros_like(start_map)

            T = start_map.size * self.search_step_ratio

            def cond(carry: Carry) -> bool:
                return ~jnp.allclose(carry.idx_map, goal_map) & (carry.t < T)

            def step_once(carry: Carry) -> Carry:
                f = self.g_ratio * carry.g + (1 - self.g_ratio) * h
                f_exp = jnp.exp(-1 * f / jnp.sqrt(size))
                f_exp = f_exp * carry.open_map
                idx_map = _st_softmax_noexp(f_exp)
                idx = jnp.argmax(idx_map)

                history = jnp.clip(carry.history + idx_map, a_max=1)
                open_map = jnp.clip(carry.open_map - idx_map, a_min=0, a_max=1)
                neighbor_map = _expand(idx_map) * obstacles_map

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

                return (
                    Carry(
                        g=g,
                        idx_map=idx_map,
                        parents=parents,
                        open_map=open_map,
                        history=history,
                        t=carry.t + 1,
                    ),
                    None,
                )

            def do_nothing(carry):
                return carry, None

            def body(carry, x):
                return jax.lax.cond(cond(carry), step_once, do_nothing, carry)

            init = Carry(
                g=g,
                idx_map=start_map,
                parents=parents,
                open_map=open_map,
                history=history,
                t=0,
            )
            if self.is_training:
                # the "body" function will be repeated for T steps. Set smaller search_step_ratio for acceleration
                carry, _ = jax.lax.scan(body, init, None, T)
            else:
                # the "step_once" function will be repeated until goal is found.
                carry = jax.lax.while_loop(cond, lambda c: step_once(c)[0], init)

            path_map = _backtrack(carry.parents, start_map, goal_map)

            return AstarOutput(path_map=path_map, history=carry.history)

        return jax.jit(forward)
