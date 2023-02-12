from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .differentiable_astar import DifferentiableAstar


class CNN(nn.Module):
    channels = [32, 64, 128, 256, 1]

    @nn.compact
    def __call__(self, map_designs, start_maps, goal_maps, is_training: bool = False):

        x = jnp.stack((map_designs, start_maps + goal_maps), -1)

        for i, b in enumerate(self.channels):
            x = nn.Conv(features=b, kernel_size=(3, 3))(x)
            x = nn.BatchNorm(use_running_average=not is_training, momentum=0.9)(x)
            x = nn.relu(x) if i < len(self.channels) - 1 else nn.sigmoid(x)

        return x[..., 0]


class VanillaAstar(nn.Module):
    """
    Vanilla differentiable A* planner

    Returns:
        g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
        search_step_ratio (float, optional): how much of the map the planner explores during training. Defaults to 1.0.
        is_training (bool, optional): if reverse-mode differentiation is enabled over loop. Defaults to False.
    """

    g_ratio: float = 0.5
    search_step_ratio: float = 1.0
    is_training: bool = False

    def setup(self):
        astar = DifferentiableAstar(
            g_ratio=self.g_ratio,
            search_step_ratio=self.search_step_ratio,
            is_training=self.is_training,
        )
        self.astar = jax.vmap(astar)

    def encode(self, map_designs, start_maps, goal_maps):
        return map_designs

    def __call__(self, map_designs, start_maps, goal_maps):
        cost_maps = self.encode(map_designs, start_maps, goal_maps)
        astar_output = self.astar(
            cost_maps,
            start_maps,
            goal_maps,
            map_designs,
        )

        return astar_output


class NeuralAstar(VanillaAstar):
    """
    Neural A* planner

    Returns:
        g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
        search_step_ratio (float, optional): how much of the map the planner explores during training. Defaults to 1.0.
        is_training (bool, optional): if reverse-mode differentiation is enabled over loop. Defaults to False.
    """

    g_ratio: float = 0.5
    search_step_ratio: float = 1.0
    is_training: bool = False

    def setup(self):
        super().setup()
        self.encoder = CNN()

    def encode(self, map_designs, start_maps, goal_maps):
        cost_maps = self.encoder(map_designs, start_maps, goal_maps, self.is_training)
        return cost_maps
