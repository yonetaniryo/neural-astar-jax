from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .differentiable_astar import DifferentiableAstar


class CNN(nn.Module):
    @nn.compact
    def __call__(self, map_designs, start_maps, goal_maps):

        x = jnp.concatenate((map_designs, start_maps + goal_maps), -1)

        for i, b in enumerate([32, 64, 128, 256, 1]):
            x = nn.Conv(features=b, kernel_size=(3, 3))(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x) if i < 3 else nn.sigmoid(x)

        return x


class NeuralAstar(nn.Module):
    def setup(self):
        astar = DifferentiableAstar()
        self.astar = jax.vmap(astar)
        self.encoder = CNN()

    def __call__(self, map_designs, start_maps, goal_maps):
        cost_maps = self.encoder(map_designs, start_maps, goal_maps)
        astar_output = self.astar(
            cost_maps[..., 0],
            start_maps[..., 0],
            goal_maps[..., 0],
            map_designs[..., 0],
        )

        return astar_output
