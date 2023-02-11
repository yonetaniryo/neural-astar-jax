from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .data import Instance


@partial(jax.jit, static_argnames="is_training")
def step(batch: Instance, state: TrainState, is_training: bool = True):
    def loss_fn(params):
        output = state.apply_fn(
            params, batch.map_design, batch.start_map, batch.goal_map
        )
        history = output.history
        loss = optax.l2_loss(history, batch.path_map).mean()
        return loss, output

    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, output), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
    else:
        loss, output = loss_fn(state.params)

    return loss, state, output
