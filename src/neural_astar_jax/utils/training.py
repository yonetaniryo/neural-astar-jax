from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .data import Instance


@partial(jax.jit, static_argnames="is_training")
def step(batch: Instance, state: TrainState, is_training: bool = True):
    def loss_fn(params):
        outputs, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch.map_design,
            batch.start_map,
            batch.goal_map,
            mutable=["batch_stats"],
        )
        loss = jnp.abs(outputs.history - batch.path_map).mean()
        return loss, (outputs, updates)

    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (outputs, updates)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates["batch_stats"])
    else:
        loss, (outputs, updates) = loss_fn(state.params)

    return loss, state, outputs, updates
