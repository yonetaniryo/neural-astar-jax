from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey, dataclass
from flax.training.train_state import TrainState

from ..planner.differentiable_astar import DifferentiableAstar


class TrainStateBN(TrainState):
    batch_stats: dict


@dataclass
class Trainer:
    planner: nn.Module
    train_loader: dataclass
    val_loader: dataclass
    learning_rate: float = 0.001
    seed: int = 0
    val_every_n_steps: int = 1

    def __post_init__(self):
        self.search_step_ratio = self.planner.search_step_ratio
        optimal_planner = DifferentiableAstar()
        key = jax.random.PRNGKey(self.seed)
        val_batch = self.val_loader.load_all_instances(key)
        optimal_plans = jax.vmap(optimal_planner)(
            val_batch.map_design,
            val_batch.start_map,
            val_batch.goal_map,
            val_batch.map_design,
        )
        self.val_batch, self.optimal_plans = val_batch, optimal_plans
        self.train_step = self._build_train_step()
        self.val_step = self._build_val_step()

    def fit(self, key: PRNGKey, state: TrainStateBN = None, max_steps: int = 10):
        batch = self.train_loader.sample_batch(key)
        if state == None:
            variables = self.planner.init(
                jax.random.PRNGKey(self.seed),
                batch.map_design,
                batch.start_map,
                batch.goal_map,
            )
            state = TrainStateBN.create(
                apply_fn=self.planner.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=optax.adam(learning_rate=self.learning_rate),
            )

        best_state = state
        best_h_mean = -jnp.inf

        for step in range(max_steps):
            key1, key = jax.random.split(key)
            train_loss, state = self.train_step(key1, state)
            if step % self.val_every_n_steps == 0:
                val_loss, p_opt, p_exp, h_mean = self.val_step(state)
                print(
                    f"{step=:02d}, {train_loss=:.4f}, {val_loss=:.4f}, {p_opt=:.4f}, {p_exp=:.4f}, {h_mean=:.4f}"
                )
                if h_mean > best_h_mean:
                    print(f"best model updated ({h_mean=:0.4f} > {best_h_mean=:0.4f})")
                    best_h_mean = h_mean
                    best_state = state

        return best_state

    def _build_train_step(self):
        def train_step(key: PRNGKey, state: TrainStateBN):
            self.planner.is_training = True
            self.planner.search_step_ratio = self.search_step_ratio
            self.planner.reset_differentiable_astar()

            batch = self.train_loader.sample_batch(key)

            def loss_fn(params):
                outputs, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    batch.map_design,
                    batch.start_map,
                    batch.goal_map,
                    mutable=["batch_stats"],
                )
                loss = jnp.abs(outputs.history - batch.path_map).mean()
                return loss, updates

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, updates), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])

            return loss, state

        return jax.jit(train_step)

    def _build_val_step(self):
        def val_step(state: TrainStateBN):
            self.planner.is_training = False
            self.planner.search_step_ratio = 1.0
            self.planner.reset_differentiable_astar()

            outputs = self.planner.apply(
                {"params": state.params, "batch_stats": state.batch_stats},
                self.val_batch.map_design,
                self.val_batch.start_map,
                self.val_batch.goal_map,
            )

            loss = jnp.abs(outputs.history - self.val_batch.path_map).mean()
            pathlen_opt = self.optimal_plans.path_map.sum((1, 2))
            pathlen_na = outputs.path_map.sum((1, 2))
            p_opt = (pathlen_opt == pathlen_na).mean()
            history_opt = self.optimal_plans.history.sum((1, 2))
            history_na = outputs.history.sum((1, 2))
            p_exp = jnp.maximum(
                (history_opt - history_na) / history_opt,
                0,
            ).mean()
            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            return loss, p_opt, p_exp, h_mean

        return jax.jit(val_step)
