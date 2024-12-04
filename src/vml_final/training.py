from functools import partial

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from tqdm import tqdm

from vml_final.data import CSVDatasetEpochLoader


def get_fgsm_input(model, x, step_size=0.01):

    def infer(x):
        return model(x)

    grad = jax.grad(infer)(x)
    step = grad * 0.01
    return x + step, x - step


@nnx.jit
def train_step(optim, x, y):
    def loss_func(model):
        y_inferred = model(x)
        errors = y_inferred - y
        mae = jnp.abs(errors).mean()
        return mae

    loss, model_grad = nnx.value_and_grad(loss_func)(optim.model)
    optim.update(model_grad)

    return loss


@nnx.jit
def eval_step(model, x, y):
    y_inferred = model(x)
    errors = y_inferred - y
    mae = jnp.abs(errors).mean()
    return mae


def do_train_epoch(
    optim: nnx.Optimizer,
    dataloader: CSVDatasetEpochLoader,
    *,
    rngs: nnx.Rngs,
):

    optim.model.train()

    optim_graphdef, optim_state = nnx.split(optim)

    @jax.jit
    def scanf(optim_state, batch):
        x, y = batch
        optim = nnx.merge(optim_graphdef, optim_state)
        step_loss = train_step(optim, x, y)

        optim_state = nnx.state(optim)

        return optim_state, step_loss

    rng = rngs()
    (
        optim_state,
        epoch_loss_per_step,
    ) = dataloader.scan_for_epoch(rng, scanf, optim_state)

    nnx.update(optim, optim_state)

    return jnp.mean(epoch_loss_per_step)


def do_eval_epoch(
    model,
    dataloader: CSVDatasetEpochLoader,
    *,
    rngs: nnx.Rngs,
):

    model.eval()

    @jax.jit
    def infer(_, batch):
        x, y = batch
        step_loss = eval_step(model, x, y)

        return None, step_loss

    rng = rngs()
    _, epoch_loss_per_step = dataloader.scan_for_epoch(rng, infer, None)

    return jnp.mean(epoch_loss_per_step)
