from functools import partial

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from tqdm import tqdm


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
def eval_step(optim, x, y):
    y_inferred = optim.model(x)
    errors = y_inferred - y
    mae = jnp.abs(errors).mean()
    return mae


def do_train_epoch(optim, dataloader, pbar: bool = True):
    loss_list = []
    for i, (x, y) in enumerate(tqdm(dataloader, disable=not pbar)):
        x, y = jnp.asarray(x), jnp.asarray(y)

        step_loss = train_step(optim, x, y)
        loss_list.append(step_loss)

    return np.mean(loss_list)


def do_eval_epoch(optim, dataloader, pbar: bool = True):
    loss_list = []
    for i, (x, y) in enumerate(tqdm(dataloader, disable=not pbar)):
        x, y = jnp.asarray(x), jnp.asarray(y)

        step_loss = eval_step(optim, x, y)
        loss_list.append(step_loss)

    return np.mean(loss_list)
