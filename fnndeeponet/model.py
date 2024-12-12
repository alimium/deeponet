import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

class FNNDeepONet(eqx.Module):
    branch_net: eqx.nn.MLP
    trunk_net: eqx.nn.MLP
    bias: jax.Array

    def __init__(
        self,
        dim_branch_input: int,
        dim_intermediate: int,
        dim_trunk_input: int,
        branch_net_width: int,
        trunk_net_width: int,
        branch_net_depth: int,
        trunk_net_depth: int,
        activation,
        key,
    ):
        branch_key, trunk_key = jr.split(key)
        self.bias = jnp.zeros((dim_trunk_input,))
        self.branch_net = eqx.nn.MLP(
            in_size=dim_branch_input,
            out_size=dim_intermediate,
            width_size=branch_net_width,
            depth=branch_net_depth,
            activation=activation,
            key=branch_key,
        )
        self.trunk_net = eqx.nn.MLP(
            in_size=dim_trunk_input,
            out_size=dim_intermediate,
            width_size=trunk_net_width,
            depth=trunk_net_depth,
            activation=activation,
            final_activation=activation,
            key=trunk_key,
        )

    def __call__(self, x_branch, x_trunk):
        branch_net_pred = self.branch_net(x_branch)
        trunk_net_pred = self.trunk_net(x_trunk)
        results = jnp.sum(branch_net_pred * trunk_net_pred, keepdims=True) + self.bias
        return results[0]


@eqx.filter_jit
def loss_fn(model, x_branch, x_trunk, output):
    preds = jax.vmap(
        jax.vmap(
            model,
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )(x_branch, x_trunk)
    return jnp.mean(jnp.square(preds - output))


@eqx.filter_jit
def update_fn(model, x_branch, x_trunk, output, opt_state, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_branch, x_trunk, output)
    if isinstance(optimizer, optax.GradientTransformationExtraArgs):
        updates, new_state = optimizer.update(grads, opt_state, model, value=loss)
    else:
        updates, new_state = optimizer.update(grads, opt_state, model)
    new_params = eqx.apply_updates(model, updates)
    return new_params, new_state, loss
