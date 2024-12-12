import argparse
import sys
import jax
import optax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from fnndeeponet.model import FNNDeepONet
from fnndeeponet.train import Trainer

SEED = 42

def plot_random_data_points(
    n,
    branch_index,
    branch_data,
    trunk_index,
    output_data,
    /,
    model=None,
    trunk_data=None,
):
    ncols = 3
    nrows = n // ncols + 1 if n % ncols != 0 else n // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        idx = jr.randint(jr.PRNGKey(i), (1,), 0, train_branch_data.shape[0])[0]
        ax.plot(
            branch_index,
            branch_data[idx, :],
            label="u(x)",
        )
        ax.plot(
            trunk_index,
            output_data[idx, :],
            label="G(u)(y)",
        )

        if model is not None and trunk_data is not None:
            pred = jax.vmap(
                model,
                in_axes=(None, 0),
            )(branch_data[idx, :], trunk_data)
            ax.plot(trunk_index, pred, label="Prediction")

        ax.set_title(f"Data point {idx}")
        ax.grid()
        ax.legend()

    plt.show()

if __name__ == "__main__":
    # Show on which platform JAX is running
    print("JAX running on", jax.devices()[0].platform.upper())
    print("Available devices:", jax.devices())

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--branch-net-depth", type=int, default=2)
    parser.add_argument("--trunk-net-depth", type=int, default=3)
    parser.add_argument("--branch-net-width", type=int, default=40)
    parser.add_argument("--trunk-net-width", type=int, default=40)
    parser.add_argument("--dim-intermediate", type=int, default=64)
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--test-data-path", type=str)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # Load data
    try:
        train_data = jnp.load(
            args.train_data_path, allow_pickle=True
        )
        test_data = jnp.load(
            args.test_data_path, allow_pickle=True
        )

        # training data
        train_branch_data = train_data["X"][0]
        train_trunk_data = train_data["X"][1]
        train_outputs = train_data["y"]

        # testing data
        test_branch_data = test_data["X"][0]
        test_trunk_data = test_data["X"][1]
        test_outputs = test_data["y"]

        if args.verbose:
            print("Training data:")
            print("\tBranch data shape:", train_branch_data.shape)
            print("\tTrunk data shape:", train_trunk_data.shape)
            print("\tOutput shape:", train_outputs.shape)
            print("Testing data:")
            print("\tBranch data shape:", test_branch_data.shape)
            print("\tTrunk data shape:", test_trunk_data.shape)
            print("\tOutput shape:", test_outputs.shape)
    except FileNotFoundError as e:
        print("Data not found. Please check the path.")
        if args.verbose:
            print(e)
    except IndexError | KeyError as e:
        print("Data is not in the correct format.")
        if args.verbose:
            print(e)

    # Initialize model
    key = jr.PRNGKey(SEED)
    optimizer = optax.chain(
        optax.adam(1e-3),
        optax.contrib.reduce_on_plateau(patience=3, cooldown=5, accumulation_size=10),
    )
    model = FNNDeepONet(
        dim_branch_input=train_branch_data.shape[1],
        dim_intermediate=args.dim_intermediate,
        dim_trunk_input=train_trunk_data.shape[1],
        branch_net_width=args.branch_net_width,
        trunk_net_width=args.trunk_net_width,
        branch_net_depth=args.branch_net_depth,
        trunk_net_depth=args.trunk_net_depth,
        activation=jax.nn.gelu,
        key=key,
    )

    # Train model
    try:
        trainer = Trainer(model, optimizer)
        trained_model = trainer.fit(
            train_branch_data,
            train_trunk_data,
            train_outputs,
            test_branch_data,
            test_trunk_data,
            test_outputs,
            args.epochs,
        )
    except Exception as e:
        print("An error occurred during training.")
        if args.verbose:
            print(e)
            

    if args.verbose:
        fit_loss = trainer.latest_training_losses
        if fit_loss is not None:
            training_losses, test_losses = fit_loss
            print(f"Training complete with {args.epochs} epochs.")
            print(f"\tTraining loss: {training_losses[-1]}, Min training loss: {min(training_losses)}")
            print(f"\tTest loss: {list(test_losses.values())[-1]}, Min test loss: {min(test_losses.values())}")

    trainer.plot_training_losses(save_path='./training_losses.png')
    if args.verbose:
        trainer.plot_training_losses()
    

    





