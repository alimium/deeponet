import matplotlib.pyplot as plt
import optax
import equinox as eqx
from tqdm.auto import tqdm
from .model import update_fn, loss_fn

def fit(
    model,
    x_branch_train,
    x_trunk_train,
    y_train,
    optimizer,
    epochs,
    x_branch_test,
    x_trunk_test,
    y_test,
):
    opt_state = optimizer.init(
        eqx.filter(model, eqx.is_array),
    )
    training_loss_history = []
    test_loss_history = {}
    for i in tqdm(range(epochs)):
        model, opt_state, loss = update_fn(
            model,
            x_branch_train,
            x_trunk_train,
            y_train,
            opt_state,
            optimizer,
        )
        training_loss_history.append(loss)
        if i == 0 or (i + 1) % 1000 == 0:
            test_loss = loss_fn(model, x_branch_test, x_trunk_test, y_test)
            test_loss_history[i] = test_loss
            print(
                f"Epoch {i if i==0 else i+1:<5}| Loss: {loss:<24}| Test Loss: {test_loss}"
            )
            print(
                "-----------+-------------------------------+----------------------------------"
            )
    return model, (training_loss_history, test_loss_history)


class Trainer:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer or self._get_default_optimizer()
        self.latest_training_losses = None

    def _get_default_optimizer(self):
        return optax.adam(1e-3)
    
    def plot_training_losses(self, save_path=None):
        if self.latest_training_losses is None:
            print("No training data available.")
            return
        training_losses, test_losses = self.latest_training_losses
        fig = plt.figure(figsize=(15, 6))
        plt.semilogy(training_losses)
        plt.semilogy(test_losses.keys(), test_losses.values(), marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss")
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Plot saved at {save_path}")
            except Exception as e:
                print("An error occurred while saving the plot.")
                print(e)
        else:
            plt.show()
    
    def fit(
        self,
        x_branch_train,
        x_trunk_train,
        y_train,
        x_branch_test,
        x_trunk_test,
        y_test,
        epochs,
        ):

        self.model, self.latest_training_losses = fit(
            self.model,
            x_branch_train,
            x_trunk_train,
            y_train,
            self.optimizer,
            epochs,
            x_branch_test,
            x_trunk_test,
            y_test,
        )

        return self.model