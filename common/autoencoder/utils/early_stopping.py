import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        val_loss_min=None,
    ):
        """
        Initializes the EarlyStopping callback.

        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Function to use for printing messages.
                                   Default: print
            val_loss_min (float, optional): Initial minimum validation loss to start with.
                                            If None, it's initialized to infinity.
                                            Default: None
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = val_loss_min
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = val_loss_min if val_loss_min is not None else np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(
        self, val_loss, model, gen_optimizer, disc_optimizer, epoch, global_step
    ):
        """
        Checks if the validation loss has improved and updates the early stopping state.

        If the validation loss improves, it saves a checkpoint. Otherwise, it increments
        the counter and sets the `early_stop` flag if patience is reached.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            gen_optimizer (torch.optim.Optimizer): The generator's optimizer.
            disc_optimizer (torch.optim.Optimizer): The discriminator's optimizer.
            epoch (int): The current epoch number.
            global_step (int): The current global training step.
        """
        if self.best_val_loss is None or val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(
                val_loss, model, gen_optimizer, disc_optimizer, global_step
            )
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"Early Stopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(
        self, val_loss, model, gen_optimizer, disc_optimizer, global_step
    ):
        """Saves the model checkpoint when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                "best_val_loss": val_loss,
                "std": model.std if hasattr(model, "std") else None,
                "mean": model.mean if hasattr(model, "mean") else None,
            },
            self.path,
        )
        self.val_loss_min = val_loss