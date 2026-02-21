import numpy as np
import torch
from typing import List
from common.metrics.metrics_streaming_probabilistic import MetricsAccumulator


class EarlyStopping:
    """Early stops the training if the monitored metric doesn't improve (according to metric_direction)
    after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        metric_direction="minimize",
        initial_best_metric=None,
    ):
        """
        Args:
            patience (int): How many epochs to wait after the last time the monitored metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                            Default: print
            metric_direction (str): Direction in which to optimize the metric.
                                    Possible values: 'minimize' or 'maximize'.
                                    Default: 'minimize'
            initial_best_metric (float): Starting best metric value.
                                         If None, will be set to np.Inf for 'minimize', or -np.Inf for 'maximize'.
                                         Default: None
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric_direction = metric_direction.lower()

        if self.metric_direction not in ["minimize", "maximize"]:
            raise ValueError("metric_direction must be either 'minimize' or 'maximize'")

        if initial_best_metric is not None:
            self.best_metric = initial_best_metric
        else:
            self.best_metric = (
                np.Inf if self.metric_direction == "minimize" else -np.Inf
            )

        self.best_epoch = None
        self.early_stop = False

    def __call__(self, metric, model, optimizer, epoch, global_step):
        """
        Checks if there is an improvement in the monitored metric and updates
        early stopping parameters accordingly.

        Args:
            metric (float): Current value of the monitored metric.
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer used.
            epoch (int): Current epoch number.
            global_step (int): Current global step.
        """
        improvement = False
        previous_best_metric = self.best_metric

        if self.metric_direction == "minimize":
            if metric < self.best_metric - self.delta:
                improvement = True
        else:
            if metric > self.best_metric + self.delta:
                improvement = True

        if improvement:
            self.best_metric = metric
            self.best_epoch = epoch
            self.save_checkpoint(
                metric, model, optimizer, global_step, previous_best_metric
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
        self, metric, model, optimizer, global_step, previous_best_metric=None
    ):
        """Saves model when metric improves."""
        if self.verbose:
            self.trace_func(
                f"Metric improved from {previous_best_metric:.6f} to {metric:.6f}. Saving model"
            )
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": metric,
                "std": getattr(model, "std", None),
                "mean": getattr(model, "mean", None),
                "max_value": getattr(model, "max_value", None),
                "min_value": getattr(model, "min_value", None),
            },
            self.path,
        )

    def save_last_epoch_checkpoint(self, metric, model, optimizer, global_step, path):
        """Saves the model checkpoint at the end of an epoch, overwriting the previous one.

        This is useful for saving the state of the last epoch, which might not be the best one.

        Args:
            metric (float): The metric value for the current epoch.
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            global_step (int): The total number of training steps.
            path (str): The path where the checkpoint will be saved.
        """
        if self.verbose:
            self.trace_func(
                f"Saving last epoch checkpoint with metric {metric:.6f}. Overwriting previous checkpoint."
            )
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": metric,
                "std": getattr(model, "std", None),
                "mean": getattr(model, "mean", None),
                "max_value": getattr(model, "max_value", None),
                "min_value": getattr(model, "min_value", None),
            },
            path,
        )


def compute_mean_std(
    loader,
    without_channels=False,
    channel_last=True,
    cascaded=False,
    residual_mode=False,
):
    """
    Computes the mean and standard deviation of a dataset provided by a DataLoader.

    Args:
        loader (DataLoader): DataLoader for the dataset.
        without_channels (bool): If True, computes statistics over the entire tensor,
                                 not per channel. Default: False.
        channel_last (bool): If True, assumes data format with channels as the last dimension.
                             Affects how per-channel statistics are computed. Default: True.
        cascaded (bool): If True, indicates a cascaded model setup, affecting data handling.
                         Default: False.
        residual_mode (bool): If True (and `cascaded` is True), computes statistics
                              on the residuals instead of the main data. Default: False.

    Returns:
        mean (torch.Tensor): The mean of the dataset.
        std (torch.Tensor): The standard deviation of the dataset.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in loader:
        if cascaded and residual_mode:
            inputs, predicted, residuals, metadata = data
            if without_channels:
                channels_sum += torch.mean(residuals)
                channels_squared_sum += torch.mean(residuals**2)
                num_batches += 1
            else:
                if channel_last:
                    channels_sum += torch.mean(residuals, dim=[0, 2, 3])
                    channels_squared_sum += torch.mean(residuals**2, dim=[0, 2, 3])
                else:
                    channels_sum += torch.mean(residuals, dim=[0, 3, 4])
                    channels_squared_sum += torch.mean(residuals**2, dim=[0, 3, 4])
                num_batches += 1
        else:
            if without_channels:
                channels_sum += torch.mean(data)
                channels_squared_sum += torch.mean(data**2)
                num_batches += 1

            else:
                if isinstance(data, (list, tuple)):
                    data = data[0]

                if channel_last:
                    channels_sum += torch.mean(data, dim=[0, 2, 3])
                    channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
                else:
                    channels_sum += torch.mean(data, dim=[0, 3, 4])
                    channels_squared_sum += torch.mean(data**2, dim=[0, 3, 4])
                num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    mean = mean.mean()
    std = std.mean()

    return mean, std


def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    """
    Creates a learning rate scheduler function for a linear warmup.

    The learning rate increases linearly from `min_lr_ratio` * initial_lr to initial_lr
    over `warmup_steps`, and then stays constant.

    Args:
        warmup_steps (int): The number of steps for the warmup phase.
        min_lr_ratio (float): The starting learning rate ratio. Default: 0.1.

    Returns:
        function: A lambda function that takes the current epoch and returns a learning rate multiplier.
    """

    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0

    return ret_lambda


def safe_mean(values):
    """
    Computes the mean of a list of values, filtering out None and NaN values.

    Args:
        values (list): A list of numerical values, possibly containing None or NaN.

    Returns:
        float or None: The mean of the filtered values, or None if the list is empty after filtering.
    """
    filtered = [v for v in values if v is not None and not np.isnan(v)]
    return np.mean(filtered) if filtered else None


def calculate_metrics(
    num_lead_times: int,
    thresholds: List[int],
    metrics_accumulators: List[MetricsAccumulator],
) -> dict:
    """
    Calculate and aggregate metrics for a probabilistic nowcasting model.

    This function processes metrics from `MetricsAccumulator` objects for each lead time,
    computes various statistics like MSE, CRPS, CSI, etc., both for individual lead times
    and as overall averages.

    Args:
        num_lead_times (int): The number of lead times to calculate metrics for.
        thresholds (List[int]): A list of precipitation thresholds for computing categorical metrics.
        metrics_accumulators (List[MetricsAccumulator]): A list of `MetricsAccumulator` objects,
                                                       one for each lead time.

    Returns:
        dict: A dictionary containing a wide range of computed metrics
    """
    crps_t_list = []

    mse_from_mean_t_list = []
    pod_from_mean_t_dict = {th: [] for th in thresholds}
    csi_from_mean_t_dict = {th: [] for th in thresholds}
    far_from_mean_t_dict = {th: [] for th in thresholds}
    hss_from_mean_t_dict = {th: [] for th in thresholds}
    csi_pool_from_mean_t_dict = {th: [] for th in thresholds}

    csi_pool_m_from_mean_lead_time = []
    csi_m_from_mean_lead_time = []
    far_m_from_mean_lead_time = []
    hss_m_from_mean_lead_time = []
    pod_m_from_mean_lead_time = []
    csi_last_thresh_from_mean_lead_time = []
    csi_pool_last_thresh_from_mean_lead_time = []

    fss_from_mean_t_dict_by_th_scale = {}
    fss_scales = (
        metrics_accumulators[0].fss_scales
        if metrics_accumulators
        and hasattr(metrics_accumulators[0], "fss_scales")
        and metrics_accumulators[0].fss_scales
        else []
    )
    if fss_scales:
        fss_from_mean_t_dict_by_th_scale = {
            th: {scale: [] for scale in fss_scales} for th in thresholds
        }

    fss_m_from_mean_lead_time = []

    # Loop over each lead time and call the merged streaming function.
    for lead_time in range(num_lead_times):
        results = metrics_accumulators[lead_time].compute()
        crps_t_list.append(results.get("crps"))

        mse_from_mean_t_list.append(results.get("mse_from_mean"))

        for th in thresholds:
            pod_from_mean_t_dict[th].append(
                results["pod_from_mean"][th] if results["pod_from_mean"] else None
            )
            csi_from_mean_t_dict[th].append(
                results["csi_from_mean"][th] if results["csi_from_mean"] else None
            )
            far_from_mean_t_dict[th].append(
                results["far_from_mean"][th] if results["far_from_mean"] else None
            )
            hss_from_mean_t_dict[th].append(
                results["hss_from_mean"][th] if results["hss_from_mean"] else None
            )
            csi_pool_from_mean_t_dict[th].append(
                results["csi_pooled_from_mean"][th]
                if results["csi_pooled_from_mean"]
                else None
            )

        csi_pool_m_from_mean_lead_time.append(
            safe_mean([csi_pool_from_mean_t_dict[th][lead_time] for th in thresholds])
        )
        csi_m_from_mean_lead_time.append(
            safe_mean([csi_from_mean_t_dict[th][lead_time] for th in thresholds])
        )
        far_m_from_mean_lead_time.append(
            safe_mean([far_from_mean_t_dict[th][lead_time] for th in thresholds])
        )
        hss_m_from_mean_lead_time.append(
            safe_mean([hss_from_mean_t_dict[th][lead_time] for th in thresholds])
        )
        pod_m_from_mean_lead_time.append(
            safe_mean([pod_from_mean_t_dict[th][lead_time] for th in thresholds])
        )
        csi_last_thresh_from_mean_lead_time.append(
            csi_from_mean_t_dict[thresholds[-1]][lead_time]
        )
        csi_pool_last_thresh_from_mean_lead_time.append(
            csi_pool_from_mean_t_dict[thresholds[-1]][lead_time]
        )

        fss_from_mean_results = results.get("fss_from_mean")
        if fss_scales:
            lead_time_fss_from_mean_scores = []
            if fss_from_mean_results:
                for th in thresholds:
                    for scale in fss_scales:
                        score = fss_from_mean_results.get(th, {}).get(scale)
                        fss_from_mean_t_dict_by_th_scale[th][scale].append(score)
                        if score is not None and not np.isnan(score):
                            lead_time_fss_from_mean_scores.append(score)
            fss_m_from_mean_lead_time.append(safe_mean(lead_time_fss_from_mean_scores))
        else:
            fss_m_from_mean_lead_time.append(None)

    crps_mean = safe_mean(crps_t_list)
    mse_from_mean_mean = safe_mean(mse_from_mean_t_list)
    pod_from_mean_mean = {th: safe_mean(pod_from_mean_t_dict[th]) for th in thresholds}
    csi_from_mean_mean = {th: safe_mean(csi_from_mean_t_dict[th]) for th in thresholds}
    far_from_mean_mean = {th: safe_mean(far_from_mean_t_dict[th]) for th in thresholds}
    hss_from_mean_mean = {th: safe_mean(hss_from_mean_t_dict[th]) for th in thresholds}
    csi_pool_from_mean_mean = {
        th: safe_mean(csi_pool_from_mean_t_dict[th]) for th in thresholds
    }

    def mean_of_dict(d):
        vals = [v for v in d.values() if v is not None]
        return np.mean(vals) if vals else None

    csi_from_mean_m = mean_of_dict(csi_from_mean_mean)
    pod_from_mean_m = mean_of_dict(pod_from_mean_mean)
    far_from_mean_m = mean_of_dict(far_from_mean_mean)
    hss_from_mean_m = mean_of_dict(hss_from_mean_mean)
    csi_pool_from_mean_m = mean_of_dict(csi_pool_from_mean_mean)

    if fss_scales:
        fss_from_mean = {
            th: {
                scale: safe_mean(fss_from_mean_t_dict_by_th_scale[th][scale])
                for scale in fss_scales
            }
            for th in thresholds
        }
        fss_mean_from_mean = {
            th: safe_mean(list(fss_from_mean[th].values())) for th in thresholds
        }
        fss_m_from_mean_by_scale = {
            scale: safe_mean(
                [fss_from_mean[th][scale] for th in thresholds if th in fss_from_mean]
            )
            for scale in fss_scales
        }
        fss_m_from_mean = safe_mean(list(fss_m_from_mean_by_scale.values()))
    else:
        fss_from_mean = None
        fss_mean_from_mean = None
        fss_m_from_mean = None
        fss_m_from_mean_by_scale = None

    return {
        "crps_t_list": crps_t_list,
        "crps_mean": crps_mean,
        "mse_from_mean_t_list": mse_from_mean_t_list,
        "pod_from_mean_t_dict": pod_from_mean_t_dict,
        "csi_from_mean_t_dict": csi_from_mean_t_dict,
        "far_from_mean_t_dict": far_from_mean_t_dict,
        "hss_from_mean_t_dict": hss_from_mean_t_dict,
        "csi_pool_from_mean_t_dict": csi_pool_from_mean_t_dict,
        "mse_from_mean_mean": mse_from_mean_mean,
        "pod_from_mean_mean": pod_from_mean_mean,
        "csi_from_mean_mean": csi_from_mean_mean,
        "far_from_mean_mean": far_from_mean_mean,
        "hss_from_mean_mean": hss_from_mean_mean,
        "csi_pool_from_mean_mean": csi_pool_from_mean_mean,
        "csi_from_mean_m": csi_from_mean_m,
        "pod_from_mean_m": pod_from_mean_m,
        "far_from_mean_m": far_from_mean_m,
        "hss_from_mean_m": hss_from_mean_m,
        "csi_pool_from_mean_m": csi_pool_from_mean_m,
        "csi_pool_m_from_mean_lead_time": csi_pool_m_from_mean_lead_time,
        "csi_m_from_mean_lead_time": csi_m_from_mean_lead_time,
        "far_m_from_mean_lead_time": far_m_from_mean_lead_time,
        "hss_m_from_mean_lead_time": hss_m_from_mean_lead_time,
        "pod_m_from_mean_lead_time": pod_m_from_mean_lead_time,
        "csi_last_thresh_from_mean_lead_time": csi_last_thresh_from_mean_lead_time,
        "csi_pool_last_thresh_from_mean_lead_time": csi_pool_last_thresh_from_mean_lead_time,
        "fss_from_mean": fss_from_mean,
        "fss_mean_from_mean": fss_mean_from_mean,
        "fss_m_from_mean": fss_m_from_mean,
        "fss_m_from_mean_by_scale": fss_m_from_mean_by_scale,
        "fss_m_from_mean_lead_time": fss_m_from_mean_lead_time,
    }


def ema(source, target, decay):
    """
    Applies Exponential Moving Average (EMA) to update model weights.

    Args:
        source (torch.nn.Module): The model with the latest weights.
        target (torch.nn.Module): The model whose weights will be updated with the EMA.
        decay (float): The decay factor for the moving average.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
