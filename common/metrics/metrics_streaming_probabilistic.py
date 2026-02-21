import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from common.metrics.fss import fss_init, fss_accum, fss_compute

def crps(
    y_true_tensor: torch.Tensor,
    y_pred_tensor: torch.Tensor,
    pool_type: str = "none",
    scale: int = 1,
    mode: str = "mean",
    eps: float = 1e-10,
) -> float:
    """
    Computes the Continuous Ranked Probability Score (CRPS) using PyTorch tensors.

    Args:
        y_true_tensor (torch.Tensor): Ground truth tensor with shape (b, T, c, h, w),
                                      where b is batch size, T is number of lead times,
                                      c is channels, h is height, and w is width.
        y_pred_tensor (torch.Tensor): Prediction tensor with shape (b, n, T, c, h, w),
                                      where n is the number of ensemble samples.
        pool_type (str, optional): Type of spatial pooling to apply.
                                   Can be "none", "avg", or "max". Defaults to "none".
        scale (int, optional): The kernel size and stride for spatial pooling. Defaults to 1.
        mode (str, optional): Aggregation mode for the final score. Can be "mean" or "sum".
                              Defaults to "mean".
        eps (float, optional): A small epsilon value to prevent division by zero in
                               standard deviation calculation. Defaults to 1e-10.

    Returns:
        float: The computed CRPS value, aggregated according to the `mode`.
    """
    device = y_true_tensor.device
    _normal_dist = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        validate_args=False,
    )
    _frac_sqrt_pi = 1.0 / np.sqrt(np.pi)
    b_shape, T_shape, c_shape, h_shape, w_shape = y_true_tensor.shape
    _, n_shape, _, _, _, _ = y_pred_tensor.shape

    gt_for_pool = y_true_tensor.reshape(b_shape * T_shape, c_shape, h_shape, w_shape)
    pred_for_pool = y_pred_tensor.reshape(
        b_shape * n_shape * T_shape, c_shape, h_shape, w_shape
    )

    if scale > 1 and pool_type in ["avg", "max"]:
        if pool_type == "avg":
            gt_pooled = F.avg_pool2d(gt_for_pool, kernel_size=scale, stride=scale)
            pred_pooled = F.avg_pool2d(pred_for_pool, kernel_size=scale, stride=scale)
        elif pool_type == "max":  # pool_type == "max"
            gt_pooled = F.max_pool2d(gt_for_pool, kernel_size=scale, stride=scale)
            pred_pooled = F.max_pool2d(pred_for_pool, kernel_size=scale, stride=scale)
    else:  # No pooling or invalid pool_type
        gt_pooled = gt_for_pool
        pred_pooled = pred_for_pool

    new_h, new_w = gt_pooled.shape[-2:]

    gt_rearr = gt_pooled.reshape(b_shape, T_shape, c_shape, new_h, new_w)
    pred_rearr = pred_pooled.reshape(b_shape, n_shape, T_shape, c_shape, new_h, new_w)

    pred_mean = torch.mean(pred_rearr, dim=1)
    pred_std = (
        torch.std(pred_rearr, dim=1, unbiased=True)
        if n_shape > 1
        else torch.zeros_like(pred_mean)
    )

    normed_diff = (pred_mean - gt_rearr) / (pred_std + eps)

    cdf = _normal_dist.cdf(normed_diff)
    pdf = torch.exp(_normal_dist.log_prob(normed_diff))

    crps_val_tensor = (pred_std + eps) * (
        normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi
    )

    if mode == "mean":
        return torch.mean(crps_val_tensor).item()
    elif mode == "sum":
        return torch.sum(crps_val_tensor).item()
    return torch.mean(crps_val_tensor).item()


class MetricsAccumulator:
    """
    A class to accumulate and compute various metrics for probabilistic weather forecasts
    in a streaming (chunk-by-chunk) manner, using the Metric of Ensemble Mean approach.

    This class is designed to handle large datasets that may not fit into memory by
    processing data in chunks.
    """

    def __init__(
        self,
        lead_time: int,
        thresholds: Optional[List[float]] = None,
        pool_size: int = 16,
        compute_mse: bool = True,
        compute_threshold: bool = True,
        compute_crps: bool = True,
        compute_fss: bool = True,
        fss_scales: Optional[List[int]] = None,
        crps_pool_type: str = "none",
        crps_scale: int = 1,
        crps_eps: float = 1e-10,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the MetricsAccumulator.

        Args:
            lead_time (int): The specific lead time index to compute metrics for.
            thresholds (Optional[List[float]], optional): A list of thresholds for
                categorical metrics (CSI, POD, etc.). Defaults to [0.5].
            pool_size (int, optional): The kernel size for max-pooling when computing
                the pooled CSI metric. Defaults to 16.
            compute_mse (bool, optional): Whether to compute Mean Squared Error. Defaults to True.
            compute_threshold (bool, optional): Whether to compute threshold-based metrics.
                Defaults to True.
            compute_crps (bool, optional): Whether to compute Continuous Ranked Probability Score.
                Defaults to True.
            crps_pool_type (str, optional): The type of pooling to apply before CRPS calculation
                ("none", "avg", "max"). Defaults to "none".
            crps_scale (int, optional): The scale/kernel size for CRPS pooling. Defaults to 1.
            crps_eps (float, optional): Epsilon for CRPS calculation to avoid division by zero.
                Defaults to 1e-10.
            device (Optional[torch.device], optional): The device to perform computations on.
                If None, defaults to CUDA if available, otherwise CPU.
        """
        self.lead_time = lead_time
        self.thresholds = thresholds if thresholds is not None else [0.5]
        self.pool_size = pool_size
        self.compute_mse = compute_mse
        self.compute_threshold = compute_threshold
        self.compute_crps = compute_crps
        self.compute_fss = compute_fss
        self.crps_pool_type = crps_pool_type
        self.crps_scale = crps_scale
        self.crps_eps = crps_eps
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"MetricsAccumulator using device: {self.device}")

        self.thresholds_tensor = torch.tensor(
            self.thresholds, device=self.device
        ).float()

        self.mse_from_mean_sum = 0.0
        self.mse_from_mean_count = 0

        self.csi_from_mean_hits = {th_val: 0 for th_val in self.thresholds}
        self.csi_from_mean_misses = {th_val: 0 for th_val in self.thresholds}
        self.csi_from_mean_false_alarms = {th_val: 0 for th_val in self.thresholds}

        self.pixel_contingency_from_mean_hits = {
            th_val: 0 for th_val in self.thresholds
        }
        self.pixel_contingency_from_mean_misses = {
            th_val: 0 for th_val in self.thresholds
        }
        self.pixel_contingency_from_mean_false_alarms = {
            th_val: 0 for th_val in self.thresholds
        }
        self.pixel_contingency_from_mean_correct_negatives = {
            th_val: 0 for th_val in self.thresholds
        }

        self.csi_pooled_from_mean_hits = {th_val: 0 for th_val in self.thresholds}
        self.csi_pooled_from_mean_misses = {th_val: 0 for th_val in self.thresholds}
        self.csi_pooled_from_mean_false_alarms = {
            th_val: 0 for th_val in self.thresholds
        }

        self.crps_sum = 0.0
        self.crps_count = 0

        self.fss_from_mean_accumulators = {}
        if self.compute_fss:
            self.fss_scales = fss_scales if fss_scales is not None else [1, 16]
            self.fss_from_mean_accumulators = {}
            for th in self.thresholds:
                self.fss_from_mean_accumulators[th] = {}
                for scale in self.fss_scales:
                    self.fss_from_mean_accumulators[th][scale] = fss_init(th, scale)

    def update(self, y_true_chunk_np: np.ndarray, y_pred_chunk_np: np.ndarray):
        """
        Updates the metric accumulators with a new chunk of data.

        Args:
            y_true_chunk_np (np.ndarray): A numpy array of ground truth data with shape
                                          (b, T, H, W), where b is batch size, T is number
                                          of lead times, H is height, and W is width.
            y_pred_chunk_np (np.ndarray): A numpy array of prediction data with shape
                                          (b, n_samples, T, H, W), where n_samples is the
                                          number of ensemble members.
        """
        y_true_chunk = torch.from_numpy(y_true_chunk_np).float().to(self.device)
        y_pred_chunk = torch.from_numpy(y_pred_chunk_np).float().to(self.device)

        b, T, H, W = y_true_chunk.shape
        n_samples = y_pred_chunk.shape[1]

        y_pred_mean = torch.mean(y_pred_chunk, dim=1)

        if self.compute_mse:
            y_true_lead_mse_mean = y_true_chunk[:, self.lead_time, :, :]
            y_pred_lead_mse_mean = y_pred_mean[:, self.lead_time, :, :]
            is_nan_true_mean_mse = torch.isnan(y_true_lead_mse_mean)
            is_nan_pred_mean_mse = torch.isnan(y_pred_lead_mse_mean)
            valid_mask_mean_mse = ~torch.logical_or(
                is_nan_true_mean_mse, is_nan_pred_mean_mse
            )
            diff2_mean = (y_true_lead_mse_mean - y_pred_lead_mse_mean) ** 2
            self.mse_from_mean_sum += torch.sum(diff2_mean[valid_mask_mean_mse]).item()
            self.mse_from_mean_count += torch.sum(valid_mask_mean_mse).item()

        if self.compute_threshold:
            y_true_lead_continuous = y_true_chunk[:, self.lead_time, :, :]
            y_pred_lead_continuous_mean = y_pred_mean[:, self.lead_time, :, :]

            y_true_pooled_fm_for_threshold_loop = None
            y_pred_pooled_fm_for_threshold_loop = None
            if self.pool_size > 1:
                y_true_lead_for_pool_mean = y_true_lead_continuous.unsqueeze(1)
                y_pred_lead_for_pool_mean = y_pred_lead_continuous_mean.unsqueeze(1)

                y_true_pooled_fm_for_threshold_loop = F.max_pool2d(
                    y_true_lead_for_pool_mean,
                    kernel_size=self.pool_size,
                    stride=self.pool_size,
                ).squeeze(
                    1
                )
                y_pred_pooled_fm_for_threshold_loop = F.max_pool2d(
                    y_pred_lead_for_pool_mean,
                    kernel_size=self.pool_size,
                    stride=self.pool_size,
                ).squeeze(
                    1
                )

            for i, th_val_tensor in enumerate(self.thresholds_tensor):
                th_key_float = self.thresholds[i]
                y_true_bin_mean_pix = (y_true_lead_continuous > th_val_tensor).float()
                y_pred_bin_mean_pix = (
                    y_pred_lead_continuous_mean > th_val_tensor
                ).float()

                nan_mask_true_mean_pix = torch.isnan(y_true_lead_continuous)
                nan_mask_pred_mean_pix = torch.isnan(y_pred_lead_continuous_mean)
                invalid_mask_mean_pix = torch.logical_or(
                    nan_mask_true_mean_pix, nan_mask_pred_mean_pix
                )

                y_true_bin_mean_pix[invalid_mask_mean_pix] = (
                    0
                )
                y_pred_bin_mean_pix[invalid_mask_mean_pix] = (
                    0
                )

                current_chunk_hits_fm_pix = torch.sum(
                    (y_pred_bin_mean_pix == 1) & (y_true_bin_mean_pix == 1)
                ).item()
                current_chunk_misses_fm_pix = torch.sum(
                    (y_pred_bin_mean_pix == 0) & (y_true_bin_mean_pix == 1)
                ).item()
                current_chunk_fa_fm_pix = torch.sum(
                    (y_pred_bin_mean_pix == 1) & (y_true_bin_mean_pix == 0)
                ).item()
                current_chunk_cn_fm_pix = torch.sum(
                    (y_pred_bin_mean_pix == 0) & (y_true_bin_mean_pix == 0)
                ).item()

                self.csi_from_mean_hits[th_key_float] += current_chunk_hits_fm_pix
                self.csi_from_mean_misses[th_key_float] += current_chunk_misses_fm_pix
                self.csi_from_mean_false_alarms[th_key_float] += current_chunk_fa_fm_pix
                self.pixel_contingency_from_mean_hits[
                    th_key_float
                ] += current_chunk_hits_fm_pix
                self.pixel_contingency_from_mean_misses[
                    th_key_float
                ] += current_chunk_misses_fm_pix
                self.pixel_contingency_from_mean_false_alarms[
                    th_key_float
                ] += current_chunk_fa_fm_pix
                self.pixel_contingency_from_mean_correct_negatives[
                    th_key_float
                ] += current_chunk_cn_fm_pix

                if self.pool_size > 1:
                    y_true_event_pooled_fm = (
                        y_true_pooled_fm_for_threshold_loop > th_val_tensor
                    ).float()
                    y_pred_event_pooled_fm = (
                        y_pred_pooled_fm_for_threshold_loop > th_val_tensor
                    ).float()

                    pooled_nan_mask_true_fm = torch.isnan(
                        y_true_pooled_fm_for_threshold_loop
                    )
                    pooled_nan_mask_pred_fm = torch.isnan(
                        y_pred_pooled_fm_for_threshold_loop
                    )
                    pooled_invalid_mask_fm = torch.logical_or(
                        pooled_nan_mask_true_fm, pooled_nan_mask_pred_fm
                    )

                    y_true_event_pooled_fm[pooled_invalid_mask_fm] = 0
                    y_pred_event_pooled_fm[pooled_invalid_mask_fm] = 0

                    current_chunk_pooled_hits = torch.sum(
                        (y_pred_event_pooled_fm == 1) & (y_true_event_pooled_fm == 1)
                    ).item()
                    current_chunk_pooled_misses = torch.sum(
                        (y_pred_event_pooled_fm == 0) & (y_true_event_pooled_fm == 1)
                    ).item()
                    current_chunk_pooled_false_alarms = torch.sum(
                        (y_pred_event_pooled_fm == 1) & (y_true_event_pooled_fm == 0)
                    ).item()

                    self.csi_pooled_from_mean_hits[
                        th_key_float
                    ] += current_chunk_pooled_hits
                    self.csi_pooled_from_mean_misses[
                        th_key_float
                    ] += current_chunk_pooled_misses
                    self.csi_pooled_from_mean_false_alarms[
                        th_key_float
                    ] += current_chunk_pooled_false_alarms
                else:
                    self.csi_pooled_from_mean_hits[
                        th_key_float
                    ] += current_chunk_hits_fm_pix
                    self.csi_pooled_from_mean_misses[
                        th_key_float
                    ] += current_chunk_misses_fm_pix
                    self.csi_pooled_from_mean_false_alarms[
                        th_key_float
                    ] += current_chunk_fa_fm_pix

        if self.compute_fss:
            y_true_lead_fss_mean_np = (
                y_true_chunk[:, self.lead_time, :, :].cpu().numpy()
            )
            y_pred_lead_fss_mean_np = y_pred_mean[:, self.lead_time, :, :].cpu().numpy()

            for j in range(b):
                X_o = y_true_lead_fss_mean_np[j]
                X_f = y_pred_lead_fss_mean_np[j]

                for th_key in self.thresholds:
                    for scale in self.fss_scales:
                        fss_obj = self.fss_from_mean_accumulators[th_key][scale]
                        fss_accum(fss_obj, X_f, X_o)

        if self.compute_crps:
            try:
                y_true_for_crps = y_true_chunk.unsqueeze(2)
                y_pred_for_crps = y_pred_chunk.unsqueeze(2)

                crps_val_chunk = crps(
                    y_true_for_crps,
                    y_pred_for_crps,
                    pool_type=self.crps_pool_type,
                    scale=self.crps_scale,
                    mode="mean",
                    eps=self.crps_eps,
                )
            except Exception as e:
                print(f"CRPS computation failed for chunk: {e}")
                crps_val_chunk = np.nan

            if not np.isnan(crps_val_chunk):
                h_eff_crps = (
                    H // self.crps_scale
                    if self.crps_scale > 0 and self.crps_pool_type in ["avg", "max"]
                    else H
                )
                w_eff_crps = (
                    W // self.crps_scale
                    if self.crps_scale > 0 and self.crps_pool_type in ["avg", "max"]
                    else W
                )

                num_elements_in_chunk_crps_avg = b * T * h_eff_crps * w_eff_crps

                self.crps_sum += crps_val_chunk * num_elements_in_chunk_crps_avg
                self.crps_count += num_elements_in_chunk_crps_avg

    def compute(self):
        """
        Computes the final metrics from the accumulated values.

        Returns:
            Dict[str, Optional[object]]: A dictionary containing the computed metrics.
            Metrics are provided for both per-sample averages and for the ensemble mean.
            Includes metrics like MSE, CRPS, and various threshold-based
            scores (CSI, POD, FAR, HSS).
        """
        results: Dict[str, Optional[object]] = {}
        results["mse_from_mean"] = (
            self.mse_from_mean_sum / self.mse_from_mean_count
            if self.mse_from_mean_count > 0
            else np.nan
        )

        if self.compute_threshold:
            csi_from_mean_dict = {}
            pod_from_mean_dict = {}
            far_from_mean_dict = {}
            hss_from_mean_dict = {}
            csi_pooled_from_mean_dict = (
                {}
            ) 

            for th_val_float in self.thresholds:
                csi_fm_h = self.csi_from_mean_hits[th_val_float]
                csi_fm_m = self.csi_from_mean_misses[th_val_float]
                csi_fm_fa = self.csi_from_mean_false_alarms[th_val_float]
                csi_from_mean_denom = csi_fm_h + csi_fm_m + csi_fm_fa
                csi_from_mean_dict[th_val_float] = (
                    csi_fm_h / csi_from_mean_denom
                    if csi_from_mean_denom > 0
                    else np.nan
                )

                pix_fm_h = self.pixel_contingency_from_mean_hits[th_val_float]
                pix_fm_m = self.pixel_contingency_from_mean_misses[th_val_float]
                pix_fm_fa = self.pixel_contingency_from_mean_false_alarms[th_val_float]
                pix_fm_cn = self.pixel_contingency_from_mean_correct_negatives[
                    th_val_float
                ]

                pod_from_mean_dict[th_val_float] = (
                    pix_fm_h / (pix_fm_h + pix_fm_m)
                    if (pix_fm_h + pix_fm_m) > 0
                    else np.nan
                )
                far_from_mean_dict[th_val_float] = (
                    pix_fm_fa / (pix_fm_h + pix_fm_fa)
                    if (pix_fm_h + pix_fm_fa) > 0
                    else np.nan
                )

                hss_fm_num = 2 * (pix_fm_h * pix_fm_cn - pix_fm_m * pix_fm_fa)
                hss_fm_den1 = (pix_fm_h + pix_fm_m) * (pix_fm_m + pix_fm_cn)
                hss_fm_den2 = (pix_fm_h + pix_fm_fa) * (pix_fm_fa + pix_fm_cn)
                hss_fm_den = hss_fm_den1 + hss_fm_den2
                hss_from_mean_dict[th_val_float] = (
                    hss_fm_num / hss_fm_den if hss_fm_den != 0 else np.nan
                )

                csi_pfm_h = self.csi_pooled_from_mean_hits[th_val_float]
                csi_pfm_m = self.csi_pooled_from_mean_misses[th_val_float]
                csi_pfm_fa = self.csi_pooled_from_mean_false_alarms[th_val_float]
                csi_pfm_denom = csi_pfm_h + csi_pfm_m + csi_pfm_fa
                csi_pooled_from_mean_dict[th_val_float] = (
                    csi_pfm_h / csi_pfm_denom if csi_pfm_denom > 0 else np.nan
                )

            results["csi_from_mean"] = csi_from_mean_dict
            results["pod_from_mean"] = pod_from_mean_dict
            results["far_from_mean"] = far_from_mean_dict
            results["hss_from_mean"] = hss_from_mean_dict
            results["csi_pooled_from_mean"] = csi_pooled_from_mean_dict
        else:
            results["csi_from_mean"] = None
            results["pod_from_mean"] = None
            results["far_from_mean"] = None
            results["hss_from_mean"] = None
            results["csi_pooled_from_mean"] = None

        if self.compute_fss:
            fss_from_mean_dict = {}
            for th_key in self.thresholds:
                fss_from_mean_dict[th_key] = {}
                for scale in self.fss_scales:
                    fss_obj = self.fss_from_mean_accumulators[th_key][scale]
                    fss_from_mean_dict[th_key][scale] = fss_compute(fss_obj)
            results["fss_from_mean"] = fss_from_mean_dict
        else:
            results["fss_from_mean"] = None

        results["crps"] = (
            self.crps_sum / self.crps_count if self.crps_count > 0 else np.nan
        )
        return results
