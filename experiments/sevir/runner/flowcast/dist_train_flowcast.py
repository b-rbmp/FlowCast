"""
Distributed training script for the FlowCast model on the SEVIR dataset.

This script uses `torchrun` for multi-GPU/multi-node training. It trains a
Conditional Flow Matching (CFM) model with an Earthformer-style UNet backbone.
Training is configured via a YAML file and includes features like partial evaluation,
EMA checkpointing, early stopping, and WandB logging.
"""

import sys
import os
import copy
import argparse
import datetime
import random
import numpy as np
import wandb
import namegenerator
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from experiments.sevir.dataset.sevirfulldataset import (
    DynamicEncodedSequentialSevirDataset,
    dynamic_encoded_sequential_collate,
    DynamicSequentialSevirDataset,
    dynamic_sequential_collate,
    post_process_samples,
)
from common.utils.utils import EarlyStopping, compute_mean_std
from common.models.flowcast.cuboid_transformer_unet import (
    CuboidTransformerUNet,
)
from omegaconf import OmegaConf
from common.cfm.cfm import (
    ConditionalFlowMatcher,
)
from common.utils.utils import warmup_lambda

from common.metrics.metrics_streaming_probabilistic import (
    MetricsAccumulator,
)
from common.utils.utils import (
    calculate_metrics,
    ema,
)
from experiments.sevir.display.cartopy import make_animation
from torchdiffeq import odeint_adjoint as odeint


def setup_ddp():
    """Initializes the distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Initializing DDP: Rank {rank}/{world_size}, Local Rank {local_rank}")
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        dist.barrier()
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    else:
        print("Not running in distributed mode. Using single device.")
        return (
            0,
            1,
            0,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )


def cleanup_ddp():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Cleaned up DDP.")


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduces a tensor's value across all DDP processes by averaging.

    Args:
        tensor (torch.Tensor): The tensor to reduce.
        world_size (int): The total number of processes.

    Returns:
        torch.Tensor: The reduced tensor with the averaged value.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

parser = argparse.ArgumentParser(description="Script for configuring hyperparameters.")

parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/runner/flowcast/flowcast_config.yaml",
)
parser.add_argument(
    "--train_file",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_training_full.h5",
)
parser.add_argument(
    "--train_meta",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_training_full_META.csv",
)
parser.add_argument(
    "--val_file",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_validation_full.h5",
)
parser.add_argument(
    "--val_meta",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_validation_full_META.csv",
)

parser.add_argument(
    "--partial_evaluation_file",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_validation_full.h5",
)
parser.add_argument(
    "--partial_evaluation_meta",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_validation_full_META.csv",
)


def main():
    """
    Main function to run the distributed training and validation loop.

    Orchestrates the entire process, from DDP setup and configuration loading
    to the training loop, evaluation, and final cleanup.
    """
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    rank, world_size, local_rank, device = setup_ddp()
    is_main_process = rank == 0

    DEBUG_MODE = config.run_params.debug_mode
    RUN_STRING = config.run_params.run_string
    CARTOPY_FEATURES = config.partial_evaluation_params.cartopy_features

    run_id_base = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + RUN_STRING
        + "_"
        + namegenerator.gen()
    )
    MAIN_RUN_ID = f"{run_id_base}_main"

    DEBUG_PRINT_PREFIX = (
        f"[DEBUG Rank {rank}] " if DEBUG_MODE else f"[Rank {rank}] "
    )

    ENABLE_WANDB = (
        config.run_params.enable_wandb and is_main_process
    )
    PARTIAL_EVALUATION = config.partial_evaluation_params.partial_evaluation
    PARTIAL_EVALUATION_INTERVAL = (
        config.partial_evaluation_params.partial_evaluation_interval
    )
    PARTIAL_EVALUATION_BATCHES = (
        config.partial_evaluation_params.partial_evaluation_batches
    )
    AUTOENCODER_CHECKPOINT = config.autoencoder_params.autoencoder_checkpoint
    THRESHOLDS = np.array([16, 74, 133, 160, 181, 219], dtype=np.float32)

    if (
        PARTIAL_EVALUATION and AUTOENCODER_CHECKPOINT is None and is_main_process
    ):
        raise ValueError(
            "Partial Evaluation is enabled but Autoencoder Checkpoint is not provided"
        )

    TRAIN_FILE = args.train_file
    TRAIN_META = args.train_meta
    VAL_FILE = args.val_file
    VAL_META = args.val_meta
    PRELOAD_MODEL = config.run_params.preload_model
    BATCH_SIZE = config.training_params.micro_batch_size
    LEARNING_RATE = config.optimizer_params.learning_rate
    NUM_EPOCHS = config.training_params.num_epochs
    NUM_WORKERS = config.training_params.num_workers
    EARLY_STOPPING_PATIENCE = config.training_params.early_stopping_patience
    EARLY_STOPPING_METRIC = config.training_params.early_stopping_metric
    LAG_TIME = config.data_params.lag_time
    LEAD_TIME = config.data_params.lead_time
    TIME_SPACING = config.data_params.time_spacing
    GRAD_CLIP = config.training_params.gradient_clip_val
    FLOW_MATCHING_METHOD = config.flow_matching_params.flow_matching_method

    if (
        EARLY_STOPPING_METRIC in ["partial_csi_m", "partial_mse"]
        and not PARTIAL_EVALUATION
        and is_main_process
    ):
        raise ValueError(
            f"Early stopping metric {EARLY_STOPPING_METRIC} requires partial evaluation to be enabled"
        )

    OPTIMIZER_TYPE = config.optimizer_params.optimizer_type
    WEIGHT_DECAY = config.optimizer_params.weight_decay

    SCHEDULER_TYPE = config.scheduler_params.scheduler_type
    LR_PLATEAU_FACTOR = config.scheduler_params.lr_plateau_factor
    LR_PLATEAU_PATIENCE = config.scheduler_params.lr_plateau_patience
    LR_COSINE_WARMUP_ITER_PERCENTAGE = (
        config.scheduler_params.lr_cosine_warmup_iter_percentage
    )
    LR_COSINE_MIN_WARMUP_LR_RATIO = (
        config.scheduler_params.lr_cosine_min_warmup_lr_ratio
    )
    LR_COSINE_MIN_LR_RATIO = config.scheduler_params.lr_cosine_min_lr_ratio

    NORMALIZED_AUTOENCODER = config.autoencoder_params.normalized_autoencoder

    USE_FP16 = config.training_params.fp16

    EMA_MODEL_SAVING = config.ema_model_saving_params.ema_model_saving
    EMA_MODEL_SAVING_DECAY = config.ema_model_saving_params.ema_model_saving_decay

    GRAD_ACCUMULATION_STEPS = (
        config.training_params.grad_accumulation_steps
    )

    model_config = OmegaConf.to_object(config.latent_model)

    BASE_UNITS = model_config["base_units"]
    SCALE_ALPHA = model_config["scale_alpha"]
    NUM_HEADS = model_config["num_heads"]
    ATTN_DROP = model_config["attn_drop"]
    PROJ_DROP = model_config["proj_drop"]
    FFN_DROP = model_config["ffn_drop"]
    DOWNSAMPLE = model_config["downsample"]
    DOWNSAMPLE_TYPE = model_config["downsample_type"]
    UPSAMPLE_TYPE = model_config["upsample_type"]
    UPSAMPLE_KERNEL_SIZE = model_config["upsample_kernel_size"]
    DEPTH = model_config["depth"]
    BLOCK_ATTN_PATTERNS = [model_config["self_pattern"]] * len(DEPTH)
    NUM_GLOBAL_VECTORS = model_config["num_global_vectors"]
    USE_GLOBAL_VECTOR_FFN = model_config["use_global_vector_ffn"]
    USE_GLOBAL_SELF_ATTN = model_config["use_global_self_attn"]
    SEPARATE_GLOBAL_QKV = model_config["separate_global_qkv"]
    GLOBAL_DIM_RATIO = model_config["global_dim_ratio"]
    SELF_PATTERN = model_config["self_pattern"]
    FFN_ACTIVATION = model_config["ffn_activation"]
    GATED_FFN = model_config["gated_ffn"]
    NORM_LAYER = model_config["norm_layer"]
    PADDING_TYPE = model_config["padding_type"]
    CHECKPOINT_LEVEL = model_config["checkpoint_level"]
    POS_EMBED_TYPE = model_config["pos_embed_type"]
    USE_RELATIVE_POS = model_config["use_relative_pos"]
    SELF_ATTN_USE_FINAL_PROJ = model_config["self_attn_use_final_proj"]
    ATTN_LINEAR_INIT_MODE = model_config["attn_linear_init_mode"]
    FFN_LINEAR_INIT_MODE = model_config["ffn_linear_init_mode"]
    FFN2_LINEAR_INIT_MODE = model_config["ffn2_linear_init_mode"]
    ATTN_PROJ_LINEAR_INIT_MODE = model_config["attn_proj_linear_init_mode"]
    CONV_INIT_MODE = model_config["conv_init_mode"]
    DOWN_UP_LINEAR_INIT_MODE = model_config["down_up_linear_init_mode"]
    GLOBAL_PROJ_LINEAR_INIT_MODE = model_config["global_proj_linear_init_mode"]
    NORM_INIT_MODE = model_config["norm_init_mode"]
    TIME_EMBED_CHANNELS_MULT = model_config["time_embed_channels_mult"]
    TIME_EMBED_USE_SCALE_SHIFT_NORM = model_config["time_embed_use_scale_shift_norm"]
    TIME_EMBED_DROPOUT = model_config["time_embed_dropout"]
    UNET_RES_CONNECT = model_config["unet_res_connect"]

    if is_main_process:
        print(f"--- Distributed Training Config ---")
        print(f"World Size: {world_size}")
        print(f"Batch Size PER GPU: {BATCH_SIZE}")
        print(f"Global Batch Size (before accumulation): {BATCH_SIZE * world_size}")
        print(f"Gradient Accumulation Steps: {GRAD_ACCUMULATION_STEPS}")
        print(
            f"Effective Global Batch Size: {BATCH_SIZE * world_size * GRAD_ACCUMULATION_STEPS}"
        )
        print(f"FP16 Enabled: {USE_FP16}")
        print(f"-----------------------------------")
        print(f"{DEBUG_PRINT_PREFIX}Run ID (Main): {MAIN_RUN_ID}")
        print(f"{DEBUG_PRINT_PREFIX}Run String: {RUN_STRING}")
        print(f"{DEBUG_PRINT_PREFIX}Training File: {TRAIN_FILE}")
        print(f"{DEBUG_PRINT_PREFIX}Training Meta: {TRAIN_META}")
        print(f"{DEBUG_PRINT_PREFIX}Validation File: {VAL_FILE}")
        print(f"{DEBUG_PRINT_PREFIX}Validation Meta: {VAL_META}")
        print(f"{DEBUG_PRINT_PREFIX}Debug Mode: {DEBUG_MODE}")
        print(f"{DEBUG_PRINT_PREFIX}Normalized Autoencoder: {NORMALIZED_AUTOENCODER}")
        print(
            f"{DEBUG_PRINT_PREFIX}Enable Wandb: {config.run_params.enable_wandb}"
        )
        print(f"{DEBUG_PRINT_PREFIX}Partial Evaluation: {PARTIAL_EVALUATION}")
        print(
            f"{DEBUG_PRINT_PREFIX}Partial Evaluation Interval: {PARTIAL_EVALUATION_INTERVAL}"
        )
        print(
            f"{DEBUG_PRINT_PREFIX}Partial Evaluation Batches: {PARTIAL_EVALUATION_BATCHES}"
        )
        print(f"{DEBUG_PRINT_PREFIX}Autoencoder Checkpoint: {AUTOENCODER_CHECKPOINT}")
        print(f"{DEBUG_PRINT_PREFIX}Training File: {TRAIN_FILE}")
        print(f"{DEBUG_PRINT_PREFIX}Training Meta: {TRAIN_META}")
        print(f"{DEBUG_PRINT_PREFIX}Preload Model: {PRELOAD_MODEL}")
        print(f"{DEBUG_PRINT_PREFIX}Learning Rate: {LEARNING_RATE}")
        print(f"{DEBUG_PRINT_PREFIX}Number of Epochs: {NUM_EPOCHS}")
        print(f"{DEBUG_PRINT_PREFIX}Number of Workers: {NUM_WORKERS}")
        print(f"{DEBUG_PRINT_PREFIX}Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
        print(f"{DEBUG_PRINT_PREFIX}Early Stopping Metric: {EARLY_STOPPING_METRIC}")
        print(f"{DEBUG_PRINT_PREFIX}Lag Time: {LAG_TIME}")
        print(f"{DEBUG_PRINT_PREFIX}Lead Time: {LEAD_TIME}")
        print(f"{DEBUG_PRINT_PREFIX}Time Spacing: {TIME_SPACING}")
        print(f"{DEBUG_PRINT_PREFIX}Gradient Clip Value: {GRAD_CLIP}")
        print(f"{DEBUG_PRINT_PREFIX}Flow Matching Method: {FLOW_MATCHING_METHOD}")

        print(f"--------- {DEBUG_PRINT_PREFIX}Flowcast Config ---------")
        print(f"{DEBUG_PRINT_PREFIX}Base Units: {BASE_UNITS}")
        print(f"{DEBUG_PRINT_PREFIX}Scale Alpha: {SCALE_ALPHA}")
        print(f"{DEBUG_PRINT_PREFIX}Depth: {DEPTH}")
        print(f"{DEBUG_PRINT_PREFIX}Block Attn Patterns: {BLOCK_ATTN_PATTERNS}")

        print(f"{DEBUG_PRINT_PREFIX}Downsample: {DOWNSAMPLE}")
        print(f"{DEBUG_PRINT_PREFIX}Downsample Type: {DOWNSAMPLE_TYPE}")
        print(f"{DEBUG_PRINT_PREFIX}Upsample Type: {UPSAMPLE_TYPE}")
        print(f"{DEBUG_PRINT_PREFIX}Num Global Vectors: {NUM_GLOBAL_VECTORS}")
        print(
            f"{DEBUG_PRINT_PREFIX}ATTN_PROJ_LINEAR_INIT_MODE: {ATTN_PROJ_LINEAR_INIT_MODE}"
        )
        print(
            f"{DEBUG_PRINT_PREFIX}Global Proj Linear Init Mode: {GLOBAL_PROJ_LINEAR_INIT_MODE}"
        )
        print(f"{DEBUG_PRINT_PREFIX}Use Global Vector FFN: {USE_GLOBAL_VECTOR_FFN}")
        print(f"{DEBUG_PRINT_PREFIX}Use Global Self Attn: {USE_GLOBAL_SELF_ATTN}")
        print(f"{DEBUG_PRINT_PREFIX}Separate Global QKV: {SEPARATE_GLOBAL_QKV}")
        print(f"{DEBUG_PRINT_PREFIX}Global Dim Ratio: {GLOBAL_DIM_RATIO}")
        print(f"{DEBUG_PRINT_PREFIX}Self Pattern: {SELF_PATTERN}")
        print(f"{DEBUG_PRINT_PREFIX}Attn Drop: {ATTN_DROP}")
        print(f"{DEBUG_PRINT_PREFIX}Proj Drop: {PROJ_DROP}")
        print(f"{DEBUG_PRINT_PREFIX}FFN Drop: {FFN_DROP}")
        print(f"{DEBUG_PRINT_PREFIX}Num Heads: {NUM_HEADS}")
        print(f"{DEBUG_PRINT_PREFIX}FFN Activation: {FFN_ACTIVATION}")
        print(f"{DEBUG_PRINT_PREFIX}Gated FFN: {GATED_FFN}")
        print(f"{DEBUG_PRINT_PREFIX}Norm Layer: {NORM_LAYER}")
        print(f"{DEBUG_PRINT_PREFIX}Padding Type: {PADDING_TYPE}")
        print(f"{DEBUG_PRINT_PREFIX}Pos Embed Type: {POS_EMBED_TYPE}")
        print(f"{DEBUG_PRINT_PREFIX}Use Relative Pos: {USE_RELATIVE_POS}")
        print(
            f"{DEBUG_PRINT_PREFIX}Self Attn Use Final Proj: {SELF_ATTN_USE_FINAL_PROJ}"
        )
        print(f"{DEBUG_PRINT_PREFIX}Checkpoint Level: {CHECKPOINT_LEVEL}")
        print(f"{DEBUG_PRINT_PREFIX}Attn Linear Init Mode: {ATTN_LINEAR_INIT_MODE}")
        print(f"{DEBUG_PRINT_PREFIX}FFN Linear Init Mode: {FFN_LINEAR_INIT_MODE}")
        print(f"{DEBUG_PRINT_PREFIX}Conv Init Mode: {CONV_INIT_MODE}")
        print(
            f"{DEBUG_PRINT_PREFIX}Down Up Linear Init Mode: {DOWN_UP_LINEAR_INIT_MODE}"
        )
        print(f"{DEBUG_PRINT_PREFIX}Norm Init Mode: {NORM_INIT_MODE}")
        print(
            f"{DEBUG_PRINT_PREFIX}Time Embed Channels Mult: {TIME_EMBED_CHANNELS_MULT}"
        )
        print(
            f"{DEBUG_PRINT_PREFIX}Time Embed Use Scale Shift Norm: {TIME_EMBED_USE_SCALE_SHIFT_NORM}"
        )
        print(f"{DEBUG_PRINT_PREFIX}Time Embed Dropout: {TIME_EMBED_DROPOUT}")
        print(f"{DEBUG_PRINT_PREFIX}UNET Res Connect: {UNET_RES_CONNECT}")

        print(f"--------- {DEBUG_PRINT_PREFIX}Optimizer Config ---------")
        print(f"{DEBUG_PRINT_PREFIX}Optimizer Type: {OPTIMIZER_TYPE}")
        print(f"{DEBUG_PRINT_PREFIX}Weight Decay: {WEIGHT_DECAY}")
        print(f"{DEBUG_PRINT_PREFIX}Scheduler Type: {SCHEDULER_TYPE}")
        print(f"{DEBUG_PRINT_PREFIX}LR Plateau Factor: {LR_PLATEAU_FACTOR}")
        print(f"{DEBUG_PRINT_PREFIX}LR Plateau Patience: {LR_PLATEAU_PATIENCE}")
        print(
            f"{DEBUG_PRINT_PREFIX}LR Cosine Warmup Iter Percentage: {LR_COSINE_WARMUP_ITER_PERCENTAGE}"
        )
        print(
            f"{DEBUG_PRINT_PREFIX}LR Cosine Min Warmup LR Ratio: {LR_COSINE_MIN_WARMUP_LR_RATIO}"
        )
        print(f"{DEBUG_PRINT_PREFIX}LR Cosine Min LR Ratio: {LR_COSINE_MIN_LR_RATIO}")

        print(f"--------- {DEBUG_PRINT_PREFIX}EMA Model Saving Config ---------")
        print(f"{DEBUG_PRINT_PREFIX}EMA Model Saving: {EMA_MODEL_SAVING}")
        if EMA_MODEL_SAVING:
            print(
                f"{DEBUG_PRINT_PREFIX}EMA Model Saving Decay: {EMA_MODEL_SAVING_DECAY}"
            )
        print(f"------------------------------------------------------------")

    project_name = "sevir-nowcasting-cfm"
    if ENABLE_WANDB:
        config_dict = {
            "learning_rate": LEARNING_RATE,
            "batch_size_per_gpu": BATCH_SIZE,
            "world_size": world_size,
            "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * world_size * GRAD_ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS,
            "num_workers": NUM_WORKERS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "lag_time": LAG_TIME,
            "lead_time": LEAD_TIME,
            "time_spacing": TIME_SPACING,
            "grad_clip": GRAD_CLIP,
            "flow_matching_method": FLOW_MATCHING_METHOD,
            "dataset": "SEVIR",
            "model": "Flowcast-CFM",
            "optimizer_type": OPTIMIZER_TYPE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler_type": SCHEDULER_TYPE,
            "lr_plateau_factor": LR_PLATEAU_FACTOR,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE,
            "lr_cosine_warmup_iter_percentage": LR_COSINE_WARMUP_ITER_PERCENTAGE,
            "lr_cosine_min_warmup_lr_ratio": LR_COSINE_MIN_WARMUP_LR_RATIO,
            "lr_cosine_min_lr_ratio": LR_COSINE_MIN_LR_RATIO,
            "base_units": BASE_UNITS,
            "scale_alpha": SCALE_ALPHA,
            "depth": DEPTH,
            "block_attn_patterns": BLOCK_ATTN_PATTERNS,
            "downsample": DOWNSAMPLE,
            "downsample_type": DOWNSAMPLE_TYPE,
            "upsample_type": UPSAMPLE_TYPE,
            "num_global_vectors": NUM_GLOBAL_VECTORS,
            "use_global_vector_ffn": USE_GLOBAL_VECTOR_FFN,
            "use_global_self_attn": USE_GLOBAL_SELF_ATTN,
            "separate_global_qkv": SEPARATE_GLOBAL_QKV,
            "global_dim_ratio": GLOBAL_DIM_RATIO,
            "self_pattern": SELF_PATTERN,
            "attn_drop": ATTN_DROP,
            "proj_drop": PROJ_DROP,
            "ffn_drop": FFN_DROP,
            "num_heads": NUM_HEADS,
            "ffn_activation": FFN_ACTIVATION,
            "gated_ffn": GATED_FFN,
            "norm_layer": NORM_LAYER,
            "padding_type": PADDING_TYPE,
            "pos_embed_type": POS_EMBED_TYPE,
            "use_relative_pos": USE_RELATIVE_POS,
            "self_attn_use_final_proj": SELF_ATTN_USE_FINAL_PROJ,
            "checkpoint_level": CHECKPOINT_LEVEL,
            "attn_linear_init_mode": ATTN_LINEAR_INIT_MODE,
            "ffn_linear_init_mode": FFN_LINEAR_INIT_MODE,
            "conv_init_mode": CONV_INIT_MODE,
            "down_up_linear_init_mode": DOWN_UP_LINEAR_INIT_MODE,
            "norm_init_mode": NORM_INIT_MODE,
            "fp16": USE_FP16,
            "ema_model_saving": EMA_MODEL_SAVING,
            "ema_model_saving_decay": EMA_MODEL_SAVING_DECAY,
        }
        wandb.init(
            project=project_name,
            name=MAIN_RUN_ID,
            config=config_dict,
        )

    ARTIFACTS_FOLDER = f"artifacts/sevir/flowcast/{MAIN_RUN_ID}"
    PLOTS_FOLDER = f"{ARTIFACTS_FOLDER}/plots"
    ANIMATIONS_FOLDER = f"{PLOTS_FOLDER}/animations"
    METRICS_FOLDER = f"{PLOTS_FOLDER}/metrics"
    MODEL_SAVE_DIR = f"{ARTIFACTS_FOLDER}/models"
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "early_stopping_model" + ".pt")

    if is_main_process:
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        os.makedirs(ANIMATIONS_FOLDER, exist_ok=True)
        os.makedirs(METRICS_FOLDER, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    print(f"{DEBUG_PRINT_PREFIX}Using device: {device}")
    if device.type == "cpu" and is_main_process:
        print(DEBUG_PRINT_PREFIX + "Warning: CPU is used for computation!")

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_dataset = DynamicEncodedSequentialSevirDataset(
        meta_csv=TRAIN_META,
        data_file=TRAIN_FILE,
        data_type="vil",
        raw_seq_len=49,
        lag_time=LAG_TIME,
        lead_time=LEAD_TIME,
        time_spacing=TIME_SPACING,
        stride=12,
        channel_last=True,
        debug_mode=DEBUG_MODE,
        transform=None,
    )
    val_dataset = DynamicEncodedSequentialSevirDataset(
        meta_csv=VAL_META,
        data_file=VAL_FILE,
        data_type="vil",
        raw_seq_len=49,
        lag_time=LAG_TIME,
        lead_time=LEAD_TIME,
        time_spacing=TIME_SPACING,
        stride=12,
        channel_last=True,
        debug_mode=DEBUG_MODE,
        transform=None,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_encoded_sequential_collate,
        num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
        pin_memory=True if not DEBUG_MODE else False,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True if not DEBUG_MODE else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_encoded_sequential_collate,
        num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
        pin_memory=True if not DEBUG_MODE else False,
        sampler=val_sampler,
        drop_last=False,
        persistent_workers=True if not DEBUG_MODE else False,
    )

    input_shape = None
    output_shape = None
    if is_main_process:
        temp_loader = (
            DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=dynamic_encoded_sequential_collate,
                num_workers=0,
            )
        )
        for batch in temp_loader:
            inputs_cpu, outputs_cpu, _ = batch
            print(f"Inputs shape (from rank 0): {inputs_cpu.shape}")
            print(f"Outputs shape (from rank 0): {outputs_cpu.shape}")
            input_shape = inputs_cpu.shape
            output_shape = outputs_cpu.shape
            break
        del temp_loader

    shapes_list = [input_shape, output_shape]
    if world_size > 1:
        dist.broadcast_object_list(shapes_list, src=0)
    input_shape, output_shape = shapes_list[0], shapes_list[1]

    if input_shape is None or output_shape is None:
        raise RuntimeError("Could not determine input/output shapes.")

    preload_model_state_dict = None
    preload_global_step = None
    preload_best_val_loss = None
    preload_std = None
    preload_mean = None
    preload_optimizer_state_dict = None
    mean = None
    std = None

    if PRELOAD_MODEL is not None:
        if is_main_process:
            print(f"{DEBUG_PRINT_PREFIX}Attempting to load checkpoint: {PRELOAD_MODEL}")
            try:
                model_info = torch.load(PRELOAD_MODEL, map_location="cpu")
                preload_model_state_dict = model_info["model_state_dict"]
                preload_global_step = model_info.get(
                    "global_step", 0
                )
                preload_best_val_loss = model_info.get("best_metric", None)
                preload_std = model_info.get("std", None)
                preload_mean = model_info.get("mean", None)
                preload_optimizer_state_dict = model_info.get(
                    "optimizer_state_dict", None
                )
                print(
                    f"{DEBUG_PRINT_PREFIX}Successfully loaded model info from checkpoint"
                )
            except FileNotFoundError:
                print(
                    f"{DEBUG_PRINT_PREFIX}Preload model file not found: {PRELOAD_MODEL}. Starting from scratch."
                )
            except Exception as e:
                print(
                    f"{DEBUG_PRINT_PREFIX}Error loading checkpoint: {e}. Starting from scratch."
                )
        loaded_info = [
            preload_model_state_dict,
            preload_global_step,
            preload_best_val_loss,
            preload_std,
            preload_mean,
            preload_optimizer_state_dict,
        ]
        if world_size > 1:
            dist.broadcast_object_list(loaded_info, src=0)
        (
            preload_model_state_dict,
            preload_global_step,
            preload_best_val_loss,
            preload_std,
            preload_mean,
            preload_optimizer_state_dict,
        ) = loaded_info
    else:
        if is_main_process:
            print(f"{DEBUG_PRINT_PREFIX}No preload model specified.")

    if preload_mean is None or preload_std is None:
        if is_main_process:
            print(f"{DEBUG_PRINT_PREFIX}Computing mean and std...")
            temp_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE * 4,
                shuffle=False,
                collate_fn=dynamic_encoded_sequential_collate,
                num_workers=NUM_WORKERS // 2 if NUM_WORKERS > 1 else 0,
                pin_memory=False,
            )
            mean, std = compute_mean_std(temp_loader, channel_last=True)
            del temp_loader
            print(f"{DEBUG_PRINT_PREFIX}Computed Mean: {mean}")
            print(f"{DEBUG_PRINT_PREFIX}Computed Std: {std}")
        mean_std_list = [mean, std]
        if world_size > 1:
            dist.broadcast_object_list(mean_std_list, src=0)
        mean, std = mean_std_list[0], mean_std_list[1]
        if mean is None or std is None:
            raise RuntimeError("Mean/Std computation/broadcast failed.")
    else:
        raise RuntimeError("To be implemented.")

    IN_TIMESTEPS = input_shape[1]
    OUTPUT_TIMESTEPS = output_shape[1]

    input_shape_flowcast = (
        IN_TIMESTEPS,
        input_shape[2],
        input_shape[3],
        input_shape[4],
    )
    output_shape_flowcast = (
        OUTPUT_TIMESTEPS,
        output_shape[2],
        output_shape[3],
        output_shape[4],
    )

    model = CuboidTransformerUNet(
        input_shape=input_shape_flowcast,
        target_shape=output_shape_flowcast,
        base_units=BASE_UNITS,
        block_units=None,
        scale_alpha=SCALE_ALPHA,
        num_heads=NUM_HEADS,
        attn_drop=ATTN_DROP,
        proj_drop=PROJ_DROP,
        ffn_drop=FFN_DROP,
        downsample=DOWNSAMPLE,
        downsample_type=DOWNSAMPLE_TYPE,
        upsample_type=UPSAMPLE_TYPE,
        upsample_kernel_size=UPSAMPLE_KERNEL_SIZE,
        depth=DEPTH,
        block_attn_patterns=BLOCK_ATTN_PATTERNS,
        num_global_vectors=NUM_GLOBAL_VECTORS,
        use_global_vector_ffn=USE_GLOBAL_VECTOR_FFN,
        use_global_self_attn=USE_GLOBAL_SELF_ATTN,
        separate_global_qkv=SEPARATE_GLOBAL_QKV,
        global_dim_ratio=GLOBAL_DIM_RATIO,
        ffn_activation=FFN_ACTIVATION,
        gated_ffn=GATED_FFN,
        norm_layer=NORM_LAYER,
        padding_type=PADDING_TYPE,
        checkpoint_level=CHECKPOINT_LEVEL,
        pos_embed_type=POS_EMBED_TYPE,
        use_relative_pos=USE_RELATIVE_POS,
        self_attn_use_final_proj=SELF_ATTN_USE_FINAL_PROJ,
        attn_linear_init_mode=ATTN_LINEAR_INIT_MODE,
        ffn_linear_init_mode=FFN_LINEAR_INIT_MODE,
        ffn2_linear_init_mode=FFN2_LINEAR_INIT_MODE,
        attn_proj_linear_init_mode=ATTN_PROJ_LINEAR_INIT_MODE,
        conv_init_mode=CONV_INIT_MODE,
        down_linear_init_mode=DOWN_UP_LINEAR_INIT_MODE,
        up_linear_init_mode=DOWN_UP_LINEAR_INIT_MODE,
        global_proj_linear_init_mode=GLOBAL_PROJ_LINEAR_INIT_MODE,
        norm_init_mode=NORM_INIT_MODE,
        time_embed_channels_mult=TIME_EMBED_CHANNELS_MULT,
        time_embed_use_scale_shift_norm=TIME_EMBED_USE_SCALE_SHIFT_NORM,
        time_embed_dropout=TIME_EMBED_DROPOUT,
        unet_res_connect=UNET_RES_CONNECT,
        mean=mean,
        std=std,
    )

    if preload_model_state_dict is not None:
        try:
            model.load_state_dict(preload_model_state_dict)
            if is_main_process:
                print(
                    f"{DEBUG_PRINT_PREFIX}Successfully loaded pre-trained model state dict."
                )
        except Exception as e:
            if is_main_process:
                print(
                    f"{DEBUG_PRINT_PREFIX}Error loading pre-trained model state dict: {e}. Model weights might be random."
                )

    model = model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if is_main_process:
            print(f"{DEBUG_PRINT_PREFIX}Wrapped model with DDP.")

    ema_model = None
    if EMA_MODEL_SAVING:
        ema_model = copy.deepcopy(
            model.module if world_size > 1 else model
        )
        ema_model = ema_model.to(device)
        if is_main_process:
            print(f"{DEBUG_PRINT_PREFIX}Created EMA model.")

    num_batches_per_epoch = len(train_loader)
    total_num_steps = int(NUM_EPOCHS * num_batches_per_epoch)
    if is_main_process:
        print(f"{DEBUG_PRINT_PREFIX}Batches per epoch per GPU: {num_batches_per_epoch}")
        print(f"{DEBUG_PRINT_PREFIX}Total training steps: {total_num_steps}")

    if OPTIMIZER_TYPE == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_TYPE == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    else:
        if is_main_process:
            raise ValueError(f"Invalid optimizer type: {OPTIMIZER_TYPE}")
        else:
            dist.barrier()
            cleanup_ddp()
            sys.exit(1)

    if preload_optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(preload_optimizer_state_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if is_main_process:
                print(
                    f"{DEBUG_PRINT_PREFIX}Successfully loaded pre-trained optimizer state dict."
                )
        except Exception as e:
            if is_main_process:
                print(
                    f"{DEBUG_PRINT_PREFIX}Error loading pre-trained optimizer state dict: {e}. Optimizer state reset."
                )

    warmup_iter = int(np.round(LR_COSINE_WARMUP_ITER_PERCENTAGE * total_num_steps))

    if SCHEDULER_TYPE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_PLATEAU_FACTOR,
            patience=LR_PLATEAU_PATIENCE,
        )
    elif SCHEDULER_TYPE == "cosine":
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda(
                warmup_steps=warmup_iter, min_lr_ratio=LR_COSINE_MIN_WARMUP_LR_RATIO
            ),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(total_num_steps - warmup_iter),
            eta_min=LR_COSINE_MIN_LR_RATIO * LEARNING_RATE,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iter],
        )
    else:
        if is_main_process:
            raise ValueError(f"Invalid scheduler type: {SCHEDULER_TYPE}")
        else:
            dist.barrier()
            cleanup_ddp()
            sys.exit(1)

    criterion = nn.MSELoss(
        reduction="none"
    )

    sigma = config.flow_matching_params.sigma
    if FLOW_MATCHING_METHOD == "vanilla":
        flow_matcher = ConditionalFlowMatcher(sigma=sigma)
    else:
        if is_main_process:
            raise ValueError(f"Invalid flow matching method: {FLOW_MATCHING_METHOD}")
        else:
            dist.barrier()
            cleanup_ddp()
            sys.exit(1)

    ae_model = None
    val_sample_loader = (
        None
    )
    if is_main_process and PARTIAL_EVALUATION:
        if not os.path.exists(AUTOENCODER_CHECKPOINT):
            raise FileNotFoundError(
                f"[Rank 0] AE Model not found at {AUTOENCODER_CHECKPOINT}"
            )

        print(f"{DEBUG_PRINT_PREFIX}Loading Autoencoder for evaluation...")

        from diffusers.models.autoencoders import AutoencoderKL

        ae_model = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            down_block_types=config.autoencoder_params.down_block_types,
            up_block_types=config.autoencoder_params.up_block_types,
            block_out_channels=config.autoencoder_params.block_out_channels,
            act_fn=config.autoencoder_params.act_fn,
            latent_channels=config.autoencoder_params.latent_channels,
            norm_num_groups=config.autoencoder_params.norm_num_groups,
            layers_per_block=config.autoencoder_params.layers_per_block,
        )

        checkpoint = torch.load(
            AUTOENCODER_CHECKPOINT, map_location=device
        )
        # Remove 'module.' prefix if it exists (in case the AE was trained with DDP)
        new_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        ae_model.load_state_dict(new_state_dict)
        ae_model = ae_model.to(device)
        ae_model.eval()
        print(f"{DEBUG_PRINT_PREFIX}Autoencoder loaded successfully.")

        VAL_SAMPLE_FILE = args.partial_evaluation_file
        VAL_SAMPLE_META = args.partial_evaluation_meta

        val_sample_dataset = DynamicSequentialSevirDataset(
            meta_csv=VAL_SAMPLE_META,
            data_file=VAL_SAMPLE_FILE,
            data_type="vil",
            raw_seq_len=49,
            lag_time=LAG_TIME,
            lead_time=LEAD_TIME,
            time_spacing=TIME_SPACING,
            stride=12,
            channel_last=False,
            debug_mode=DEBUG_MODE,
        )

        val_sample_loader = DataLoader(
            val_sample_dataset,
            batch_size=(
                BATCH_SIZE // 4 if BATCH_SIZE > 4 else BATCH_SIZE
            ),
            shuffle=False,
            collate_fn=dynamic_sequential_collate,
            num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
            pin_memory=True if not DEBUG_MODE else False,
        )
        print(f"{DEBUG_PRINT_PREFIX}Test loader created for partial evaluation.")

    early_stopping = None
    best_val_loss_init = None

    if EARLY_STOPPING_METRIC == "val_loss":
        best_val_loss_init = (
            float("inf") if preload_best_val_loss is None else preload_best_val_loss
        )
        metric_direction = "minimize"
    elif EARLY_STOPPING_METRIC in ["partial_csi_m", "partial_mse"]:
        best_val_loss_init = (
            -np.inf if preload_best_val_loss is None else preload_best_val_loss
        )
        metric_direction = (
            "maximize" if EARLY_STOPPING_METRIC == "partial_csi_m" else "minimize"
        )
        if (
            metric_direction == "minimize"
        ):
            best_val_loss_init = (
                float("inf") if preload_best_val_loss is None else preload_best_val_loss
            )
            print(
                f"{DEBUG_PRINT_PREFIX} Early stopping set to MINIMIZE {EARLY_STOPPING_METRIC}"
            )
        else:
            print(
                f"{DEBUG_PRINT_PREFIX} Early stopping set to MAXIMIZE {EARLY_STOPPING_METRIC}"
            )
    else:
        if is_main_process:
            print(
                f"{DEBUG_PRINT_PREFIX} Warning: Unknown early stopping metric '{EARLY_STOPPING_METRIC}'. Defaulting to validation loss."
            )
        best_val_loss_init = (
            float("inf") if preload_best_val_loss is None else preload_best_val_loss
        )
        metric_direction = "minimize"

    initial_metric_tensor = torch.tensor(
        best_val_loss_init if best_val_loss_init is not None else float("inf"),
        device=device,
    )
    if world_size > 1:
        dist.broadcast(initial_metric_tensor, src=0)
    synced_best_val_loss_init = initial_metric_tensor.item()

    if is_main_process:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            verbose=True,
            path=MODEL_SAVE_PATH,
            initial_best_metric=synced_best_val_loss_init,
            metric_direction=metric_direction,
        )
        print(
            f"{DEBUG_PRINT_PREFIX}Initialized EarlyStopping with initial best metric: {synced_best_val_loss_init}, direction: {metric_direction}"
        )

    if is_main_process:
        print(f"{DEBUG_PRINT_PREFIX}Starting training, run id: {MAIN_RUN_ID}")

    global_step = 0 if preload_global_step is None else preload_global_step

    scaler = torch.amp.GradScaler(device=device.type, enabled=USE_FP16)

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(
            epoch
        )

        model.train()
        train_loss_accum = 0.0
        train_count = 0

        train_bar_desc = f"Training Epoch {epoch} (Rank {rank})"
        train_bar = tqdm(
            train_loader,
            desc=train_bar_desc,
            disable=not is_main_process,
            position=rank,
            leave=False,
        )

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_bar):
            inputs, outputs, metadata = batch
            x1 = outputs.to(device, non_blocking=True)
            x0_cond = inputs.to(device, non_blocking=True)
            x0_noise = torch.randn_like(x1, device=device)

            current_model = model.module if world_size > 1 else model
            x0_cond = current_model.normalize(x0_cond)
            x1 = current_model.normalize(x1)

            final_batch_loss = None

            with torch.amp.autocast(device_type=device.type, enabled=USE_FP16):
                t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                    x0_noise, x1
                )
                v_t = model(t, x_t, x0_cond)
                raw_per_sample_loss = criterion(v_t, u_t)
                dims_to_reduce = list(range(1, raw_per_sample_loss.ndim))
                sample_losses = raw_per_sample_loss.mean(
                    dim=dims_to_reduce
                )

                final_batch_loss = sample_losses.mean()

            if final_batch_loss is None:
                raise ValueError("final_batch_loss is None")

            scaled_loss = final_batch_loss / GRAD_ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()

            raw_loss_value = final_batch_loss.item()
            train_loss_accum += raw_loss_value
            train_count += 1

            if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if SCHEDULER_TYPE == "cosine":
                    scheduler.step()

                if EMA_MODEL_SAVING and ema_model is not None:
                    ema(
                        (
                            model.module if world_size > 1 else model
                        ),
                        ema_model,
                        EMA_MODEL_SAVING_DECAY,
                    )

                if is_main_process:
                    current_lr = optimizer.param_groups[0]["lr"]

                    log_data = {
                        "training_loss_step": raw_loss_value,
                        "learning_rate": current_lr,
                    }

                    if ENABLE_WANDB:
                        wandb.log(log_data, step=global_step)

            global_step += BATCH_SIZE
            if is_main_process:
                train_bar.set_postfix({"training_loss": f"{raw_loss_value:.4f}"})

            if DEBUG_MODE and batch_idx >= 2:
                if is_main_process:
                    print(
                        f"{DEBUG_PRINT_PREFIX}Debug break after {batch_idx+1} batches."
                    )
                break
        train_bar.close()

        avg_train_loss_local = (
            train_loss_accum / train_count if train_count > 0 else 0.0
        )
        loss_tensor = torch.tensor(avg_train_loss_local, device=device)
        if world_size > 1:
            loss_tensor = reduce_tensor(
                loss_tensor, world_size
            )
        avg_train_loss_global = loss_tensor.item()

        model.eval()
        if ema_model is not None:
            ema_model.eval()

        val_loss_accum = 0.0
        val_count = 0

        val_bar_desc = f"Validation Epoch {epoch} (Rank {rank})"
        val_bar = tqdm(
            val_loader,
            desc=val_bar_desc,
            disable=not is_main_process,
            position=rank,
            leave=False,
        )

        with torch.no_grad():
            for batch in val_bar:
                inputs, outputs, metadata = batch
                x1 = outputs.to(device, non_blocking=True)
                x0_cond = inputs.to(device, non_blocking=True)
                x0_noise = torch.randn_like(
                    x1, device=device
                )

                current_model = model.module if world_size > 1 else model
                x0_cond = current_model.normalize(x0_cond)
                x1 = current_model.normalize(x1)

                with torch.amp.autocast(device_type=device.type, enabled=USE_FP16):
                    t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                        x0_noise, x1
                    )
                    v_t = model(t, x_t, x0_cond)
                    per_sample_loss = criterion(v_t, u_t)
                    loss = per_sample_loss.mean()

                val_loss_accum += loss.item() * inputs.size(
                    0
                )
                val_count += inputs.size(0)

                if is_main_process:
                    val_bar.set_postfix(
                        {"validation_loss": "{:.4f}".format(loss.item())}
                    )

                if (
                    DEBUG_MODE and val_count // BATCH_SIZE >= 3
                ):
                    if is_main_process:
                        print(f"{DEBUG_PRINT_PREFIX}Debug break during validation.")
                    break
        val_bar.close()

        val_loss_total_tensor = torch.tensor(val_loss_accum, device=device)
        val_count_total_tensor = torch.tensor(val_count, device=device)

        if world_size > 1:
            dist.all_reduce(val_loss_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_total_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss_global = (
            (val_loss_total_tensor / val_count_total_tensor).item()
            if val_count_total_tensor > 0
            else 0.0
        )

        if SCHEDULER_TYPE == "plateau":
            scheduler.step(avg_val_loss_global)

        partial_eval_results = None
        if (
            is_main_process
            and PARTIAL_EVALUATION
            and (epoch % PARTIAL_EVALUATION_INTERVAL == 0)
        ):

            print(
                f"{DEBUG_PRINT_PREFIX}Running Partial Evaluation for Epoch {epoch}..."
            )
            eval_model = (
                ema_model
                if EMA_MODEL_SAVING and ema_model is not None
                else (model.module if world_size > 1 else model)
            )
            eval_model.eval()

            partial_eval_results = partial_evaluate_model(
                model=eval_model,
                device=device,
                val_sample_loader=val_sample_loader,
                thresholds=THRESHOLDS,
                global_step=global_step,
                epoch=epoch,
                ae_model=ae_model,
                normalized_autoencoder=NORMALIZED_AUTOENCODER,
                use_fp16=USE_FP16,
                partial_evaluation_batches=PARTIAL_EVALUATION_BATCHES,
                lead_time=LEAD_TIME,
                enable_wandb=ENABLE_WANDB,
                wandb_instance=wandb if ENABLE_WANDB else None,
                debug_print_prefix=DEBUG_PRINT_PREFIX,
                plots_folder=METRICS_FOLDER,
                cartopy_features=CARTOPY_FEATURES,
                ema_model_evaluated=EMA_MODEL_SAVING
                and ema_model is not None,
                batch_size_autoencoder=(
                    None if BATCH_SIZE > 2 else BATCH_SIZE
                ),
            )
            print(f"{DEBUG_PRINT_PREFIX}Partial Evaluation finished.")

        if is_main_process:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Finished Epoch {epoch} - Train Loss: {avg_train_loss_global:.4f}, Val Loss: {avg_val_loss_global:.4f}, LR: {current_lr:.6f}"
            )
            if ENABLE_WANDB:
                log_data = {
                    "epoch": epoch,
                    "avg_training_loss": avg_train_loss_global,
                    "avg_validation_loss": avg_val_loss_global,
                    "learning_rate": current_lr,
                }
                wandb.log(log_data, step=global_step)

        stop_signal_tensor = torch.tensor(
            0, device=device, dtype=torch.int
        )
        if is_main_process and early_stopping is not None:
            if EARLY_STOPPING_METRIC == "val_loss":
                current_metric = avg_val_loss_global
            elif EARLY_STOPPING_METRIC == "partial_mse":
                current_metric = (
                    partial_eval_results.get("mse_mean", float("inf"))
                    if partial_eval_results
                    else float("inf")
                )
            elif EARLY_STOPPING_METRIC == "partial_csi_m":
                current_metric = (
                    partial_eval_results.get("csi_from_mean_m", -np.inf)
                    if partial_eval_results
                    else -np.inf
                )
            else:
                current_metric = avg_val_loss_global

            model_to_save = (
                ema_model
                if EMA_MODEL_SAVING and ema_model is not None
                else (model.module if world_size > 1 else model)
            )

            early_stopping(current_metric, model_to_save, optimizer, epoch, global_step)

            if early_stopping.early_stop:
                print(f"{DEBUG_PRINT_PREFIX}Early stopping triggered.")
                stop_signal_tensor = torch.tensor(1, device=device, dtype=torch.int)

        if world_size > 1:
            dist.broadcast(stop_signal_tensor, src=0)

        if stop_signal_tensor.item() == 1:
            if is_main_process:
                print("Early stopping condition met. Finalizing training.")
            break

        if world_size > 1:
            dist.barrier()

    if is_main_process:
        print(f"{DEBUG_PRINT_PREFIX}Finished training, run id: {MAIN_RUN_ID}")
        if ENABLE_WANDB:
            wandb.finish()

    cleanup_ddp()

def partial_evaluate_model(
    model,
    device,
    val_sample_loader,
    thresholds,
    global_step,
    epoch,
    ae_model,
    normalized_autoencoder,
    use_fp16,
    partial_evaluation_batches,
    lead_time,
    enable_wandb,
    wandb_instance,
    debug_print_prefix,
    plots_folder,
    cartopy_features,
    ema_model_evaluated,
    batch_size_autoencoder=None,
):
    """
    Runs a partial evaluation on a subset of the validation data.

    This is executed periodically on the main process during training. It generates
    predictions using ODE integration, decodes them with the autoencoder, calculates
    nowcasting metrics (CSI, MSE, etc.), and creates animations for visual inspection.

    Note: This function runs only on the main DDP process (rank 0). In the future
    we can parallelize this function across all processes.

    Args:
        model (nn.Module): The unwrapped model (or its EMA version) to be evaluated.
        device (torch.device): The device of the main process.
        val_sample_loader (DataLoader): DataLoader for the validation subset.
        thresholds (np.ndarray): Array of thresholds for computing categorical metrics.
        global_step (int): The current global training step for logging.
        epoch (int): The current epoch number.
        ae_model (nn.Module): The pre-trained autoencoder for decoding predictions.
        normalized_autoencoder (bool): Flag indicating if the AE expects normalized inputs.
        use_fp16 (bool): Whether to use mixed-precision for inference.
        partial_evaluation_batches (int): The number of batches to evaluate.
        lead_time (int): The number of future frames to predict and evaluate.
        enable_wandb (bool): If True, log results to WandB.
        wandb_instance: The active WandB run instance.
        debug_print_prefix (str): Prefix for print statements.
        plots_folder (str): Directory to save generated plots and animations.
        cartopy_features (list): List of features to draw on cartopy plots.
        ema_model_evaluated (bool): Flag to indicate if the EMA model is being evaluated.
        batch_size_autoencoder (int, optional): Batch size for the autoencoder's forward pass.

    Returns:
        dict or None: A dictionary containing the computed metrics, or None if evaluation fails.
    """
    results = None
    model.eval()
    if ae_model:
        ae_model.eval()

    with torch.no_grad():
        metrics_accumulators = [
            MetricsAccumulator(
                lead_time=lt,
                thresholds=thresholds,
                pool_size=16,
                compute_mse=True,
                compute_threshold=True,
                compute_crps=True,
                compute_fss=True,
                fss_scales=[1, 4, 16],
                device=device,
            )
            for lt in range(lead_time)
        ]
        count = 0
        y_pred_batches = []
        y_true_batches = []

        eval_bar = tqdm(
            val_sample_loader, desc=f"Partial Eval Epoch {epoch}", leave=False
        )

        for batch in eval_bar:
            x_cond, x_true, metadata = batch
            x_cond = x_cond.to(device, non_blocking=True)
            x_true = x_true.to(
                device, non_blocking=True
            )

            B, C, T_in, H, W = x_cond.shape
            x_cond = x_cond.permute(0, 2, 1, 3, 4).reshape(B * T_in, C, H, W)

            if normalized_autoencoder:
                x_cond = x_cond / 255.0

            if ae_model:
                encoded_chunks = []
                bs_ae = (
                    batch_size_autoencoder
                    if batch_size_autoencoder is not None
                    else x_cond.shape[0]
                )
                for i in range(0, x_cond.shape[0], bs_ae):
                    chunk = x_cond[i : i + bs_ae]
                    encoded_chunk = ae_model.encode(chunk)
                    encoded_chunk = encoded_chunk.latent_dist.mode()
                    encoded_chunks.append(encoded_chunk)
                x_cond = torch.cat(encoded_chunks, dim=0)
            else:
                print(
                    f"{debug_print_prefix}Warning: AE model not available for encoding in partial eval."
                )
                latent_channels, latent_H, latent_W = (
                    4,
                    H // 8,
                    W // 8,
                )
                x_cond = torch.randn(
                    B * T_in, latent_channels, latent_H, latent_W, device=device
                )

            latent_channels, latent_H, latent_W = (
                x_cond.shape[1],
                x_cond.shape[2],
                x_cond.shape[3],
            )
            x_cond = x_cond.reshape(
                B, T_in, latent_channels, latent_H, latent_W
            ).permute(0, 2, 1, 3, 4)

            x_cond = model.normalize(x_cond)
            x_cond = x_cond.permute(0, 2, 3, 4, 1)

            B, Tin, Hz, Wz, Cz = x_cond.shape
            x_true = x_true.squeeze(
                1
            )
            T_future = x_true.shape[1]
            H_true, W_true = x_true.shape[2], x_true.shape[3]

            x_true_downsampled_example = torch.zeros(
                (B, T_future, Hz, Wz, Cz), device=device
            )
            x0_noise = torch.randn_like(x_true_downsampled_example, device=device)
            x0_flat = x0_noise.view(B * T_future, Hz, Wz, Cz)

            def flow_dynamics(t, x_flat):
                x_flow_local = x_flat.view(B, T_future, Hz, Wz, Cz)
                t_batched = t * torch.ones(B, device=x_flow_local.device)
                with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                    v_t = model(t_batched, x_flow_local, x_cond)
                return v_t.contiguous().view(B * T_future, Hz, Wz, Cz)

            t_span = torch.tensor([0.0, 1.0], device=x0_flat.device)

            solution = odeint(
                flow_dynamics,
                x0_flat,
                t_span,
                method="euler",
                options={"step_size": 0.1},
                adjoint_params=model.parameters(),
            )
            x_final_flat = solution[-1]
            x_pred_sample = x_final_flat.view(B, T_future, Hz, Wz, Cz)

            x_pred = x_pred_sample.unsqueeze(1)

            x_pred_np = x_pred.cpu().numpy()
            x_true_np = x_true.cpu().numpy()

            x_pred_np = (x_pred_np * model.std.numpy() + model.mean.numpy()).astype(
                np.float32
            )

            B, S, T, H_latent, W_latent, C_latent = x_pred_np.shape
            x_pred_np = x_pred_np.reshape(B * S * T, H_latent, W_latent, C_latent)

            x_pred_tensor = torch.from_numpy(x_pred_np).to(
                device
            )
            x_pred_tensor = x_pred_tensor.permute(0, 3, 1, 2)

            if ae_model:
                decoded_chunks = []
                bs_ae = (
                    batch_size_autoencoder
                    if batch_size_autoencoder is not None
                    else x_pred_tensor.shape[0]
                )
                for i in range(0, x_pred_tensor.shape[0], bs_ae):
                    chunk = x_pred_tensor[i : i + bs_ae]
                    decoded_chunk = ae_model.decode(chunk)
                    decoded_chunk = decoded_chunk.sample
                    decoded_chunks.append(decoded_chunk)
                x_pred_tensor = torch.cat(
                    decoded_chunks, dim=0
                )
            else:
                print(
                    f"{debug_print_prefix}Warning: AE model not available for decoding in partial eval."
                )
                x_pred_tensor = (
                    torch.rand(B * S * T, 1, H_true, W_true, device=device) * 255.0
                )

            if normalized_autoencoder:
                x_pred_tensor = x_pred_tensor * 255.0

            if torch.isnan(x_pred_tensor).any():
                print(f"{debug_print_prefix} WARNING: NaN values found in x_pred after decode (likely due to FP16) - Please rerun with fp16: false")

            x_pred_tensor = x_pred_tensor.reshape(B, S, T, 1, H_true, W_true)
            x_pred_tensor = x_pred_tensor.permute(
                0, 1, 2, 4, 5, 3
            )
            if x_pred_tensor.shape[-1] == 1:
                x_pred_tensor = x_pred_tensor.squeeze(-1)

            x_pred_np = x_pred_tensor.cpu().numpy().astype(np.float32)

            y_pred_batches.append(x_pred_np)
            y_true_batches.append(x_true_np)

            count += B
            if (
                count >= partial_evaluation_batches * val_sample_loader.batch_size
            ):
                break
        eval_bar.close()

        if not y_pred_batches:
            print(
                f"{debug_print_prefix}No batches processed during partial evaluation."
            )
            return None

        y_pred_array = np.concatenate(y_pred_batches, axis=0)
        y_true_array = np.concatenate(y_true_batches, axis=0)

        y_pred_array = post_process_samples(
            y_pred_array, clamp_min=0.0, clamp_max=255.0
        )

        for metrics_accumulator in metrics_accumulators:
            metrics_accumulator.update(y_true_array, y_pred_array)

        results = calculate_metrics(
            num_lead_times=lead_time,
            metrics_accumulators=metrics_accumulators,
            thresholds=thresholds,
        )
        EMA_SUFFIX = "(EMA)" if ema_model_evaluated else ""
        print(
            f"{debug_print_prefix}Partial Results {EMA_SUFFIX}: MSE: {results.get('mse_from_mean_mean', 'N/A')}, "
            f"CSI-M: {results.get('csi_from_mean_m', 'N/A')}, CSI (pool)-M: {results.get('csi_pooled_from_mean_m', 'N/A')}, "
            f"HSS-M: {results.get('hss_from_mean_m', 'N/A')}, FAR-M: {results.get('far_from_mean_m', 'N/A')}, "
            f"POD-M: {results.get('pod_from_mean_m', 'N/A')}, FSS-M: {results.get('fss_m_from_mean', 'N/A')}"
        )

        EMA_SUFFIX_WANDB = "_EMA" if ema_model_evaluated else ""
        if enable_wandb and wandb_instance:
            log_dict = {
                f"partial_mse{EMA_SUFFIX_WANDB}": results["mse_from_mean_mean"],
                f"partial_csi_m{EMA_SUFFIX_WANDB}": results["csi_from_mean_m"],
                f"partial_csi_pool_m{EMA_SUFFIX_WANDB}": results[
                    "csi_pool_from_mean_m"
                ],
                f"partial_hss_m{EMA_SUFFIX_WANDB}": results["hss_from_mean_m"],
                f"partial_far_m{EMA_SUFFIX_WANDB}": results["far_from_mean_m"],
                f"partial_pod_m{EMA_SUFFIX_WANDB}": results["pod_from_mean_m"],
                f"partial_fss_m{EMA_SUFFIX_WANDB}": results["fss_m_from_mean"],
            }
            wandb_instance.log(log_dict, step=global_step)

        try:
            sample_pred_plot = y_pred_array[0, 0]
            sample_true_plot = y_true_array[0]

            epoch_anim_folder_suffix = "_ema" if ema_model_evaluated else ""
            epoch_anim_folder = os.path.join(
                plots_folder, f"animations{epoch_anim_folder_suffix}", f"Epoch_{epoch}"
            )
            os.makedirs(epoch_anim_folder, exist_ok=True)

            fig1 = plt.figure()
            anim1 = make_animation(
                sample_pred_plot,
                metadata[0],
                title=f"Output Epoch {epoch}{EMA_SUFFIX}",
                fig=fig1,
                cartopy_features=cartopy_features,
            )
            anim1_path = os.path.join(
                epoch_anim_folder, f"output_test_animation_sample0.gif"
            )
            anim1.save(anim1_path, writer="imagemagick", fps=6)
            plt.close(fig1)

            fig2 = plt.figure()
            anim2 = make_animation(
                sample_true_plot,
                metadata[0],
                title=f"Target Epoch {epoch}",
                fig=fig2,
                cartopy_features=cartopy_features,
            )
            anim2_path = os.path.join(epoch_anim_folder, "target_test_animation.gif")
            anim2.save(anim2_path, writer="imagemagick", fps=6)
            plt.close(fig2)

            if enable_wandb and wandb_instance:
                wandb_instance.log(
                    {
                        f"Prediction Animation{EMA_SUFFIX_WANDB}": wandb.Video(
                            anim1_path, fps=6, format="gif"
                        ),
                        "Target Animation": wandb.Video(
                            anim2_path, fps=6, format="gif"
                        ),
                    },
                    step=global_step,
                )

        except Exception as e:
            print(f"{debug_print_prefix} Error creating or saving animations: {e}")

    return results


if __name__ == "__main__":
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if (
        not is_distributed
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        print("WARNING: Multiple GPUs available but not running in distributed mode.")
        print(
            "Use `torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS your_script_name.py [args]`"
        )

    main()
