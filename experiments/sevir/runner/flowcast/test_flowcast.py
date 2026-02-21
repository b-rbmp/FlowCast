"""
This script evaluates a pre-trained FlowCast model on the SEVIR test dataset.

It loads a trained model and its configuration, then iterates through the test set
to generate probabilistic forecasts. For each input sequence, it performs the following steps:
1.  Encodes the input radar frames into a latent space using a pre-trained autoencoder.
2.  Generates multiple forecast samples by solving an Ordinary Differential Equation (ODE)
    with the FlowCast model, starting from random noise.
3.  Decodes the latent predictions back into pixel space using the autoencoder.
4.  Accumulates the predictions and ground truth to calculate a comprehensive set of
    nowcasting metrics (e.g., MSE, CSI, SSIM).
5.  Saves animations of sample forecasts and plots of the final metrics.
"""

import gc
import sys
import os
import time
import wandb
import namegenerator
import datetime

from omegaconf import OmegaConf

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import DataLoader, Subset
from experiments.sevir.display.cartopy import make_animation
import random
from tqdm import tqdm
from common.models.flowcast.cuboid_transformer_unet import (
    CuboidTransformerUNet,
)

from experiments.sevir.dataset.sevirfulldataset import (
    DynamicSequentialSevirDataset,
    dynamic_sequential_collate,
    post_process_samples,
)
from common.metrics.metrics_streaming_probabilistic import (
    MetricsAccumulator,
)
from common.utils.utils import calculate_metrics
import argparse


parser = argparse.ArgumentParser(description="Script for testing FlowCast model.")

parser.add_argument(
    "--artifacts_folder",
    type=str,
    default="saved_models/sevir/flowcast",
    help="Artifacts folder to load model from",
)
parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/runner/flowcast/flowcast_config.yaml",
    help="Path to the configuration file.",
)
parser.add_argument(
    "--test_data_percentage",
    type=float,
    default=1.0,
    help="Percentage of the test data to use (0.0 to 1.0).",
)
parser.add_argument(
    "--test_file",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_testing_full.h5",
)
parser.add_argument(
    "--test_meta",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_testing_full_META.csv",
)
args = parser.parse_args()
if not (0.0 <= args.test_data_percentage <= 1.0):
    raise ValueError("test_data_percentage must be between 0.0 and 1.0")
config = OmegaConf.load(args.config)

DEBUG_MODE = config.run_params.debug_mode
ENABLE_WANDB = config.run_params.enable_wandb
RUN_STRING = config.run_params.run_string

BATCH_SIZE = config.test_params.micro_batch_size
NUM_WORKERS = config.test_params.num_workers
PROBABILISTIC_SAMPLES = config.test_params.probabilistic_samples
BATCH_SIZE_AUTOENCODER = config.test_params.batch_size_autoencoder
CARTOPY_FEATURES = config.test_params.cartopy_features
EULER_STEPS = config.test_params.euler_steps

LAG_TIME = config.data_params.lag_time
LEAD_TIME = config.data_params.lead_time
TIME_SPACING = config.data_params.time_spacing

PRELOAD_AE_MODEL = config.autoencoder_params.autoencoder_checkpoint
NORMALIZED_AUTOENCODER = config.autoencoder_params.normalized_autoencoder
LATENT_CHANNELS = config.autoencoder_params.latent_channels
NORM_NUM_GROUPS = config.autoencoder_params.norm_num_groups
LAYERS_PER_BLOCK = config.autoencoder_params.layers_per_block
ACT_FN = config.autoencoder_params.act_fn
BLOCK_OUT_CHANNELS = config.autoencoder_params.block_out_channels
DOWN_BLOCK_TYPES = config.autoencoder_params.down_block_types
UP_BLOCK_TYPES = config.autoencoder_params.up_block_types


RUN_ID = (
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + "_"
    + RUN_STRING
    + "_"
    + namegenerator.gen()
)
ARTIFACTS_FOLDER = args.artifacts_folder

DEBUG_PRINT_PREFIX = "[DEBUG] " if DEBUG_MODE else ""
TEST_FILE = args.test_file
TEST_META = args.test_meta
THRESHOLDS = np.array([16, 74, 133, 160, 181, 219], dtype=np.float32)

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

print(f"{DEBUG_PRINT_PREFIX}Debug Mode: {DEBUG_MODE}")
print(f"{DEBUG_PRINT_PREFIX}Testing File: {TEST_FILE}")
print(f"{DEBUG_PRINT_PREFIX}Testing Meta: {TEST_META}")
print(f"{DEBUG_PRINT_PREFIX}Normalized Autoencoder: {NORMALIZED_AUTOENCODER}")
print(f"{DEBUG_PRINT_PREFIX}Batch Size: {BATCH_SIZE}")
print(f"{DEBUG_PRINT_PREFIX}Euler Steps: {EULER_STEPS}")
print(f"{DEBUG_PRINT_PREFIX}Number of Workers: {NUM_WORKERS}")
print(f"{DEBUG_PRINT_PREFIX}Lag Time: {LAG_TIME}")
print(f"{DEBUG_PRINT_PREFIX}Lead Time: {LEAD_TIME}")
print(f"{DEBUG_PRINT_PREFIX}Time Spacing: {TIME_SPACING}")
print(f"{DEBUG_PRINT_PREFIX}Thresholds: {THRESHOLDS}")
print(f"{DEBUG_PRINT_PREFIX}Probabilistic Samples: {PROBABILISTIC_SAMPLES}")
print(f"{DEBUG_PRINT_PREFIX}Preload AE Model: {PRELOAD_AE_MODEL}")
print(f"{DEBUG_PRINT_PREFIX}Latent Channels: {LATENT_CHANNELS}")
print(f"{DEBUG_PRINT_PREFIX}Norm Num Groups: {NORM_NUM_GROUPS}")
print(f"{DEBUG_PRINT_PREFIX}Layers Per Block: {LAYERS_PER_BLOCK}")
print(f"{DEBUG_PRINT_PREFIX}Activation Function: {ACT_FN}")
print(f"{DEBUG_PRINT_PREFIX}Block Out Channels: {BLOCK_OUT_CHANNELS}")
print(f"{DEBUG_PRINT_PREFIX}Down Block Types: {DOWN_BLOCK_TYPES}")
print(f"{DEBUG_PRINT_PREFIX}Up Block Types: {UP_BLOCK_TYPES}")
print(f"--------- {DEBUG_PRINT_PREFIX}Flowcast Config ---------")
print(f"{DEBUG_PRINT_PREFIX}Base Units: {BASE_UNITS}")
print(f"{DEBUG_PRINT_PREFIX}Scale Alpha: {SCALE_ALPHA}")
print(f"{DEBUG_PRINT_PREFIX}Depth: {DEPTH}")
print(f"{DEBUG_PRINT_PREFIX}Block Attn Patterns: {BLOCK_ATTN_PATTERNS}")

print(f"{DEBUG_PRINT_PREFIX}Downsample: {DOWNSAMPLE}")
print(f"{DEBUG_PRINT_PREFIX}Downsample Type: {DOWNSAMPLE_TYPE}")
print(f"{DEBUG_PRINT_PREFIX}Upsample Type: {UPSAMPLE_TYPE}")
print(f"{DEBUG_PRINT_PREFIX}Num Global Vectors: {NUM_GLOBAL_VECTORS}")
print(f"{DEBUG_PRINT_PREFIX}ATTN_PROJ_LINEAR_INIT_MODE: {ATTN_PROJ_LINEAR_INIT_MODE}")
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
print(f"{DEBUG_PRINT_PREFIX}Self Attn Use Final Proj: {SELF_ATTN_USE_FINAL_PROJ}")
print(f"{DEBUG_PRINT_PREFIX}Checkpoint Level: {CHECKPOINT_LEVEL}")
print(f"{DEBUG_PRINT_PREFIX}Attn Linear Init Mode: {ATTN_LINEAR_INIT_MODE}")
print(f"{DEBUG_PRINT_PREFIX}FFN Linear Init Mode: {FFN_LINEAR_INIT_MODE}")
print(f"{DEBUG_PRINT_PREFIX}Conv Init Mode: {CONV_INIT_MODE}")
print(f"{DEBUG_PRINT_PREFIX}Down Up Linear Init Mode: {DOWN_UP_LINEAR_INIT_MODE}")
print(f"{DEBUG_PRINT_PREFIX}Norm Init Mode: {NORM_INIT_MODE}")
print(f"{DEBUG_PRINT_PREFIX}Time Embed Channels Mult: {TIME_EMBED_CHANNELS_MULT}")
print(
    f"{DEBUG_PRINT_PREFIX}Time Embed Use Scale Shift Norm: {TIME_EMBED_USE_SCALE_SHIFT_NORM}"
)
print(f"{DEBUG_PRINT_PREFIX}Time Embed Dropout: {TIME_EMBED_DROPOUT}")
print(f"{DEBUG_PRINT_PREFIX}UNET Res Connect: {UNET_RES_CONNECT}")
print(f"{DEBUG_PRINT_PREFIX}Batch Size Autoencoder: {BATCH_SIZE_AUTOENCODER}")

PLOTS_FOLDER = ARTIFACTS_FOLDER + "/plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)
ANIMATIONS_FOLDER = PLOTS_FOLDER + "/animations"
os.makedirs(ANIMATIONS_FOLDER, exist_ok=True)
METRICS_FOLDER = PLOTS_FOLDER + "/metrics"
os.makedirs(METRICS_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEBUG_PRINT_PREFIX}Using device: {device}")
if device.type == "cpu":
    print(DEBUG_PRINT_PREFIX + "CPU is used")
else:
    print(f"{DEBUG_PRINT_PREFIX}Number of GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"{DEBUG_PRINT_PREFIX}Using {torch.cuda.device_count()} GPUs!")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

MODEL_SAVE_DIR = ARTIFACTS_FOLDER + "/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "early_stopping_model" + ".pt")

def safe_encode(model, x):
    """
    Safely encode using the model, handling the DataParallel wrapper.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module.encode(x)
    return model.encode(x)


def safe_decode(model, x):
    """
    Safely decode using the model, handling the DataParallel wrapper.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module.decode(x)
    return model.decode(x)


if ENABLE_WANDB:
    wandb.init(
        project="sevir-nowcasting-testing-cfm",
        name=RUN_ID,
        config={
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "lag_time": LAG_TIME,
            "lead_time": LEAD_TIME,
            "time_spacing": TIME_SPACING,
            "probabilistic_samples": PROBABILISTIC_SAMPLES,
            "model": "flowcast",
            "model_save_path": MODEL_SAVE_PATH,
        },
    )

PRELOAD_MODEL = MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else None
if PRELOAD_MODEL is None:
    raise FileNotFoundError(f"Model not found at {MODEL_SAVE_PATH}")
else:
    print(f"{DEBUG_PRINT_PREFIX}Model found at {MODEL_SAVE_PATH}")

    full_test_dataset = DynamicSequentialSevirDataset(
        meta_csv=TEST_META,
        data_file=TEST_FILE,
        data_type="vil",
        raw_seq_len=49,
        lag_time=LAG_TIME,
        lead_time=LEAD_TIME,
        time_spacing=TIME_SPACING,
        stride=12,
        channel_last=False,
        debug_mode=DEBUG_MODE,
    )

    if args.test_data_percentage < 1.0:
        num_samples = int(len(full_test_dataset) * args.test_data_percentage)
        test_dataset = Subset(full_test_dataset, range(num_samples))
    else:
        test_dataset = full_test_dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_sequential_collate,
        num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
        pin_memory=True if not DEBUG_MODE else False,
    )

    if not os.path.exists(PRELOAD_AE_MODEL):
        raise FileNotFoundError(f"Model not found at {PRELOAD_AE_MODEL}")

    from diffusers.models.autoencoders import AutoencoderKL

    ae_model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=DOWN_BLOCK_TYPES,
        up_block_types=UP_BLOCK_TYPES,
        block_out_channels=BLOCK_OUT_CHANNELS,
        act_fn=ACT_FN,
        latent_channels=LATENT_CHANNELS,
        norm_num_groups=NORM_NUM_GROUPS,
        layers_per_block=LAYERS_PER_BLOCK,
    )

    checkpoint = torch.load(PRELOAD_AE_MODEL, map_location=device)
    # Remove 'module.' prefix if it exists (in case the AE was trained with DDP)
    new_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    ae_model.load_state_dict(new_state_dict)
    ae_model = ae_model.to(device)
    if torch.cuda.device_count() > 1:
        ae_model = torch.nn.DataParallel(ae_model)
    ae_model.eval()

    input_shape = None
    output_shape = None
    for batch in test_loader:
        inputs, outputs, metadata = batch
        inputs_decoded = inputs[:, :, 0, :, :].to(device)
        encoded_obj = safe_encode(ae_model, inputs_decoded)
        inputs_decoded = encoded_obj.latent_dist.mode()

        print(f"Inputs shape: {inputs.shape}")
        print(f"Inputs Decoded shape: {inputs_decoded.shape}")
        print(f"Outputs shape: {outputs.shape}")
        input_shape = inputs.shape
        inputs_decoded_shape = inputs_decoded.shape
        output_shape = outputs.shape
        break
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    checkpoint_mean = checkpoint["mean"]
    checkpoint_std = checkpoint["std"]
    checkpoint_model_state_dict = checkpoint["model_state_dict"]

    IN_TIMESTEPS = input_shape[2]
    OUTPUT_TIMESTEPS = output_shape[2]

    input_shape_flowcast = (
        IN_TIMESTEPS,
        inputs_decoded_shape[2],
        inputs_decoded_shape[3],
        inputs_decoded_shape[1],
    )
    output_shape_flowcast = (
        OUTPUT_TIMESTEPS,
        inputs_decoded_shape[2],
        inputs_decoded_shape[3],
        inputs_decoded_shape[1],
    )
    loaded_model = CuboidTransformerUNet(
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
        # global vectors
        num_global_vectors=NUM_GLOBAL_VECTORS,
        use_global_vector_ffn=USE_GLOBAL_VECTOR_FFN,
        use_global_self_attn=USE_GLOBAL_SELF_ATTN,
        separate_global_qkv=SEPARATE_GLOBAL_QKV,
        global_dim_ratio=GLOBAL_DIM_RATIO,
        # misc
        ffn_activation=FFN_ACTIVATION,
        gated_ffn=GATED_FFN,
        norm_layer=NORM_LAYER,
        padding_type=PADDING_TYPE,
        checkpoint_level=CHECKPOINT_LEVEL,
        pos_embed_type=POS_EMBED_TYPE,
        use_relative_pos=USE_RELATIVE_POS,
        self_attn_use_final_proj=SELF_ATTN_USE_FINAL_PROJ,
        # initialization
        attn_linear_init_mode=ATTN_LINEAR_INIT_MODE,
        ffn_linear_init_mode=FFN_LINEAR_INIT_MODE,
        ffn2_linear_init_mode=FFN2_LINEAR_INIT_MODE,
        attn_proj_linear_init_mode=ATTN_PROJ_LINEAR_INIT_MODE,
        conv_init_mode=CONV_INIT_MODE,
        down_linear_init_mode=DOWN_UP_LINEAR_INIT_MODE,
        up_linear_init_mode=DOWN_UP_LINEAR_INIT_MODE,
        global_proj_linear_init_mode=GLOBAL_PROJ_LINEAR_INIT_MODE,
        norm_init_mode=NORM_INIT_MODE,
        # timestep embedding
        time_embed_channels_mult=TIME_EMBED_CHANNELS_MULT,
        time_embed_use_scale_shift_norm=TIME_EMBED_USE_SCALE_SHIFT_NORM,
        time_embed_dropout=TIME_EMBED_DROPOUT,
        unet_res_connect=UNET_RES_CONNECT,
        mean=checkpoint_mean,
        std=checkpoint_std,
    )
    loaded_model.load_state_dict(checkpoint_model_state_dict)
    loaded_model = loaded_model.to(device)
    if torch.cuda.device_count() > 1:
        loaded_model = torch.nn.DataParallel(loaded_model)
    loaded_model.eval()

    metrics_accumulators = [
        MetricsAccumulator(
            lead_time=lead_time,
            thresholds=THRESHOLDS,
            pool_size=16,
            compute_mse=True,
            compute_threshold=True,
            compute_crps=True,
            compute_fss=True,
            fss_scales=[1, 4, 16],
            device=device,
        )
        for lead_time in range(LEAD_TIME)
    ]

    test_bar = tqdm(test_loader, desc="Testing Model")
    count = 0
    y_pred = []
    y_true = []
    total_prediction_time = 0.0
    total_samples_processed = 0
    for idx, batch in enumerate(test_bar):
        x_cond, x_true, metadata = batch

        B, C, T_in, H, W = x_cond.shape

        x_cond = x_cond.permute(0, 2, 1, 3, 4).reshape(B * T_in, C, H, W)

        with torch.no_grad():
            x_cond = x_cond.to(device)

            if NORMALIZED_AUTOENCODER:
                x_cond = x_cond / 255.0
            encoded_obj = safe_encode(ae_model, x_cond)
            x_cond = encoded_obj.latent_dist.mode()


        latent_channels, latent_H, latent_W = (
            x_cond.shape[1],
            x_cond.shape[2],
            x_cond.shape[3],
        )
        x_cond = x_cond.reshape(B, T_in, latent_channels, latent_H, latent_W).permute(
            0, 2, 1, 3, 4
        )

        x_cond = (
            loaded_model.module.normalize(x_cond)
            if isinstance(loaded_model, torch.nn.DataParallel)
            else loaded_model.normalize(x_cond)
        )
        x_cond = x_cond.permute(0, 2, 3, 4, 1)

        B, Tin, Hz, Wz, Cz = x_cond.shape

        x_true = x_true.squeeze(1)
        T_future = x_true.shape[1]
        sample_predictions = []

        x_true_downsampled_example = torch.zeros(
            (B, T_future, Hz, Wz, Cz),
            device=device,
        )

        start_time = time.time()
        for sample_idx in range(PROBABILISTIC_SAMPLES):
            torch.manual_seed(idx * PROBABILISTIC_SAMPLES + sample_idx)

            x0_noise = torch.randn_like(x_true_downsampled_example, device=device)
            x0_flat = x0_noise.view(B * (T_future), Hz, Wz, Cz)

            def flow_dynamics(t, x_flat):
                x_flow_local = x_flat.view(B, (T_future), Hz, Wz, Cz)
                t_batched = t * torch.ones(B, device=x_flow_local.device)

                with torch.no_grad():
                    v_t = loaded_model(
                        t_batched, x_flow_local, x_cond
                    )
                return v_t.view(B * (T_future), Hz, Wz, Cz)

            t_span = torch.tensor([0.0, 1.0], device=x0_flat.device)
            if EULER_STEPS == 0:
                solution = odeint(
                    flow_dynamics,
                    x0_flat,
                    t_span,
                    method="adaptive_heun",
                    rtol=1e-2,
                    atol=1e-3,
                    adjoint_params=loaded_model.parameters(),
                )
            else:
                euler_step_size = 1.0 / float(EULER_STEPS)
                solution = odeint(
                    flow_dynamics,
                    x0_flat,
                    t_span,
                    method="euler",
                    options=dict(step_size=euler_step_size),
                    atol=1e-3,
                    rtol=1e-2,
                    adjoint_params=loaded_model.parameters(),
                )
            x_final_flat = solution[-1]
            x_pred_sample = x_final_flat.view(
                B, (T_future), Hz, Wz, Cz
            )

            sample_predictions.append(x_pred_sample.unsqueeze(1))

        x_pred = torch.cat(sample_predictions, dim=1)

        end_time = time.time()
        total_prediction_time += end_time - start_time
        total_samples_processed += B

        x_pred = x_pred.cpu().detach().numpy()
        x_true_np = x_true.cpu().numpy()

        mean_val = (
            loaded_model.module.mean.item()
            if isinstance(loaded_model, torch.nn.DataParallel)
            else loaded_model.mean.item()
        )
        std_val = (
            loaded_model.module.std.item()
            if isinstance(loaded_model, torch.nn.DataParallel)
            else loaded_model.std.item()
        )
        x_pred = (x_pred * std_val + mean_val).astype(np.float32)

        B, S, T, H, W, C = x_pred.shape

        x_pred = x_pred.reshape(B * S * T, H, W, C)

        if isinstance(x_pred, np.ndarray):
            x_pred = torch.from_numpy(x_pred).to(device)

        x_pred = x_pred.permute(0, 3, 1, 2)

        with torch.no_grad():
            if BATCH_SIZE_AUTOENCODER is not None:
                encoded_chunks = []
                for i in range(0, x_pred.shape[0], BATCH_SIZE_AUTOENCODER):
                    chunk = x_pred[i : i + BATCH_SIZE_AUTOENCODER]
                    decoded_chunk_obj = safe_decode(ae_model, chunk)
                    final_decoded_chunk = decoded_chunk_obj.sample
                    encoded_chunks.append(
                        final_decoded_chunk
                    )

                x_pred = torch.cat(encoded_chunks, dim=0)
            else:
                decoded_obj_fallback = safe_decode(ae_model, x_pred)
                x_pred = decoded_obj_fallback.sample

        if NORMALIZED_AUTOENCODER:
            x_pred = x_pred * 255.0
        new_channels, new_H, new_W = x_pred.shape[1], x_pred.shape[2], x_pred.shape[3]

        x_pred = x_pred.reshape(B, S, T, new_channels, new_H, new_W)
        x_pred = x_pred.permute(0, 1, 2, 4, 5, 3)

        if x_pred.shape[-1] == 1:
            x_pred = x_pred.squeeze(-1)

        x_pred = x_pred.cpu().detach().numpy().astype(np.float16)

        y_pred.append(x_pred)
        y_true.append(x_true_np)

        if idx % int((400 / BATCH_SIZE) / PROBABILISTIC_SAMPLES) == 0 and idx > 0:

            y_pred_array = np.concatenate(y_pred, axis=0)
            y_pred_array = post_process_samples(
                y_pred_array, clamp_min=0.0, clamp_max=255.0
            )
            y_true_array = np.concatenate(y_true, axis=0)

            for lead_time, metrics_accumulator in enumerate(metrics_accumulators):
                metrics_accumulator.update(y_true_array, y_pred_array)

            batch_size_y_true = y_pred_array.shape[0]

            y_pred = []
            y_true = []

            results = calculate_metrics(
                num_lead_times=LEAD_TIME,
                metrics_accumulators=metrics_accumulators,
                thresholds=THRESHOLDS,
            )

            if ENABLE_WANDB:
                global_step = idx * batch_size_y_true
                wandb.log(
                    {
                        "partial_mse": results["mse_from_mean_mean"],
                        "partial_crps": results["crps_mean"],
                        "partial_csi_m": results["csi_from_mean_m"],
                        "partial_csi_pool_m": results["csi_pool_from_mean_m"],
                        "partial_hss_m": results["hss_from_mean_m"],
                        "partial_far_m": results["far_from_mean_m"],
                        "partial_pod_m": results["pod_from_mean_m"],
                        "partial_fss_m": results["fss_m_from_mean"],
                    },
                    step=global_step,
                )

        if idx == 0:
            sample_pred = x_pred[0]
            sample_pred = post_process_samples(
                sample_pred, clamp_min=0.0, clamp_max=255.0
            )
            for i in range(sample_pred.shape[0]):
                sample_pred_plot = sample_pred[i]
                fig1 = plt.figure()
                anim = make_animation(
                    sample_pred_plot,
                    metadata[0],
                    title="Outputs",
                    fig=fig1,
                    cartopy_features=CARTOPY_FEATURES,
                )
                anim.save(
                    os.path.join(
                        PLOTS_FOLDER, "animations", f"output_test_animation{i}.gif"
                    ),
                    writer="imagemagick",
                    fps=6,
                )
                plt.close(fig1)

            fig2 = plt.figure()
            anim = make_animation(
                x_true_np[0],
                metadata[0],
                title="Target",
                fig=fig2,
                cartopy_features=CARTOPY_FEATURES,
            )
            anim.save(
                os.path.join(PLOTS_FOLDER, "animations", "target_test_animation.gif"),
                writer="imagemagick",
                fps=6,
            )
            plt.close(fig2)

        count += 1
        if DEBUG_MODE and count > 10:
            print(f"{DEBUG_PRINT_PREFIX}Breaking early due to DEBUG_MODE")
            break

    if len(y_pred) > 0:
        y_pred_array = np.concatenate(y_pred, axis=0)
        y_pred_array = post_process_samples(
            y_pred_array, clamp_min=0.0, clamp_max=255.0
        )
        y_true_array = np.concatenate(y_true, axis=0)
        for lead_time, metrics_accumulator in enumerate(metrics_accumulators):
            metrics_accumulator.update(y_true_array, y_pred_array)

    del y_pred
    del y_true
    gc.collect()

    results = calculate_metrics(
        num_lead_times=LEAD_TIME,
        metrics_accumulators=metrics_accumulators,
        thresholds=THRESHOLDS,
    )

    crps_mean = results["crps_mean"]

    if total_samples_processed > 0:
        average_time_per_prediction = total_prediction_time / total_samples_processed
        print(
            f"Average time per ensemble prediction: {average_time_per_prediction:.4f} seconds"
        )

    print(f"CRPS: {crps_mean}")

    mse_from_mean_mean = results["mse_from_mean_mean"]
    csi_from_mean_m = results["csi_from_mean_m"]
    csi_pool_from_mean_m = results["csi_pool_from_mean_m"]
    hss_from_mean_m = results["hss_from_mean_m"]
    far_from_mean_m = results["far_from_mean_m"]
    pod_from_mean_m = results["pod_from_mean_m"]
    csi_from_mean_mean_dict = results["csi_from_mean_mean"]
    far_from_mean_mean_dict = results["far_from_mean_mean"]
    hss_from_mean_mean_dict = results["hss_from_mean_mean"]
    pod_from_mean_mean_dict = results["pod_from_mean_mean"]
    csi_pool_from_mean_mean_dict = results["csi_pool_from_mean_mean"]

    print("--- Metrics from Ensemble Mean ---")
    print(f"Mean MSE : {mse_from_mean_mean}")
    print(f"CSI-M : {csi_from_mean_m}")
    print(f"CSI (16-pooled)-M : {csi_pool_from_mean_m}")
    print(f"HSS-M : {hss_from_mean_m}")
    print(f"FAR-M : {far_from_mean_m}")
    print(f"POD-M : {pod_from_mean_m}")
    print("CSI per threshold:", csi_from_mean_mean_dict)
    print("FAR per threshold:", far_from_mean_mean_dict)
    print("HSS per threshold:", hss_from_mean_mean_dict)
    print("POD per threshold:", pod_from_mean_mean_dict)
    print(f"CSI (16-pooled) mean per threshold: {csi_pool_from_mean_mean_dict}")
    csi_m_from_mean_lead_time = results["csi_m_from_mean_lead_time"]
    csi_last_thresh_from_mean_lead_time = results["csi_last_thresh_from_mean_lead_time"]
    csi_pool_m_from_mean_lead_time = results["csi_pool_m_from_mean_lead_time"]
    csi_pool_last_thresh_from_mean_lead_time = results[
        "csi_pool_last_thresh_from_mean_lead_time"
    ]
    hss_m_from_mean_lead_time = results["hss_m_from_mean_lead_time"]
    far_m_from_mean_lead_time = results["far_m_from_mean_lead_time"]
    pod_m_from_mean_lead_time = results["pod_m_from_mean_lead_time"]
    print("--- Lead Time Metrics ---")
    print(f"CSI-M by lead time: {csi_m_from_mean_lead_time}")
    print(f"CSI-M (219) by lead time: {csi_last_thresh_from_mean_lead_time}")
    print(f"CSI (16-pooled)-M by lead time: {csi_pool_m_from_mean_lead_time}")
    print(
        f"CSI (16-pooled) (219) by lead time: {csi_pool_last_thresh_from_mean_lead_time}"
    )
    print(f"HSS-M by lead time: {hss_m_from_mean_lead_time}")
    print(f"FAR-M by lead time: {far_m_from_mean_lead_time}")
    print(f"POD-M by lead time: {pod_m_from_mean_lead_time}")

    print(DEBUG_PRINT_PREFIX + "Finished testing the model")
