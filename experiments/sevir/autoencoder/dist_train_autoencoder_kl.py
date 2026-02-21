"""
Distributed training script for a Variational Autoencoder (VAE) with a GAN-style
discriminator on the SEVIR dataset.
"""

import sys
import os

sys.path.append(os.getcwd())  # Add the current working directory to the path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import datetime
import numpy as np
import wandb
import namegenerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from common.utils.utils import warmup_lambda
from common.autoencoder.losses.lpips import LPIPSWithDiscriminator

import random
from tqdm import tqdm

from experiments.sevir.dataset.sevirfulldataset_autoencoder import (
    DynamicAutoencoderSevirDataset,
    sequential_collate,
)
from diffusers.models.autoencoders import AutoencoderKL
from common.autoencoder.utils.early_stopping import EarlyStopping
from experiments.sevir.display.cartopy import plot_pair_frames
import argparse


def setup_ddp():
    """Initializes the DDP process group."""
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
        return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    """Cleans up the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Cleaned up DDP.")


def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for recursive calls.
        sep (str): Separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


rank, world_size, local_rank, device = setup_ddp()

if rank == 0:
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("CPU is used")

parser = argparse.ArgumentParser(description="Script for configuring hyperparameters.")
parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/autoencoder/autoencoder_kl_config.yaml",
    help="Path to the YAML configuration file.",
)
parser.add_argument(
    "--train_file",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_training_full.h5",
)
parser.add_argument(
    "--train_meta",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_training_full_META.csv",
)
parser.add_argument(
    "--val_file",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_validation_full.h5",
)
parser.add_argument(
    "--val_meta",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_validation_full_META.csv",
)

args = parser.parse_args()

config = OmegaConf.load(args.config)
run_params = config.run_params
training_params = config.training_params
optimizer_params = config.optimizer_params
scheduler_params = config.scheduler_params
model_params = config.model_params
loss_params = config.loss_params
TRAIN_FILE = args.train_file
TRAIN_META = args.train_meta
VAL_FILE = args.val_file
VAL_META = args.val_meta

DEBUG_MODE = run_params.debug_mode
RUN_STRING = run_params.run_string

run_id_timestamp_string = (
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + RUN_STRING
)
random_name_part_list = [None]
if rank == 0:
    random_name_part_list[0] = namegenerator.gen()

if world_size > 1:
    dist.broadcast_object_list(random_name_part_list, src=0)
random_name_part = random_name_part_list[0]
if random_name_part is None and rank != 0:
    print(
        f"[WARNING] Rank {rank} did not receive random_name_part, generating locally. RUN_ID might be inconsistent."
    )
    random_name_part = namegenerator.gen()


RUN_ID = run_id_timestamp_string + "_" + random_name_part

DEBUG_PRINT_PREFIX = f"[DEBUG Rank {rank}] " if DEBUG_MODE else f"[Rank {rank}] "

ENABLE_WANDB = run_params.enable_wandb

NORMALIZE_DATASET = training_params.normalize_dataset
PRELOAD_MODEL = training_params.preload_model
BATCH_SIZE = training_params.micro_batch_size
NUM_EPOCHS = training_params.num_epochs
NUM_WORKERS = training_params.num_workers
EARLY_STOPPING_PATIENCE = training_params.early_stopping_patience
WARMUP_GENERATOR_EPOCHS = training_params.warmup_generator_epochs

LEARNING_RATE = optimizer_params.learning_rate
OPTIMIZER_TYPE = optimizer_params.optimizer_type
WEIGHT_DECAY = optimizer_params.weight_decay

SCHEDULER_TYPE = scheduler_params.scheduler_type
LR_PLATEAU_FACTOR = scheduler_params.lr_plateau_factor
LR_PLATEAU_PATIENCE = scheduler_params.lr_plateau_patience
LR_COSINE_MIN_LR_RATIO = scheduler_params.lr_cosine_min_lr_ratio
LR_COSINE_WARMUP_ITER_PERCENTAGE = scheduler_params.lr_cosine_warmup_iter_percentage
LR_COSINE_MIN_WARMUP_LR_RATIO = scheduler_params.lr_cosine_min_warmup_lr_ratio

GRADIENT_CLIP_VAL = training_params.gradient_clip_val

# Autoencoder
LATENT_CHANNELS = model_params.latent_channels
NORM_NUM_GROUPS = model_params.norm_num_groups
LAYERS_PER_BLOCK = model_params.layers_per_block
ACT_FN = model_params.act_fn
BLOCK_OUT_CHANNELS = model_params.block_out_channels
DOWN_BLOCK_TYPES = model_params.down_block_types
UP_BLOCK_TYPES = model_params.up_block_types

KL_WEIGHT = loss_params.kl_weight
DISC_WEIGHT = loss_params.disc_weight


if rank == 0:
    print(f"{DEBUG_PRINT_PREFIX}Run ID: {RUN_ID}")
    print(f"{DEBUG_PRINT_PREFIX}Debug Mode: {DEBUG_MODE}")
    print(f"{DEBUG_PRINT_PREFIX}Training File: {TRAIN_FILE}")
    print(f"{DEBUG_PRINT_PREFIX}Training Meta: {TRAIN_META}")
    print(f"{DEBUG_PRINT_PREFIX}Preload Model: {PRELOAD_MODEL}")
    print(f"{DEBUG_PRINT_PREFIX}Batch Size: {BATCH_SIZE}")
    print(f"{DEBUG_PRINT_PREFIX}Learning Rate: {LEARNING_RATE}")
    print(f"{DEBUG_PRINT_PREFIX}Number of Epochs: {NUM_EPOCHS}")
    print(f"{DEBUG_PRINT_PREFIX}Number of Workers: {NUM_WORKERS}")
    print(f"{DEBUG_PRINT_PREFIX}Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{DEBUG_PRINT_PREFIX}Warmup Generator Epochs: {WARMUP_GENERATOR_EPOCHS}")
    print(f"{DEBUG_PRINT_PREFIX}Optimizer Type: {OPTIMIZER_TYPE}")
    print(f"{DEBUG_PRINT_PREFIX}Scheduler Type: {SCHEDULER_TYPE}")
    print(f"{DEBUG_PRINT_PREFIX}LR Plateau Factor: {LR_PLATEAU_FACTOR}")
    print(f"{DEBUG_PRINT_PREFIX}LR Plateau Patience: {LR_PLATEAU_PATIENCE}")
    print(f"{DEBUG_PRINT_PREFIX}Weight Decay: {WEIGHT_DECAY}")
    print(
        f"{DEBUG_PRINT_PREFIX}LR Cosine Warmup Iter Percentage: {LR_COSINE_WARMUP_ITER_PERCENTAGE}"
    )
    print(
        f"{DEBUG_PRINT_PREFIX}LR Cosine Min Warmup LR Ratio: {LR_COSINE_MIN_WARMUP_LR_RATIO}"
    )
    print(f"{DEBUG_PRINT_PREFIX}LR Cosine Min LR Ratio: {LR_COSINE_MIN_LR_RATIO}")
    print(f"{DEBUG_PRINT_PREFIX}Gradient Clip Value: {GRADIENT_CLIP_VAL}")
    print(f"{DEBUG_PRINT_PREFIX}Latent Channels: {LATENT_CHANNELS}")
    print(f"{DEBUG_PRINT_PREFIX}Norm Num Groups: {NORM_NUM_GROUPS}")
    print(f"{DEBUG_PRINT_PREFIX}Layers Per Block: {LAYERS_PER_BLOCK}")
    print(f"{DEBUG_PRINT_PREFIX}Activation Function: {ACT_FN}")
    print(f"{DEBUG_PRINT_PREFIX}Block Out Channels: {BLOCK_OUT_CHANNELS}")
    print(f"{DEBUG_PRINT_PREFIX}Down Block Types: {DOWN_BLOCK_TYPES}")
    print(f"{DEBUG_PRINT_PREFIX}Up Block Types: {UP_BLOCK_TYPES}")
    print(f"{DEBUG_PRINT_PREFIX}KL Weight: {KL_WEIGHT}")
    print(f"{DEBUG_PRINT_PREFIX}Discriminator Weight: {DISC_WEIGHT}")

if ENABLE_WANDB and rank == 0:
    primitive_config = OmegaConf.to_container(config, resolve=True)
    flat_config = flatten_dict(primitive_config)
    wandb.init(
        project="sevir-nowcasting",
        config=flat_config,
        name=RUN_ID,
    )

ARTIFACTS_FOLDER = "artifacts/sevir/autoencoder_kl/" + RUN_ID
PLOTS_FOLDER = ARTIFACTS_FOLDER + "/plots"
ANIMATIONS_FOLDER = PLOTS_FOLDER + "/animations"
METRICS_FOLDER = PLOTS_FOLDER + "/metrics"

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

if rank == 0:
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(ANIMATIONS_FOLDER, exist_ok=True)
    os.makedirs(METRICS_FOLDER, exist_ok=True)
    MODEL_SAVE_DIR = ARTIFACTS_FOLDER + "/models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "early_stopping_model" + ".pt")
else:
    MODEL_SAVE_PATH = None

train_dataset = DynamicAutoencoderSevirDataset(
    meta_csv=TRAIN_META,
    data_file=TRAIN_FILE,
    data_type="vil",
    raw_seq_len=49,
    channel_last=False,
    debug_mode=DEBUG_MODE,
    normalize=NORMALIZE_DATASET,
)
val_dataset = DynamicAutoencoderSevirDataset(
    meta_csv=VAL_META,
    data_file=VAL_FILE,
    data_type="vil",
    raw_seq_len=49,
    channel_last=False,
    debug_mode=DEBUG_MODE,
    normalize=NORMALIZE_DATASET,
)

train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
)
val_sampler = DistributedSampler(
    val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=sequential_collate,
    num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
    pin_memory=True if not DEBUG_MODE else False,
    sampler=train_sampler,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=sequential_collate,
    num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
    pin_memory=True if not DEBUG_MODE else False,
    sampler=val_sampler,
)


total_num_steps = len(train_loader) * NUM_EPOCHS

input_shape_list = [None]
if rank == 0:
    temp_input_shape = None
    if len(train_loader) > 0:
        for batch in train_loader:
            inputs, _ = batch
            print(f"Inputs shape: {inputs.shape}")
            temp_input_shape = inputs.shape
            break
    input_shape_list[0] = temp_input_shape

if world_size > 1:
    dist.broadcast_object_list(input_shape_list, src=0)
input_shape = input_shape_list[0]

if input_shape is None:
    raise RuntimeError(f"Rank {rank}: Could not determine input_shape.")


preload_model_state_dict = None
preload_global_step = None
preload_best_val_loss = None
loaded_checkpoint_info = [None, None, None]

if rank == 0:
    if PRELOAD_MODEL is not None:
        try:
            model_info = torch.load(PRELOAD_MODEL, map_location="cpu")
            loaded_checkpoint_info[0] = model_info.get("model_state_dict")
            loaded_checkpoint_info[1] = model_info.get("global_step")
            loaded_checkpoint_info[2] = model_info.get("best_val_loss")
            if loaded_checkpoint_info[0] is not None:
                print(
                    f"{DEBUG_PRINT_PREFIX}Successfully loaded model information from checkpoint."
                )
        except Exception as e:
            print(f"{DEBUG_PRINT_PREFIX}Error loading checkpoint: {e}. Starting fresh.")

if world_size > 1:
    dist.broadcast_object_list(loaded_checkpoint_info, src=0)

preload_model_state_dict = loaded_checkpoint_info[0]
preload_global_step = loaded_checkpoint_info[1]
preload_best_val_loss = loaded_checkpoint_info[2]

model = AutoencoderKL(
    in_channels=input_shape[1],
    out_channels=input_shape[1],
    down_block_types=list(DOWN_BLOCK_TYPES),
    up_block_types=list(UP_BLOCK_TYPES),
    block_out_channels=list(BLOCK_OUT_CHANNELS),
    act_fn=ACT_FN,
    latent_channels=LATENT_CHANNELS,
    norm_num_groups=NORM_NUM_GROUPS,
    layers_per_block=LAYERS_PER_BLOCK,
)

if preload_model_state_dict is not None:
    model.load_state_dict(preload_model_state_dict)
    if rank == 0:
        print(
            f"{DEBUG_PRINT_PREFIX}Successfully loaded model state dict from checkpoint"
        )


model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

criterion = LPIPSWithDiscriminator(
    kl_weight=KL_WEIGHT,
    disc_weight=DISC_WEIGHT,
    disc_in_channels=1,
).to(device)

if OPTIMIZER_TYPE == "adam":
    gen_optimizer = torch.optim.Adam(model.module.parameters(), lr=LEARNING_RATE)
    disc_optimizer = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=LEARNING_RATE
    )
elif OPTIMIZER_TYPE == "adamw":
    gen_optimizer = torch.optim.AdamW(
        model.module.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    disc_optimizer = torch.optim.AdamW(
        criterion.discriminator.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
else:
    raise ValueError(f"Invalid optimizer type: {OPTIMIZER_TYPE}")

warmup_iter = int(np.round(LR_COSINE_WARMUP_ITER_PERCENTAGE * total_num_steps))
if SCHEDULER_TYPE == "cosine":
    gen_warmup_scheduler = LambdaLR(
        gen_optimizer,
        lr_lambda=warmup_lambda(
            warmup_steps=warmup_iter, min_lr_ratio=LR_COSINE_MIN_WARMUP_LR_RATIO
        ),
    )
    gen_cosine_scheduler = CosineAnnealingLR(
        gen_optimizer,
        T_max=(total_num_steps - warmup_iter),
        eta_min=LR_COSINE_MIN_LR_RATIO * LEARNING_RATE,
    )
    gen_scheduler = SequentialLR(
        gen_optimizer,
        schedulers=[gen_warmup_scheduler, gen_cosine_scheduler],
        milestones=[warmup_iter],
    )
    disc_warmup_scheduler = LambdaLR(
        disc_optimizer,
        lr_lambda=warmup_lambda(
            warmup_steps=warmup_iter, min_lr_ratio=LR_COSINE_MIN_WARMUP_LR_RATIO
        ),
    )
    disc_cosine_scheduler = CosineAnnealingLR(
        disc_optimizer,
        T_max=(total_num_steps - warmup_iter),
        eta_min=LR_COSINE_MIN_LR_RATIO * LEARNING_RATE,
    )
    disc_scheduler = SequentialLR(
        disc_optimizer,
        schedulers=[disc_warmup_scheduler, disc_cosine_scheduler],
        milestones=[warmup_iter],
    )
elif SCHEDULER_TYPE == "plateau":
    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
    )
    disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        disc_optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
    )
else:
    raise ValueError(f"Invalid scheduler type: {SCHEDULER_TYPE}")

best_val_loss = float("inf") if preload_best_val_loss is None else preload_best_val_loss

early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE,
    verbose=True,
    path=MODEL_SAVE_PATH,
    val_loss_min=best_val_loss,
)

print(DEBUG_PRINT_PREFIX + "Starting training, run id: " + RUN_ID)

global_step = 0 if preload_global_step is None else preload_global_step

def train_epoch(
    train_bar,
    global_step,
    model,
    criterion,
    device,
    gen_optimizer,
    disc_optimizer,
    gen_scheduler,
    disc_scheduler,
    scheduler_type,
):
    """
    Runs one full training epoch.

    This function iterates through the training data, performing two main updates
    per batch:
    1. Generator (Autoencoder) Update: Updates the autoencoder's weights based on
       reconstruction, KL, and adversarial losses.
    2. Discriminator Update: Updates the discriminator's weights based on its
       ability to distinguish real vs. reconstructed images.

    Args:
        train_bar (tqdm): Progress bar for the training loader.
        global_step (int): Current global step for logging.
        model (torch.nn.Module): The AutoencoderKL model.
        criterion (LPIPSWithDiscriminator): The loss function.
        device (torch.device): The computation device.
        gen_optimizer (torch.optim.Optimizer): Autoencoder optimizer.
        disc_optimizer (torch.optim.Optimizer): Discriminator optimizer.
        gen_scheduler: Learning rate scheduler for the autoencoder.
        disc_scheduler: Learning rate scheduler for the discriminator.
        scheduler_type (str): Type of scheduler ('cosine' or 'plateau').

    Returns:
        A tuple containing the updated global step and the average total,
        generator, and discriminator losses for the epoch.
    """
    global_step_counter = global_step
    model.train()
    criterion.train()
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    count = 0

    for batch in train_bar:
        input_frame, _ = batch

        input_frame = input_frame.to(device)

        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()

        # -------------------------
        # 1) Generator update
        # -------------------------

        # Forward pass: get reconstruction and latent distribution.
        outputs_dict = model(input_frame)
        recon = outputs_dict.sample  # reconstructed images
        # Obtain posterior (latent distribution) from encoding
        posterior = model.module.encode(input_frame)

        # Compute generator loss (optimizer_idx = 0)
        loss_gen, log_dict_gen = criterion(
            input_frame,
            recon,
            posterior,
            optimizer_idx=0,
            mask=None,
            last_layer=model.module.decoder.conv_out.weight,
            split="train",
        )
        loss_gen.backward()
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), GRADIENT_CLIP_VAL)
        gen_optimizer.step()

        # -------------------------
        # 2) Discriminator update
        # -------------------------
        loss_disc, log_dict_disc = criterion(
            input_frame,
            recon,
            posterior,
            optimizer_idx=1,
            mask=None,
            last_layer=model.module.decoder.conv_out.weight,
            split="train",
        )
        loss_disc.backward()
        torch.nn.utils.clip_grad_norm_(
            criterion.discriminator.parameters(), GRADIENT_CLIP_VAL
        )
        disc_optimizer.step()

        total_gen_loss += loss_gen.item()
        total_disc_loss += loss_disc.item()
        if rank == 0:
            train_bar.set_postfix(
                {
                    "train_gen_loss": f"{loss_gen.item():.3f}",
                    "train_disc_loss": f"{loss_disc.item():.3f}",
                }
            )

        global_step_counter += input_frame.shape[0] * world_size
        count += 1

        if ENABLE_WANDB and rank == 0:
            wandb.log(
                {
                    "train_gen_loss": loss_gen.item(),
                    "train_disc_loss": loss_disc.item(),
                },
                step=global_step_counter,
            )

        if DEBUG_MODE and count > 2:
            if rank == 0:
                print(f"{DEBUG_PRINT_PREFIX}Breaking early due to DEBUG_MODE")
            break

        if scheduler_type == "cosine":
            gen_scheduler.step()
            disc_scheduler.step()

    avg_gen_loss = total_gen_loss / count
    avg_disc_loss = total_disc_loss / count
    avg_total_loss = (total_gen_loss + total_disc_loss) / (2 * count)
    return global_step_counter, avg_total_loss, avg_gen_loss, avg_disc_loss


def validate(
    model,
    criterion,
    device,
    val_bar,
    gen_scheduler,
    disc_scheduler,
    scheduler_type,
    epoch,
):
    """
    Runs a full validation pass.

    Iterates through the validation data to compute generator and discriminator
    losses. It also handles LR scheduler steps for 'plateau' schedulers, logs
    metrics, generates sample reconstructions, and calls the early stopping handler.

    Args:
        model (torch.nn.Module): The AutoencoderKL model.
        criterion (LPIPSWithDiscriminator): The loss function.
        device (torch.device): The computation device.
        val_bar (tqdm): Progress bar for the validation loader.
        gen_scheduler: Learning rate scheduler for the autoencoder.
        disc_scheduler: Learning rate scheduler for the discriminator.
        scheduler_type (str): The type of scheduler.
        epoch (int): The current epoch number, for saving plots.

    Returns:
        A tuple containing the average total, generator, and discriminator
        validation losses.
    """
    model.eval()
    criterion.eval()
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_bar:
            input_frame, metadata = batch

            input_frame = input_frame.to(device)

            outputs_dict = model(input_frame)
            recon = outputs_dict.sample
            posterior = model.module.encode(input_frame)

            # Compute generator loss (optimizer_idx = 0)
            loss_gen, log_dict_gen = criterion(
                input_frame,
                recon,
                posterior,
                optimizer_idx=0,
                mask=None,
                last_layer=model.module.decoder.conv_out.weight,
                split="val",
            )
            # Compute discriminator loss (optimizer_idx = 1)
            loss_disc, log_dict_disc = criterion(
                input_frame,
                recon,
                posterior,
                optimizer_idx=1,
                mask=None,
                last_layer=model.module.decoder.conv_out.weight,
                split="val",
            )

            total_gen_loss += loss_gen.item()
            total_disc_loss += loss_disc.item()
            if rank == 0:
                val_bar.set_postfix(
                    {
                        "val_gen_loss": f"{loss_gen.item():.3f}",
                        "val_disc_loss": f"{loss_disc.item():.3f}",
                    }
                )

            count += 1
            if count == 1 and rank == 0:
                input_frame = input_frame.cpu().detach().numpy()
                recon = recon.cpu().detach().numpy()
                for i in range(input_frame.shape[0]):
                    input_img = input_frame[i, 0, ...]
                    recon_img = recon[i, 0, ...]

                    if NORMALIZE_DATASET:
                        input_img = input_img * 255.0
                        recon_img = recon_img * 255.0

                    metadata_img = metadata[i]

                    fig = plot_pair_frames(
                        input_img,
                        recon_img,
                        meta1=metadata_img,
                        meta2=metadata_img,
                        title="Comparison between Recon and Original",
                        title_frame1="Original",
                        title_frame2="Reconstructed",
                    )
                    os.makedirs(f"{PLOTS_FOLDER}/examples/epoch_{epoch}", exist_ok=True)
                    plt.savefig(
                        f"{PLOTS_FOLDER}/examples/epoch_{epoch}/batch_{count}_{i}.png"
                    )
                    plt.close()
            if DEBUG_MODE and count > 2:
                if rank == 0:
                    print(f"{DEBUG_PRINT_PREFIX}Breaking early due to DEBUG_MODE")
                break

    avg_gen_loss_tensor = torch.tensor(total_gen_loss / count).to(device)
    avg_disc_loss_tensor = torch.tensor(total_disc_loss / count).to(device)
    dist.all_reduce(avg_gen_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_disc_loss_tensor, op=dist.ReduceOp.SUM)
    avg_gen_loss = avg_gen_loss_tensor.item() / world_size
    avg_disc_loss = avg_disc_loss_tensor.item() / world_size
    avg_total_loss = (avg_gen_loss + avg_disc_loss) / 2

    if scheduler_type == "plateau":
        gen_scheduler.step(avg_total_loss)
        disc_scheduler.step(avg_total_loss)

    current_lr = gen_optimizer.param_groups[0]["lr"]
    if ENABLE_WANDB and rank == 0:
        wandb.log(
            {
                "val_total_loss": avg_total_loss,
                "val_gen_loss": avg_gen_loss,
                "val_disc_loss": avg_disc_loss,
                "learning_rate": current_lr,
            }
        )

    if epoch >= WARMUP_GENERATOR_EPOCHS:
        if not criterion.discriminator_active:
            if rank == 0:
                print(DEBUG_PRINT_PREFIX + "Switching to full mode")
            criterion.activate_discriminator()
        else:
            if rank == 0:
                early_stopping(
                    avg_gen_loss,
                    model,
                    gen_optimizer,
                    disc_optimizer,
                    epoch,
                    global_step,
                )

    return avg_total_loss, avg_gen_loss, avg_disc_loss


for epoch in range(NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)
    if rank == 0:
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    else:
        train_bar = train_loader

    global_step, avg_total_train_loss, avg_gen_train_loss, avg_disc_train_loss = (
        train_epoch(
            train_bar,
            global_step,
            model,
            criterion,
            device,
            gen_optimizer,
            disc_optimizer,
            gen_scheduler,
            disc_scheduler,
            SCHEDULER_TYPE,
        )
    )

    if rank == 0:
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
    else:
        val_bar = val_loader

    avg_total_val_loss, avg_gen_val_loss, avg_disc_val_loss = validate(
        model,
        criterion,
        device,
        val_bar,
        gen_scheduler,
        disc_scheduler,
        SCHEDULER_TYPE,
        epoch,
    )
    if rank == 0:
        print(
            f"Finished Epoch {epoch} - Train Loss: {avg_total_train_loss:.3f}, Val Loss: {avg_total_val_loss:.3f}, Gen Train Loss: {avg_gen_train_loss:.3f}, Gen Val Loss: {avg_gen_val_loss:.3f}, Disc Train Loss: {avg_disc_train_loss:.3f}, Disc Val Loss: {avg_disc_val_loss:.3f}"
        )

    stop_signal = torch.tensor(0).to(device)
    if rank == 0 and early_stopping.early_stop:
        print(DEBUG_PRINT_PREFIX + "Early stopping")
        stop_signal = torch.tensor(1).to(device)

    dist.broadcast(stop_signal, src=0)
    if stop_signal == 1:
        break


if rank == 0:
    print(DEBUG_PRINT_PREFIX + "Finished training, run id: " + RUN_ID)

cleanup_ddp()

del (
    model,
    gen_optimizer,
    disc_optimizer,
    gen_scheduler,
    disc_scheduler,
    train_loader,
    val_loader,
    train_bar,
    val_bar,
)
torch.cuda.empty_cache()
