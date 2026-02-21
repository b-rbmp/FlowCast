"""
Generates a latent-space dataset from SEVIR radar data using a pre-trained
autoencoder.

This script reads SEVIR HDF5 files, encodes each event sequence into its
latent representation, and saves the results to new HDF5 files. This creates
a smaller, more efficient dataset for training downstream generative models like FlowCast.
"""


import os
import sys

sys.path.append(os.getcwd())
import argparse

import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from diffusers.models.autoencoders import AutoencoderKL
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(
    description="Generate latent HDF5 dataset named 'vil' for training & validation."
)

parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/autoencoder/autoencoder_kl_config.yaml",
    help="Path to the YAML configuration file.",
)
parser.add_argument(
    "--preload_model",
    type=str,
    default="saved_models/sevir/autoencoder/models/early_stopping_model.pt",
    help="Path to saved autoencoder model",
)
parser.add_argument("--data_folder", type=str, default="sevir_full", help="Data Folder")
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
parser.add_argument(
    "--out_dir", type=str, default="datasets/sevir/data/sevir_latent_vae"
)

args = parser.parse_args()

config = OmegaConf.load(args.config)
model_params = config.model_params
training_params = config.training_params
run_params = config.run_params

DEBUG_MODE = run_params.debug_mode
DEBUG_PRINT_PREFIX = "[DEBUG] " if DEBUG_MODE else ""

TRAIN_FILE = args.train_file
TRAIN_META = args.train_meta
VAL_FILE = args.val_file
VAL_META = args.val_meta
OUT_DIR = args.out_dir

OUT_TRAIN_H5 = os.path.join(OUT_DIR, "nowcast_training_full.h5")
OUT_VAL_H5 = os.path.join(OUT_DIR, "nowcast_validation_full.h5")

OUT_TRAIN_META = os.path.join(OUT_DIR, "nowcast_training_full_META.csv")
OUT_VAL_META = os.path.join(OUT_DIR, "nowcast_validation_full_META.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEBUG_PRINT_PREFIX}Using device: {device}")

if not os.path.exists(args.preload_model):
    raise FileNotFoundError(f"Model not found at {args.preload_model}")


model = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    down_block_types=list(model_params.down_block_types),
    up_block_types=list(model_params.up_block_types),
    block_out_channels=list(model_params.block_out_channels),
    act_fn=model_params.act_fn,
    latent_channels=model_params.latent_channels,
    norm_num_groups=model_params.norm_num_groups,
    layers_per_block=model_params.layers_per_block,
)

checkpoint = torch.load(args.preload_model, map_location=device)
state_dict = checkpoint["model_state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

def encode_event(event_3d, model, autoenc_type):
    """
    Encodes a single event sequence into its latent representation.

    Args:
        event_3d (np.ndarray): The event data, shaped (H, W, T).
        model (torch.nn.Module): The pre-trained autoencoder.
        autoenc_type (str): The autoencoder type (e.g., "vae"), which determines
                            how the latent code is extracted.

    Returns:
        np.ndarray: The latent representation of the event, shaped (h, w, T, C).
    """
    H, W, T = event_3d.shape

    frames = torch.from_numpy(event_3d).float().permute(2, 0, 1).unsqueeze(1).to(device)
    with torch.no_grad():
        latent = model.encode(frames)
        if autoenc_type == "vae":
            latent = latent.latent_dist.mode()
    latent = latent.cpu()

    T_, C, h, w = latent.shape
    if T_ != T:
        print(f"Warning: mismatch in time dimension: got {T_}, expected {T}.")

    latent = latent.permute(2, 3, 0, 1)
    return latent.numpy().astype(np.float32)


def create_latent_h5(
    in_h5_path,
    in_meta_csv,
    out_h5_path,
    out_meta_csv,
    model,
    autoenc_type,
    debug_mode=False,
):
    """
    Creates a new HDF5 file containing the latent representations of events.

    Reads events from an input HDF5 file, encodes them, and saves the latents
    to a new HDF5 file. Also updates the corresponding metadata CSV.

    Args:
        in_h5_path (str): Path to the input HDF5 file.
        in_meta_csv (str): Path to the input metadata CSV.
        out_h5_path (str): Path for the output HDF5 file.
        out_meta_csv (str): Path for the output metadata CSV.
        model (torch.nn.Module): The pre-trained autoencoder.
        autoenc_type (str): The autoencoder type ("vae").
        debug_mode (bool): If True, process only a small subset of events.
    """
    if not os.path.exists(in_h5_path):
        print(f"Input file {in_h5_path} does not exist. Skipping.")
        return

    if not os.path.exists(in_meta_csv):
        print(f"Input meta {in_meta_csv} does not exist. Skipping.")
        return

    meta_df = pd.read_csv(in_meta_csv)
    if len(meta_df) == 0:
        print(f"No rows in {in_meta_csv}, skipping.")
        return

    os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)
    if os.path.exists(out_h5_path):
        os.remove(out_h5_path)

    dataset_name = "vil"
    with h5py.File(in_h5_path, "r") as in_h5:
        if dataset_name not in in_h5:
            print(
                f"ERROR: dataset '{dataset_name}' not found in {in_h5_path}. Skipping."
            )
            return
        in_data = in_h5[dataset_name]
        N, H, W, T = in_data.shape

        out_h5 = h5py.File(out_h5_path, "w")

        dset = None

        new_meta_list = []
        event_count = 0

        for i, row in tqdm(
            meta_df.iterrows(), total=len(meta_df), desc=f"Encoding {in_h5_path}"
        ):
            old_file_row = row["file_row"]
            if old_file_row >= N:
                print(
                    f"Skipping row {i}, file_row={old_file_row} out of range (N={N})."
                )
                continue

            event_3d = in_data[old_file_row]

            event_3d = event_3d.astype(np.float32)
            if training_params.normalize_dataset:
                event_3d /= 255.0

            latents_4d = encode_event(event_3d, model, autoenc_type)

            if np.isnan(latents_4d).any():
                print(
                    f"Warning: NaN found in encoded event {i}, file_row={old_file_row}. Skipping."
                )
                continue
            if dset is None:
                h, w, t_new, c_new = latents_4d.shape
                dset = out_h5.create_dataset(
                    dataset_name,
                    shape=(0, h, w, t_new, c_new),
                    maxshape=(None, h, w, t_new, c_new),
                    dtype=latents_4d.dtype,
                    chunks=(1, h, w, t_new, c_new),
                    compression="gzip",
                    compression_opts=4,
                )

            dset.resize((event_count + 1,) + dset.shape[1:])
            dset[event_count, ...] = latents_4d

            new_row = row.copy()
            new_row["file_row"] = event_count
            new_meta_list.append(new_row)

            event_count += 1
            if debug_mode and event_count >= 50:
                print(
                    f"{DEBUG_PRINT_PREFIX}Stopping early after 50 events (debug_mode)."
                )
                break

        out_h5.close()

    new_meta_df = pd.DataFrame(new_meta_list)
    new_meta_df.to_csv(out_meta_csv, index=False)
    print(f"Wrote {event_count} events to {out_h5_path} with a single dataset 'vil'.")
    print(f"New metadata => {out_meta_csv}")


create_latent_h5(
    in_h5_path=TRAIN_FILE,
    in_meta_csv=TRAIN_META,
    out_h5_path=OUT_TRAIN_H5,
    out_meta_csv=OUT_TRAIN_META,
    model=model,
    autoenc_type="vae",
    debug_mode=DEBUG_MODE,
)

create_latent_h5(
    in_h5_path=VAL_FILE,
    in_meta_csv=VAL_META,
    out_h5_path=OUT_VAL_H5,
    out_meta_csv=OUT_VAL_META,
    model=model,
    autoenc_type="vae",
    debug_mode=DEBUG_MODE,
)

print("Done generating 'vil' latent-only HDF5 files for train & validation.")
