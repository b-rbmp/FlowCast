"""
This script preprocesses the complete SEVIR dataset for nowcasting tasks.

It reads the raw SEVIR data, filters out low-quality events, and splits the data
into training, validation, and testing sets based on provided cutoff dates.
The processed data is saved into HDF5 files, with corresponding metadata saved
in CSV files.
"""

import sys
import os
import argparse

sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import h5py
import torch

import common.utils.bicubic_interpolation as bicubic_interpolation


def apply_bicubic_interpolation(data, downsample_factor):
    """
    Downsamples the spatial dimensions of the input data using bicubic interpolation.

    Args:
        data (np.ndarray): The input data tensor with shape (H, W, T), where H is height,
                           W is width, and T is the number of time steps.
        downsample_factor (int): The factor by which to reduce the spatial dimensions.
                                 For example, a factor of 2 will halve the height and width.

    Returns:
        np.ndarray: The downsampled data tensor with shape (H/factor, W/factor, T).
    """

    data = torch.from_numpy(data)

    data = data.permute(2, 0, 1)

    if torch.cuda.is_available():
        data = data.cuda()

    data_downsampled = bicubic_interpolation.imresize(
        data, scale=(1 / downsample_factor)
    )

    if torch.cuda.is_available():
        data_downsampled = data_downsampled.cpu()
        data = data.cpu()

    data_downsampled = data_downsampled.permute(1, 2, 0)

    return data_downsampled.numpy()


def convert_sevir_nowcasting(
    catalog_csv_path="datasets/sevir/data/sevir_complete/CATALOG.csv",
    data_dir="datasets/sevir/data/sevir_complete/data",
    validation_cutoff_date="2019-01-01 00:00:00",
    testing_cutoff_date="2019-06-01 00:00:00",
    output_dir="sevir_full",
    img_type="vis",
    keep_dtype=False,
    downsample_factor=1,
):
    """
    Converts the raw SEVIR dataset into training, validation, and testing sets.

    This function reads the SEVIR catalog, filters out events with a high percentage
    of missing frames, and splits the data based on `validation_cutoff_date` and
    `testing_cutoff_date`. Each data split is saved as an HDF5 file, and its
    corresponding metadata is saved as a CSV file. The metadata includes an

    additional `file_row` column for efficient indexing into the HDF5 file.
    The training and validation data can be spatially downsampled.

    Args:
        validation_cutoff_date (str or datetime): The cutoff date for the training set.
            Events before this date are used for training.
        testing_cutoff_date (str or datetime): The cutoff date for the validation set.
            Events between the validation and testing cutoffs are used for validation.
            Events at or after this date are used for testing.
        output_dir (str): The directory where the processed HDF5 and CSV files will be saved.
        img_type (str, optional): The type of SEVIR image to process (e.g., 'vis', 'vil').
                                  Defaults to "vis".
        keep_dtype (bool, optional): If True, preserves the original data type of the images.
                                     If False, converts images to `np.float32`. Defaults to False.
        downsample_factor (int, optional): The factor for spatial downsampling of training
                                           and validation data. A factor of 1 means no
                                           downsampling. Defaults to 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    training_h5_path = os.path.join(output_dir, "nowcast_training_full.h5")
    validation_h5_path = os.path.join(output_dir, "nowcast_validation_full.h5")
    testing_h5_path = os.path.join(output_dir, "nowcast_testing_full.h5")
    training_meta_csv = os.path.join(output_dir, "nowcast_training_full_META.csv")
    validation_meta_csv = os.path.join(output_dir, "nowcast_validation_full_META.csv")
    testing_meta_csv = os.path.join(output_dir, "nowcast_testing_full_META.csv")

    meta_columns = [
        "id",
        "time_utc",
        "episode_id",
        "event_id",
        "event_type",
        "minute_offsets",
        "llcrnrlat",
        "llcrnrlon",
        "urcrnrlat",
        "urcrnrlon",
        "proj",
        "height_m",
        "width_m",
    ]

    print("Reading catalog CSV...")
    catalog = pd.read_csv(catalog_csv_path, parse_dates=["time_utc"])
    catalog = catalog[catalog["img_type"] == img_type].reset_index(drop=True)
    # Filter out events with pct_missing > 0.01
    catalog = catalog[catalog["pct_missing"] <= 0.05].reset_index(drop=True)

    # Convert cutoff dates to datetime if needed.
    if isinstance(testing_cutoff_date, str):
        testing_cutoff_date = pd.to_datetime(testing_cutoff_date)
    if isinstance(validation_cutoff_date, str):
        validation_cutoff_date = pd.to_datetime(validation_cutoff_date)

    catalog_train = catalog[catalog["time_utc"] < validation_cutoff_date].copy()
    catalog_val = catalog[
        (catalog["time_utc"] >= validation_cutoff_date)
        & (catalog["time_utc"] < testing_cutoff_date)
    ].copy()
    catalog_test = catalog[catalog["time_utc"] >= testing_cutoff_date].copy()

    print(
        f"Total {img_type} events: {len(catalog)}; Training: {len(catalog_train)}; Validation: {len(catalog_val)}; Testing: {len(catalog_test)}"
    )

    train_meta_list = []
    val_meta_list = []
    test_meta_list = []

    train_h5 = None
    val_h5 = None
    test_h5 = None
    train_ds = None
    val_ds = None
    test_ds = None
    train_count = 0
    val_count = 0
    test_count = 0

    def append_event(ds, event, current_count):
        new_count = current_count + 1
        ds.resize((new_count,) + ds.shape[1:])
        ds[current_count, :, :, :] = event
        return new_count

    def process_row(row):
        full_path = os.path.join(data_dir, row["file_name"])
        file_index = int(row["file_index"])
        with h5py.File(full_path, "r") as hf:
            event = hf[img_type][file_index]
            if not keep_dtype:
                event = event.astype(np.float32)
        return event

    print("Processing training events...")
    for i, row in catalog_train.iterrows():
        try:
            event = process_row(row)
            if downsample_factor > 1:
                event = apply_bicubic_interpolation(event, downsample_factor)
        except Exception as e:
            print(f"Error processing row {i} (id {row['id']}): {e}")
            continue
        if train_ds is None:
            train_h5 = h5py.File(training_h5_path, "w")
            H, W, T = event.shape
            train_ds = train_h5.create_dataset(
                img_type,
                shape=(0, H, W, T),
                maxshape=(None, H, W, T),
                dtype=event.dtype,
                chunks=(1, H, W, T),
                compression="gzip",
                compression_opts=4,
            )
        train_count = append_event(train_ds, event, train_count)
        meta = {k: row[k] for k in meta_columns}
        meta["file_row"] = train_count - 1
        train_meta_list.append(meta)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} training events...")
    if train_h5 is not None:
        train_h5.close()
    print(f"Finished processing training events: {train_count} events saved.")

    print("Processing validation events...")
    for i, row in catalog_val.iterrows():
        try:
            event = process_row(row)
            if downsample_factor > 1:
                event = apply_bicubic_interpolation(event, downsample_factor)
        except Exception as e:
            print(f"Error processing row {i} (id {row['id']}): {e}")
            continue
        if val_ds is None:
            val_h5 = h5py.File(validation_h5_path, "w")
            H, W, T = event.shape
            val_ds = val_h5.create_dataset(
                img_type,
                shape=(0, H, W, T),
                maxshape=(None, H, W, T),
                dtype=event.dtype,
                chunks=(1, H, W, T),
                compression="gzip",
                compression_opts=4,
            )
        val_count = append_event(val_ds, event, val_count)
        meta = {k: row[k] for k in meta_columns}
        meta["file_row"] = val_count - 1
        val_meta_list.append(meta)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} validation events...")
    if val_h5 is not None:
        val_h5.close()
    print(f"Finished processing validation events: {val_count} events saved.")

    print("Processing testing events...")
    for i, row in catalog_test.iterrows():
        try:
            event = process_row(row)
        except Exception as e:
            print(f"Error processing row {i} (id {row['id']}): {e}")
            continue
        if test_ds is None:
            test_h5 = h5py.File(testing_h5_path, "w")
            H, W, T = event.shape
            test_ds = test_h5.create_dataset(
                img_type,
                shape=(0, H, W, T),
                maxshape=(None, H, W, T),
                dtype=event.dtype,
                chunks=(1, H, W, T),
                compression="gzip",
                compression_opts=4,
            )
        test_count = append_event(test_ds, event, test_count)
        meta = {k: row[k] for k in meta_columns}
        meta["file_row"] = test_count - 1
        test_meta_list.append(meta)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} testing events...")
    if test_h5 is not None:
        test_h5.close()
    print(f"Finished processing testing events: {test_count} events saved.")

    pd.DataFrame(train_meta_list, columns=meta_columns + ["file_row"]).to_csv(
        training_meta_csv, index=False
    )
    pd.DataFrame(val_meta_list, columns=meta_columns + ["file_row"]).to_csv(
        validation_meta_csv, index=False
    )
    pd.DataFrame(test_meta_list, columns=meta_columns + ["file_row"]).to_csv(
        testing_meta_csv, index=False
    )
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for testing FlowCast model.")
    
    parser.add_argument("--catalog_csv_path", type=str, default="datasets/sevir/data/sevir_complete/CATALOG.csv")
    parser.add_argument("--data_dir", type=str, default="datasets/sevir/data/sevir_complete/data")
    parser.add_argument("--val_cutoff", type=str, default="2019-01-01 00:00:00")
    parser.add_argument("--test_cutoff", type=str, default="2019-06-01 00:00:00")
    parser.add_argument("--output_dir", type=str, default="sevir_full")
    parser.add_argument("--img_type", type=str, default="vil")
    parser.add_argument("--keep_dtype", type=bool, default=True)
    parser.add_argument("--downsample_factor", type=int, default=1)

    args = parser.parse_args()


    convert_sevir_nowcasting(
        args.catalog_csv_path,
        args.data_dir,
        args.val_cutoff,
        args.test_cutoff,
        args.output_dir,
        args.img_type,
        args.keep_dtype,
        args.downsample_factor,
    )

