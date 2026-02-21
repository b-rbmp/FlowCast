"""
PyTorch Dataset classes for the SEVIR dataset, tailored for nowcasting tasks.

Includes two main dataset handlers:
1. `DynamicSequentialSevirDataset`: For raw SEVIR image data. Extracts sequences
   using a sliding window and splits them into input/target tensors.
2. `DynamicEncodedSequentialSevirDataset`: Similar to the above, but for data that
   has been pre-processed by an autoencoder into a latent representation.

Also provides corresponding collate functions.
"""

import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset


def post_process_samples(
    samples: np.ndarray, clamp_min: int = 0, clamp_max: int = 255
) -> np.ndarray:
    """
    Clamps the values in a numpy array to a given range.

    Args:
        samples (np.ndarray): Input array.
        clamp_min (int): Minimum value.
        clamp_max (int): Maximum value.

    Returns:
        np.ndarray: The clamped array.
    """
    processed_samples = np.clip(samples, a_min=clamp_min, a_max=clamp_max)
    return processed_samples


class DynamicSequentialSevirDataset(Dataset):
    """
    Dataset for SEVIR nowcasting that extracts sequential windows from raw image data.

    Uses a sliding-window approach to generate multiple samples from a single event.
    Each window is split into an input (lag frames) and a target (lead frames).
    Supports subsampling of frames via `time_spacing`.

    Assumes raw event data is shaped (H, W, T).
    """
    def __init__(
        self,
        meta_csv,
        data_file,
        data_type="vil",
        raw_seq_len=49,
        lag_time=13,
        lead_time=12,
        time_spacing=1,
        stride=12,
        transform=None,
        channel_last=True,
        debug_mode=False,
    ):
        """
        Initializes the dataset.

        Args:
            meta_csv (str): Path to the metadata CSV.
            data_file (str): Path to the HDF5 data file.
            data_type (str): Key for the data in the HDF5 file (e.g., "vil").
            raw_seq_len (int): Total frames in a raw event sequence.
            lag_time (int): Number of input frames.
            lead_time (int): Number of output frames to predict.
            time_spacing (int): Interval for subsampling frames (e.g., 2 means every other frame).
            stride (int): Step size for the sliding window.
            transform (callable): Optional transform to apply to each segment.
            channel_last (bool): If true, output tensors are (T, H, W, C).
            debug_mode (bool): If true, loads a small subset of data.
        """
        self.meta_csv = meta_csv
        self.data_file = data_file
        self.data_type = data_type
        self.raw_seq_len = raw_seq_len
        self.lag_time = lag_time
        self.lead_time = lead_time
        self.time_spacing = time_spacing
        self.seq_len = (lag_time + lead_time) * self.time_spacing
        self.stride = stride
        self.transform = transform
        self.channel_last = channel_last
        self.debug_mode = debug_mode

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError(f"No events found in the metadata.")
        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        self.hdf_file = h5py.File(self.data_file, "r")

        if self.raw_seq_len < self.seq_len:
            raise ValueError(
                "raw_seq_len must be at least (lag_time+lead_time)*time_spacing"
            )
        self.n_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride
        if self.debug_mode:
            self.n_seq_per_event = 1

        self.event_seq_counts = np.full(
            len(self.metadata), self.n_seq_per_event, dtype=np.int32
        )
        self.cum_counts = np.cumsum(self.event_seq_counts)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        Maps an index to an event and a window within it. Reads the data, extracts
        the segment, splits it into input (X) and target (Y) tensors, and returns them.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            A tuple containing the input tensor (X), target tensor (Y), and metadata dict.
        """
        event_idx = int(np.searchsorted(self.cum_counts, index, side="right"))
        if event_idx == 0:
            seq_idx = index
        else:
            seq_idx = index - int(self.cum_counts[event_idx - 1])

        row = self.metadata.iloc[event_idx]
        file_index = (
            int(row["file_row"]) if "file_row" in row else int(row["file_index"])
        )

        event = self.hdf_file[self.data_type][file_index]
        event = event.astype(np.float32)

        start = seq_idx * self.stride
        segment = event[..., start : start + self.seq_len]

        if self.transform is not None:
            segment = self.transform(segment)

        total_timesteps = self.seq_len
        y_end = total_timesteps - 1
        y_indices = [y_end - i * self.time_spacing for i in range(self.lead_time)]
        y_indices.reverse()
        Y = segment[..., y_indices]
        x_indices = [i * self.time_spacing for i in range(self.lag_time)]
        X = segment[..., x_indices]

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

        X = X.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)

        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

        if self.channel_last:
            X = X.permute(1, 2, 3, 0)
            Y = Y.permute(1, 2, 3, 0)

        return X, Y, row.to_dict()


def dynamic_sequential_collate(batch):
    """
    Collate function for `DynamicSequentialSevirDataset`.

    Stacks the tensors from a batch of samples.

    Args:
        batch (list): A list of (X, Y, metadata) tuples.

    Returns:
        A tuple containing batched X tensors, batched Y tensors, and a list of metadata.
    """
    X_list, Y_list, meta_list = zip(*batch)
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    return X, Y, list(meta_list)


class DynamicEncodedSequentialSevirDataset(Dataset):
    """
    Dataset for pre-encoded (latent) SEVIR data.

    Similar to `DynamicSequentialSevirDataset` but for data that has already
    been processed by an autoencoder. Assumes input data has a channel
    dimension, e.g., (H, W, T, C).
    """

    def __init__(
        self,
        meta_csv,
        data_file,
        data_type="vil",
        raw_seq_len=49,
        lag_time=13,
        lead_time=12,
        time_spacing=1,
        stride=12,
        transform=None,
        channel_last=True,
        debug_mode=False,
    ):
        """
        Initializes the dataset for encoded data.

        Args:
            meta_csv (str): Path to the metadata CSV.
            data_file (str): Path to the HDF5 data file.
            data_type (str): Key for the data in the HDF5 file.
            raw_seq_len (int): Total frames in a raw event sequence.
            lag_time (int): Number of input frames.
            lead_time (int): Number of output frames to predict.
            time_spacing (int): Interval for subsampling frames.
            stride (int): Step size for the sliding window.
            transform (callable): Optional transform to apply.
            channel_last (bool): If true, output tensors are (T, H, W, C).
            debug_mode (bool): If true, loads a small subset of data.
        """
        self.meta_csv = meta_csv
        self.data_file = data_file
        self.data_type = data_type
        self.raw_seq_len = raw_seq_len
        self.lag_time = lag_time
        self.lead_time = lead_time
        self.time_spacing = time_spacing
        self.seq_len = (lag_time + lead_time) * self.time_spacing
        self.stride = stride
        self.transform = transform
        self.channel_last = channel_last
        self.debug_mode = debug_mode

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError(f"No events found in {self.meta_csv}.")

        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        self.hdf_file = h5py.File(self.data_file, "r")

        if self.raw_seq_len < self.seq_len:
            raise ValueError("raw_seq_len must be >= (lag_time+lead_time)*time_spacing")

        self.n_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride

        self.event_seq_counts = np.full(
            len(self.metadata), self.n_seq_per_event, dtype=np.int32
        )
        self.cum_counts = np.cumsum(self.event_seq_counts)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, index):
        """
        Retrieves a single sample of encoded data.

        Maps an index to an event and window, reads the data, extracts the segment,
        splits it into input (X) and target (Y) latent tensors, and returns them.

        Args:
            index (int): Index of the sample.

        Returns:
            A tuple containing the input tensor (X), target tensor (Y), and metadata dict.
        """
        event_idx = int(np.searchsorted(self.cum_counts, index, side="right"))
        if event_idx == 0:
            seq_idx = index
        else:
            seq_idx = index - int(self.cum_counts[event_idx - 1])

        row = self.metadata.iloc[event_idx]
        file_index = (
            int(row["file_row"]) if "file_row" in row else int(row["file_index"])
        )

        event = self.hdf_file[self.data_type][file_index]
        event = event.astype(np.float32)
        start = seq_idx * self.stride
        end = start + self.seq_len
        segment = event[..., start:end, :]

        if self.transform is not None:
            segment = self.transform(segment)

        total_timesteps = self.seq_len
        x_indices = [i * self.time_spacing for i in range(self.lag_time)]
        y_end = total_timesteps - 1
        y_indices = [y_end - i * self.time_spacing for i in range(self.lead_time)]
        y_indices.reverse()

        X = segment[..., x_indices, :]
        Y = segment[..., y_indices, :]

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y)

        if self.channel_last:
            X = X.permute(2, 0, 1, 3)
            Y = Y.permute(2, 0, 1, 3)
        else:
            X = X.permute(3, 2, 0, 1)
            Y = Y.permute(3, 2, 0, 1)

        return X, Y, row.to_dict()


def dynamic_encoded_sequential_collate(batch):
    """
    Collate function for `DynamicEncodedSequentialSevirDataset`.

    Stacks tensors from a batch of samples.

    Args:
        batch (list): A list of (X, Y, metadata) tuples.

    Returns:
        A tuple of batched X tensors, batched Y tensors, and a list of metadata.
    """
    X_list, Y_list, meta_list = zip(*batch)
    X = torch.stack(X_list, dim=0)
    Y = torch.stack(Y_list, dim=0)
    return X, Y, list(meta_list)
