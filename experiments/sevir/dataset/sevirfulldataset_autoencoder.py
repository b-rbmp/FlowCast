"""
PyTorch Dataset for training an autoencoder on the SEVIR dataset.

This dataset class is designed to load SEVIR event data and provide
single, randomly selected frames for autoencoder training.
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DynamicAutoencoderSevirDataset(Dataset):
    """
    SEVIR dataset for autoencoder training.

    Instead of returning sequences, this version returns a single
    randomly selected frame from each event.
    """

    def __init__(
        self,
        meta_csv,
        data_file,
        data_type="vil",
        raw_seq_len=49,
        transform=None,
        channel_last=True,
        debug_mode=False,
        normalize=False,
    ):
        """
        Initializes the dataset.

        Args:
            meta_csv (str): Path to the metadata CSV file.
            data_file (str): Path to the HDF5 data file.
            data_type (str): Key for the data in the HDF5 file.
            raw_seq_len (int): Total frames in a raw event sequence.
            transform (callable): Optional transform to apply to the data.
            channel_last (bool): If True, assumes data is (H, W, T).
            debug_mode (bool): If True, loads only a small subset of data.
            normalize (bool): If True, scales data to [0, 1].
        """
        self.meta_csv = meta_csv
        self.data_file = data_file
        self.data_type = data_type
        self.raw_seq_len = raw_seq_len
        self.transform = transform
        self.channel_last = channel_last
        self.debug_mode = debug_mode
        self.normalize = normalize

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError("No events found in the metadata.")
        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        self.hdf_file = h5py.File(self.data_file, "r")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Retrieves a single random frame from an event.

        Args:
            index (int): Index of the event to sample from.

        Returns:
            A tuple containing the frame tensor (1, H, W) and its metadata dict.
        """
        row = self.metadata.iloc[index]
        file_index = int(row["file_row"]) if "file_row" in row else int(row["file_index"])

        event = self.hdf_file[self.data_type][file_index].astype(np.float32)

        if self.transform is not None:
            event = self.transform(event)

        frame_idx = np.random.randint(0, self.raw_seq_len)
        frame = event[..., frame_idx]

        frame = torch.from_numpy(frame).float().unsqueeze(0)

        if self.normalize:
            frame /= 255.0
            
        return frame, row.to_dict()


def sequential_collate(batch):
    """
    Collate function for `DynamicAutoencoderSevirDataset`.

    Stacks the frames from a batch of samples.

    Args:
        batch (list): A list of (frame, metadata) tuples.

    Returns:
        A tuple containing a batched frame tensor and a list of metadata.
    """
    frame, row = zip(*batch)
    frame = np.stack(frame, axis=0)
    frame = torch.from_numpy(frame).float()
    return frame, list(row)
