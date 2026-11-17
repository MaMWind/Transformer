import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info



class TinyStoriesDataset(IterableDataset):  # stores the data and defines the rules for retrieving it (how to select chunks and create x and y)

    def __init__(self, path, block_size, samples_per_epoch=None, seed=0, epoch=0, start_sample=0):
        self.data= np.load(path, mmap_mode="r")
        self.block_size  = block_size
        self.max_start = len(self.data) - block_size - 1
        if self.max_start < 0:
            raise ValueError(
                f"Data has {len(self.data)} tokens, need more than block_size + 1 = {block_size + 1}."
            )

        self.samples_per_epoch = (samples_per_epoch if samples_per_epoch is not None else max((len(self.data) - 1) // block_size, 1))

        self.seed = seed
        self.epoch = epoch
        self.start_sample = start_sample

    def __len__(self):
        return max(self.samples_per_epoch - self.start_sample, 0)

    def _start_indices(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)

        starts = torch.randint(
            0,
            self.max_start + 1,
            (self.samples_per_epoch,),
            generator=gen
        )

        return starts[self.start_sample:]
    

    def __iter__(self):
        starts = self._start_indices()

        info = get_worker_info()
        if info is not None:
            starts = starts[info.id::info.num_workers]

        for start in starts.tolist():

            chunk = self.data[
                start:start + self.block_size + 1
            ]

            chunk = torch.tensor(
                np.array(chunk),
                dtype=torch.long
            )

            x = chunk[:-1]
            y = chunk[1:]

            yield x, y