import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader



class TinyStoriesDataset(IterableDataset):  # stores the data and defines the rules for retrieving it (how to select chunks and create x and y)

    def __init__(self, path, block_size):
        self.data = np.load(
            path,
            mmap_mode="r"
        )

        self.block_size = block_size

    def __iter__(self):

        max_start = len(self.data) - self.block_size - 1

        while True:

            start = torch.randint(
                0,
                max_start + 1,
                (1,)
            ).item()

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
