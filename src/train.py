import numpy as np
import argparse
from src.dataset import TinyStoriesDataset
from src.model import Llama2Model
from src.prepare_data import prepare_dataset
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import os.path


# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model,cfg, device):
    for epoch in range(cfg["num_epochs"]):

        if not os.path.isfile("data/train.npy"):
            prepare_dataset()

        train_dataset = TinyStoriesDataset( 
            "data/train.npy",
            256
        )
        
        train_loader = DataLoader(       # requests samples from the Dataset through an iterator and combines them into batches
            train_dataset,
            batch_size=32
        )
        
        print(epoch)
        model.train()
        total_loss = 0
        it = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr = cfg["lr"])
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            print(x.shape)
            if cfg["max_iter"] > 0 and it > cfg["max_iter"]:
                break
        # Reset gradients
            optimizer.zero_grad()

        # Forward pass
            logits = model(x)

        # Calculate loss
            loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        # Backpropagation
            loss.backward()

        # Update weights
            optimizer.step()

            total_loss += loss.item()
            it+=1   

        average_loss = total_loss / len(train_loader)

        print(
        f"Epoch [{epoch + 1}/{cfg["num_epochs"]}], "
        f"Loss: {average_loss:.4f}"
        )