import numpy as np
import argparse
from src.dataset import TinyStoriesDataset
from src.model import Llama2Model
from src.prepare_data import prepare_dataset
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import os.path
import signal


# Loss function
criterion = nn.CrossEntropyLoss()

CHECKPOINT_PATH = "checkpoints/last.pt"

#
# Pause / resume helper functions
#

class PauseRequest:
    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        print("\nPause requested: finishing current step and saving checkpoint...")
        self.requested = True

def save_checkpoint(path, model, optimizer, epoch, it, total_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "it": it,
        "total_loss": total_loss,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }

    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()

    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f"Checkpoint saved: {path} (epoch {epoch + 1}, batch {it})")

def load_checkpoint(path, model, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    print(f"Resumed form {path}: epoch {state['epoch'] + 1}, batch {state['it']}")
    return state["epoch"], state["it"], state["total_loss"]


# Training loop
def train(model, cfg, device, checkpoint_path=CHECKPOINT_PATH, resume=False):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    start_epoch, start_it, start_loss = 0, 0, 0.0
    if resume:
        if os.path.isfile(checkpoint_path):
            start_epoch, start_it, start_loss = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
        else:
            print(f"No checkpoint at {checkpoint_path}, starting from scratch.")

    save_every = cfg.get("save_every", 500) # batches between automatic checkpoints
    pause = PauseRequest()

    if not os.path.isfile("data/train.npy"):
        prepare_dataset()

    for epoch in range(start_epoch, cfg["num_epochs"]):

        resuming_this_epoch = (epoch == start_epoch and start_it > 0)
        total_loss = start_loss if resuming_this_epoch else 0.0
        it = start_it if resuming_this_epoch else 0

        train_dataset = TinyStoriesDataset(
            "data/train.npy",
            cfg["context_length"],
            samples_per_epoch= cfg.get("samples_per_epoch"),
            seed=cfg.get("seed", 0),
            epoch=epoch,
            start_sample=it * cfg["batch_size"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size = cfg["batch_size"]
        )

        print(epoch)
        model.train()

        for x,y in train_loader:
            print(it)
            if cfg["max_iter"] > 0 and it >= cfg["max_iter"]:
                break

            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            it += 1

            if pause.requested:
                save_checkpoint(checkpoint_path, model,optimizer=optimizer, epoch=epoch, it=it, total_loss=total_loss)
                print("Training paused. Re-run with resume=True (--resume) to continue.")
                return False

            if save_every > 0 and it % save_every == 0:
                save_checkpoint(checkpoint_path, model,optimizer=optimizer, epoch=epoch, it=it, total_loss=total_loss)

        average_loss = total_loss / max(it, 1)

        print(
            f"Epoch [{epoch + 1}/{cfg['num_epochs']}], "
            f"Loss: {average_loss:.4f}"
        )

        save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, 0, 0.0)
    return True
