import numpy as np
import argparse
from dataset import TinyStoriesDataset
from model import Llama2Model
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr",type=float, default=0.001)
    parser.add_argument("--epochs",type=int, default=10)
    parser.add_argument("--vocab",type=int, default=32000)
    parser.add_argument("--context",type=int, default=2048)
    parser.add_argument("--emb_dim",type=int, default=512)   # this has to be divisible by the number of attention heads
    parser.add_argument("--n_heads",type=int, default=8)
    parser.add_argument("--n_layers",type=int, default=8)
    parser.add_argument("--hidden_dim",type=int, default=1344)   #(8/3) * emb_dim rounded to a multiple of 64
    parser.add_argument("--max_iter", type=int, default=-1)
    parser.add_argument("--subset", type=int)
    return parser.parse_args()


# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
def train_loop(num_epochs, criterion, optimizer, model, train_loader, max_iter, device):
    for epoch in range(num_epochs):
        
        print(epoch)
        model.train()
        total_loss = 0
        it = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            print(x.shape)
            if max_iter > 0 and it > max_iter:
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
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
        )
    
def main():
    block_size = 256
    batch_size = 32
    args = get_args()

    train_dataset = TinyStoriesDataset( 
    "train.npy",
    block_size
    )

    train_loader = DataLoader(       # requests samples from the Dataset through an iterator and combines them into batches
    train_dataset,
    batch_size=batch_size
    )


    cnfg = {
        "vocab_size": args.vocab,    
        "context_length": args.context,  
        "emb_dim": 768,         
        "n_heads": args.n_heads,           
        "n_layers": args.n_layers,          
        "hidden_dim": args.hidden_dim,     
        "dtype": torch.bfloat16
    }

    model = Llama2Model(cnfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loop(args.epochs, criterion, optimizer, model, train_loader, args.max_iter, device)

# Save model
    torch.save(model.state_dict(), "model.pth")



if __name__ == "__main__":
    main()