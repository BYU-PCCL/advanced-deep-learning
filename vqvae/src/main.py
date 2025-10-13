import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.components import Encoder, Decoder, ResBlock
from src.fsq import FSQ
from src.vae import VAE
from src.vqvae import VQVAE
from src.my_vector_quantizer import MyVectorQuantizer

# -------------------
# Command Line Arguments
# -------------------
parser = argparse.ArgumentParser(description="Run VAE experiments")
parser.add_argument('--data', action='store_true', help='Only download the dataset and exit')
parser.add_argument('--exp', action='append', choices=['vae', 'vqvae', 'myvqvae', 'fsq'], help='Experiments to run (can specify multiple)')
args = parser.parse_args()

# -------------------
# Setup and Data Loading
# -------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.STL10(root="./data", split="train", download=True, transform=transform)
test_set  = torchvision.datasets.STL10(root="./data", split="test",  download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

if args.data:
    print("Dataset downloaded. Exiting as --data flag was set.")
    sys.exit(0)

# -------------------
# Training Utilities
# -------------------
epochs = 8
lr = 3e-4

def train_epoch(model, loader, opt, epoch, name, total_epochs, pbar):
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if isinstance(model, VAE):
            # KL annealing: start small, increase to 1.0
            beta = min(1.0, epoch / max(1, total_epochs//3))
            _, loss = model(x, beta=beta)
        else:
            _, loss, _ = model(x)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        pbar.update(1)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, epoch, total_epochs, pbar):
    model.eval()
    total = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if isinstance(model, VAE):
            beta = 1.0
            _, loss = model(x, beta=beta)
        else:
            _, loss, _ = model(x)
        total += loss.item() * x.size(0)
        pbar.update(1)
    return total / len(loader.dataset)

def train_and_eval(model, name):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    total_train_steps = epochs * len(train_loader)
    total_eval_steps = epochs * len(test_loader)
    print(f"Starting training for {name}")
    train_pbar = tqdm(total=total_train_steps, desc=f"{name} Train", position=0)
    eval_pbar = tqdm(total=total_eval_steps, desc="Eval", position=1)
    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, opt, ep, name, epochs, train_pbar)
        te = eval_epoch(model, test_loader, ep, epochs, eval_pbar)
        train_losses.append(tr); test_losses.append(te)
        train_pbar.set_postfix(epoch=f"{ep:02d}/{epochs}", train_loss=f"{tr:.4f}", test_loss=f"{te:.4f}")
    train_pbar.close()
    eval_pbar.close()
    return train_losses, test_losses, model



# -------------------
# Run Training
# -------------------
experiments = {
    'vae': ('VAE', VAE(latent_dim=512)),
    'vqvae': ('VQ-VAE', VQVAE(dim=128, codebook_size=512, custom=False)),
    'myvqvae': ('MyVQ-VAE', VQVAE(dim=128, codebook_size=512, custom=True)),
    # 'fsq': ('FSQ', FSQ(dim=128, levels=16))
}

results = {}
if args.exp:
    for exp in args.exp:
        name, model = experiments[exp]
        results[name] = train_and_eval(model, name)
else:
    for exp, (name, model) in experiments.items():
        results[name] = train_and_eval(model, name)

# -------------------
# Plot Loss Curves
# -------------------
os.makedirs('images', exist_ok=True)
plt.figure(figsize=(10, 6))
for name, (tr, te, _) in results.items():
    plt.plot(tr, label=f"{name} train")
    plt.plot(te, '--', label=f"{name} test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Test Loss (STL10, 96x96)")
plt.legend()
plt.grid(True)
plt.savefig('images/loss_curves.png')

# -------------------
# Visualization
# -------------------
@torch.no_grad()
def show_one(results):
    x, _ = next(iter(test_loader))
    x = x[:1].to(device)
    orig = x[0].permute(1, 2, 0).cpu().numpy()

    for name, (_, _, model) in results.items():
        model.eval()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        if isinstance(model, VQVAE):
            x_hat, _, info = model(x)
            tokens, vocab, indices = info
            title = f"{name}\nTokens={tokens}, Vocab={vocab}"
        elif isinstance(model, FSQ):
            x_hat, _, info = model(x)
            tokens, levels, _ = info
            title = f"{name}\nTokens={tokens}, Levels={levels}"
        else:  # VAE
            x_hat, _ = model(x, beta=1.0)
            title = f"{name}\nLatent={model.latent_dim}, Vocab=N/A"

        dec = x_hat[0].permute(1, 2, 0).cpu().numpy()
        
        axs[0].imshow(orig)
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].text(0.5, 0.5, title, ha="center", va="center", fontsize=11)
        axs[1].axis("off")

        axs[2].imshow(dec)
        axs[2].set_title("Decoded")
        axs[2].axis("off")

        plt.tight_layout()
        filename = name.lower().replace(' ', '').replace('-', '') + '.png'
        plt.savefig(f'images/{filename}')
        plt.close(fig)  # Close to free memory

show_one(results)
