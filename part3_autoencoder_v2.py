# sparse_autoencoder_train.py (FIXED VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os


class RobustSparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        # Wider bottleneck for better feature separation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 6),  # Increased from 4x
            nn.LeakyReLU(0.1),  # Better gradient flow
            nn.Linear(input_dim * 6, input_dim * 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 12, input_dim * 6),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim * 6, input_dim)
        )
        self.l1_penalty = 0.001  # Increased from 0.0001


def load_and_normalize_activations(activations_dir):
    """Load and preprocess activations with tighter controls"""
    activations = torch.load(os.path.join(activations_dir, "all_activations.pt"))

    # More aggressive normalization
    activations = (activations - activations.mean()) / (activations.std() + 1e-8)
    activations = activations / 3.0  # Force smaller range
    activations = torch.clamp(activations, min=-3.0, max=3.0)

    print(f"Normalized - Mean: {activations.mean():.2f}, Std: {activations.std():.2f}")
    print(f"Clamped Range: {activations.min():.2f} to {activations.max():.2f}")
    return activations.float()


def train_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and normalize data
    activations = load_and_normalize_activations(args.activations_dir)
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    input_dim = activations.shape[-1]
    autoencoder = RobustSparseAutoencoder(
        input_dim,
        encoder_dim=args.encoder_dim,
        l1_penalty=args.l1_penalty
    ).to(device)

    # Optimizer with very low LR
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        autoencoder.train()
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = batch[0].to(device)
            optimizer.zero_grad()

            # Forward pass
            recon, l1_loss = autoencoder(x)
            loss = F.mse_loss(recon, x) + l1_loss

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 0.05)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation (using encode() method)
        with torch.no_grad():
            val_data = activations[-len(activations) // 10:].to(device)
            recon, l1_loss = autoencoder(val_data)
            val_loss = F.mse_loss(recon, val_data) + l1_loss.item()

            # Sparsity check using encode()
            sample = activations[:100].to(device)
            encoded = autoencoder.encode(sample)
            sparsity = (encoded.abs() < 0.01).float().mean()

        # Update scheduler
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {epoch_loss / len(loader):.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Sparsity: {sparsity:.1%}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(autoencoder.state_dict(), args.save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", default="merged_activations")
    parser.add_argument("--save_path", default="sparse_ae.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoder_dim", type=int, default=3)  # Reduced
    parser.add_argument("--l1_penalty", type=float, default=0.00005)  # Lower
    parser.add_argument("--lr", type=float, default=3e-7)  # Ultra-low
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train_autoencoder(args)