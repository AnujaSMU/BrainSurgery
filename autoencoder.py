# train_autoencoder.py
import torch
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoder_dim=4, l1_penalty=0.001):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),  # 4x expansion
            nn.ReLU(),  # Smoother than ReLU
            nn.Linear(input_dim * 4, input_dim * 8)  # Higher capacity
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 8, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.l1_penalty = l1_penalty  # Reduced from 0.001

    def forward(self, x):
        encoded = self.encoder(x)
        l1_loss = torch.norm(encoded, p=1) * self.l1_penalty / x.shape[0]
        decoded = self.decoder(encoded)
        return decoded, l1_loss


def load_activations(activations_dir):
    """Load concatenated activations from dataset directory"""
    activations_path = os.path.join(activations_dir, "all_activations.pt")
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"No activation data found at {activations_path} - run data generation first!")
    return torch.load(activations_path)


def train_autoencoder(activations, args):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    activations = activations.view(-1, activations.shape[-1]).float()
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    autoencoder = SparseAutoencoder(
        input_dim=activations.shape[-1],
        encoder_dim=args.encoder_dim,
        l1_penalty=args.l1_penalty
    ).to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        autoencoder.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = batch[0].to(device)
            optimizer.zero_grad()

            x_recon, l1_loss = autoencoder(x)
            recon_loss = criterion(x_recon, x)
            loss = recon_loss + l1_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(autoencoder.state_dict(), args.save_path)
            print(f"Saved new best model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on LLM activations")
    parser.add_argument("--activations_dir", type=str, default="merged_activations",
                        help="Path to directory with activation data")
    parser.add_argument("--save_path", type=str, default="sparse_autoencoder.pth",
                        help="Path to save trained autoencoder")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--encoder_dim", type=int, default=4,
                        help="Multiplier for encoder hidden dimension (encoder_dim * input_dim)")
    parser.add_argument("--l1_penalty", type=float, default=0.001,
                        help="Weight for L1 regularization")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    args = parser.parse_args()

    # Load pre-generated activations
    activations = load_activations(args.activations_dir)
    print(f"Loaded activations with shape: {activations.shape}")

    activations = (activations - activations.mean()) / (activations.std() + 1e-8)
    activations = torch.clamp(activations, min=-10, max=10)  # Hard clip outliers

    # Verify
    print(f"Post-Norm Stats - Mean: {activations.mean():.2f}, Std: {activations.std():.2f}")
    print(f"Value Range: {activations.min():.2f} to {activations.max():.2f}")


    # Start training
    train_autoencoder(activations, args)