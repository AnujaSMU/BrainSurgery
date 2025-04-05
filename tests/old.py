import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class RobustSparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 6),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim * 6, input_dim * 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 12, input_dim * 6),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim * 6, input_dim)
        )

    def forward(self, x):
        x = self.input_norm(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, torch.norm(encoded, p=1) * 0.001 / x.shape[0]

    def encode(self, x):
        x = self.input_norm(x)
        return self.encoder(x)

def load_model(model_path="models/sparse_ae.pth", input_dim=1152):
    """Load the trained sparse autoencoder model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustSparseAutoencoder(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def analyze_activations(model, activations):
    """Analyze activations using the sparse autoencoder"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = activations.to(device)
    
    with torch.no_grad():
        # Normalize activations
        activations = (activations - activations.mean()) / (activations.std() + 1e-8)
        activations = activations / 3.0
        activations = torch.clamp(activations, min=-3.0, max=3.0)
        
        # Get encoded features
        encoded = model.encode(activations)
        decoded, _ = model(activations)
        
        # Calculate reconstruction error
        recon_error = F.mse_loss(decoded, activations)
        
        # Calculate sparsity metrics
        sparsity = (encoded.abs() < 0.01).float().mean()
        
        return {
            'encoded_features': encoded,
            'reconstructed': decoded,
            'recon_error': recon_error.item(),
            'sparsity': sparsity.item()
        }

def main():
    # Load test activation
    activation_path = "activation_dataset/activations_0.pt"
    if not os.path.exists(activation_path):
        raise FileNotFoundError(f"No activation data found at {activation_path}")
    
    activations = torch.load(activation_path)
    print(f"Loaded activations shape: {activations.shape}")

    # Load model
    model = load_model()
    print("Loaded sparse autoencoder model")

    # Analyze activations
    results = analyze_activations(model, activations)
    
    print("\nAnalysis Results:")
    print(f"Reconstruction Error: {results['recon_error']:.6f}")
    print(f"Sparsity (% features < 0.01): {results['sparsity']*100:.2f}%")
    print(f"Encoded Features Shape: {results['encoded_features'].shape}")
    
    # Optional: Show top activated features
    encoded_features = results['encoded_features']
    feature_activations = encoded_features.abs().mean(dim=0)
    top_features = torch.topk(feature_activations, k=10)
    
    print("\nTop 10 Most Active Features:")
    for idx, value in zip(top_features.indices.tolist(), top_features.values.tolist()):
        print(f"Feature {idx}: {value:.4f}")

if __name__ == "__main__":
    main()