import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class RobustSparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        # Reducing multipliers from 6/12 to 3/6 to match saved model
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 3),  # Changed from 6 to 3
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim * 3, input_dim * 6)  # Changed from 12 to 6
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 6, input_dim * 3),  # Changed from 12 to 6, 6 to 3
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim * 3, input_dim)  # Changed from 6 to 3
        )
        self.l1_penalty = 0.001

    def forward(self, x):
        x = self.input_norm(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, torch.norm(encoded, p=1) * self.l1_penalty / x.shape[0]

    def encode(self, x):
        x = self.input_norm(x)
        return self.encoder(x)

def load_model(model_path="models/sparse_ae.pth", input_dim=1152):
    """Load the trained sparse autoencoder model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustSparseAutoencoder(input_dim).to(device)
    # Load state dict and convert to float32
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def analyze_activations(model, activations):
    """Analyze activations using the sparse autoencoder"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = activations.to(device).float()  # Ensure float32
    
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

def analyze_feature(model, activations, texts, feature_idx, top_k=10):
    """Analyze a specific feature by finding inputs that activate it most strongly"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = activations.to(device).float()
    
    with torch.no_grad():
        # Get encoded features
        encoded = model.encode(activations)
        
        # Get activations for specific feature
        feature_activations = encoded[:, feature_idx]
        
        # Find top k activating examples
        top_indices = torch.topk(feature_activations, k=min(top_k, len(feature_activations)))
        
        results = []
        for idx, activation_strength in zip(top_indices.indices, top_indices.values):
            idx_item = idx.item()
            # Check if the index is valid for the texts list
            text = texts[idx_item] if idx_item < len(texts) else f"[Text not available for index {idx_item}]"
            results.append({
                'text': text,
                'activation': activation_strength.item(),
                'original_idx': idx_item
            })
            
        return results

def main():
    # Load test activation and corresponding texts
    activation_path = "activation_dataset/activations_0.pt"
    prompts = [
        "Once upon a time in a distant land",  # To explore story structure, named entities, and sequence continuity
        "What is the capital of France?",  # To probe knowledge recall and factual associations
        "Write a Python function to reverse a string.",  # To test code generation and logical reasoning
        "Alice: Hi Bob! How are you?",  # To analyze conversational context and response generation
        "The theory of relativity explains...",  # To evaluate scientific reasoning and coherence
        "Tell me a joke that would make even a robot laugh.",  # To assess humor generation and creativity
        "If we were both code, I'd say you're the syntax to my logic."  # Flirty and metaphorical language
    ]
    
    if not os.path.exists(activation_path):
        raise FileNotFoundError("Missing activation data file")
    
    activations = torch.load(activation_path)
    
    print(f"Loaded {len(prompts)} prompt samples with activations shape: {activations.shape}")

    # Load model
    model = load_model()
    print("Loaded sparse autoencoder model")

    # Get overall analysis
    results = analyze_activations(model, activations)
    
    print("\nGlobal Analysis Results:")
    print(f"Reconstruction Error: {results['recon_error']:.6f}")
    print(f"Sparsity (% features < 0.01): {results['sparsity']*100:.2f}%")
    
    # Find most active features
    encoded_features = results['encoded_features']
    feature_activations = encoded_features.abs().mean(dim=0)
    top_features = torch.topk(feature_activations, k=5)
    
    print("\nAnalyzing Top 5 Most Active Features:")
    for feature_idx, activation_strength in zip(top_features.indices, top_features.values):
        print(f"\nFeature {feature_idx} (avg activation: {activation_strength:.4f}):")
        
        # Analyze specific feature
        feature_results = analyze_feature(model, activations, prompts, feature_idx, top_k=5)
        
        print("Top 5 activating texts:")
        for i, result in enumerate(feature_results, 1):
            print(f"\n{i}. Activation strength: {result['activation']:.4f}")
            print(f"Text: {result['text'][:200]}...")  # Show first 200 chars

if __name__ == "__main__":
    main()