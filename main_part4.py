import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

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

def preprocess_activations(activations):
    """Normalize and preprocess activations"""
    activations = activations.float()  # Ensure float32
    # Normalize activations
    activations = (activations - activations.mean()) / (activations.std() + 1e-8)
    activations = activations / 3.0
    activations = torch.clamp(activations, min=-3.0, max=3.0)
    return activations

def find_interpretable_features(model, all_activations, prompts, top_k=5, feature_count=10):
    """Find and analyze the most interpretable features"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Process all activations through the encoder
    all_encoded = []
    processed_activations = []
    prompt_indices = []
    
    with torch.no_grad():
        for i, activations in enumerate(all_activations):
            activations = activations.to(device)
            processed = preprocess_activations(activations)
            processed_activations.append(processed)
            
            # Encode each activation set
            encoded = model.encode(processed)
            all_encoded.append(encoded)
            
            # Keep track of which prompt each activation belongs to
            prompt_indices.extend([i] * len(encoded))
    
    # Combine all encodings
    all_encoded = torch.cat(all_encoded, dim=0)
    prompt_indices = torch.tensor(prompt_indices, device=device)
    
    # Find features with highest average activations across all samples
    feature_strengths = all_encoded.abs().mean(dim=0)
    top_features = torch.topk(feature_strengths, k=feature_count)
    
    print(f"\nAnalyzing Top {feature_count} Most Active Features:")
    
    results = []
    for rank, (feature_idx, activation_strength) in enumerate(zip(top_features.indices, top_features.values)):
        print(f"\n### Feature {feature_idx} (avg activation: {activation_strength:.4f}) ###")
        
        # Get activations for this feature across all samples
        feature_activations = all_encoded[:, feature_idx]
        
        # Find top activating examples
        top_indices = torch.topk(feature_activations, k=min(top_k, len(feature_activations)))
        
        feature_result = {
            'feature_idx': feature_idx.item(),
            'avg_activation': activation_strength.item(),
            'rank': rank,
            'examples': []
        }
        
        print("Top activating prompts:")
        for i, (idx, strength) in enumerate(zip(top_indices.indices, top_indices.values)):
            prompt_idx = prompt_indices[idx].item()
            prompt_text = prompts[prompt_idx]
            
            example = {
                'prompt_idx': prompt_idx,
                'activation': strength.item(),
                'prompt': prompt_text
            }
            feature_result['examples'].append(example)
            
            print(f"\n{i+1}. Prompt {prompt_idx}: Activation strength: {strength:.4f}")
            print(f"Text: {prompt_text}")
        
        results.append(feature_result)
    
    return results

def main():
    # Define prompts 
    prompts = [
        "Once upon a time in a distant land",  # To explore story structure, named entities, and sequence continuity
        "What is the capital of France?",  # To probe knowledge recall and factual associations
        "Write a Python function to reverse a string.",  # To test code generation and logical reasoning
        "Alice: Hi Bob! How are you?",  # To analyze conversational context and response generation
        "The theory of relativity explains...",  # To evaluate scientific reasoning and coherence
        "Tell me a joke that would make even a robot laugh.",  # To assess humor generation and creativity
        "If we were both code, I'd say you're the syntax to my logic."  # Flirty and metaphorical language
    ]
    
    # Load all activation files
    activation_dir = Path("activation_dataset")
    if not activation_dir.exists():
        raise FileNotFoundError(f"Directory not found: {activation_dir}")
    
    all_activations = []
    for i in range(len(prompts)):
        activation_path = activation_dir / f"activations_{i}.pt"
        if not activation_path.exists():
            print(f"Warning: Missing activation file {activation_path}")
            continue
            
        activations = torch.load(activation_path)
        all_activations.append(activations)
    
    if not all_activations:
        raise FileNotFoundError("No activation files found")
    
    print(f"Loaded {len(all_activations)} activation files for {len(prompts)} prompts")
    
    # Load model
    model = load_model()
    print("Loaded sparse autoencoder model")
    
    # Find interpretable features
    results = find_interpretable_features(model, all_activations, prompts, top_k=3, feature_count=5)
    
    print("\n=====================================")
    
    for result in results:
        feature_idx = result['feature_idx']
        print(f"\nFeature {feature_idx} (Rank {result['rank']+1}, Avg activation: {result['avg_activation']:.4f}):")
        
        # Show top examples
        examples = result['examples']
        common_prompts = set(ex['prompt_idx'] for ex in examples)
        
        print("Activates strongly for prompt types:")
        for prompt_idx in common_prompts:
            print(f"- Prompt {prompt_idx}: {prompts[prompt_idx]}")

if __name__ == "__main__":
    main()