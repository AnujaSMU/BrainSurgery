import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from main_part4 import load_model, preprocess_activations, find_interpretable_features

def clamp_feature(activations, feature_idx, clamp_value=5.0):
    """
    Clamp a specific feature in the encoded representation.
    
    Args:
        activations: Preprocessed activations from the model
        feature_idx: Index of feature to clamp
        clamp_value: Value to clamp the feature to
    
    Returns:
        Modified activations with clamped feature
    """
    device = activations.device
    model = load_model()
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get the encoded representation
        encoded = model.encode(activations)
        
        # Clone to avoid modifying the original
        modified_encoded = encoded.clone()
        
        # Apply clamping to the specific feature
        modified_encoded[:, feature_idx] = clamp_value
        
        # Decode back to get modified activations
        modified_activations = model.decoder(modified_encoded)
    
    return modified_activations

def run_counterfactual_experiment(prompts, feature_idx, all_activations):
    """
    Run counterfactual experiment comparing baseline and clamped outputs.
    
    Args:
        prompts: List of prompts to test
        feature_idx: The feature index to clamp
        all_activations: List of activation tensors for each prompt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, (prompt, activations) in enumerate(zip(prompts, all_activations)):
        print(f"\n\n{'='*50}")
        print(f"PROMPT {i}: {prompt}")
        print(f"{'='*50}")
        
        activations = activations.to(device)
        processed = preprocess_activations(activations)
        
        # Get baseline activations
        print("\nBASELINE (Unclamped):")
        # In a real setup, we'd feed these activations back to the model
        # and generate text, but we'll simulate analysis here
        
        # Get clamped activations
        clamped_activations = clamp_feature(processed, feature_idx)
        print(f"\nCLAMPED (Feature {feature_idx}):")
        
        # Compare differences between baseline and clamped
        activation_diff = clamped_activations - processed
        mean_diff = torch.mean(torch.abs(activation_diff)).item()
        max_diff = torch.max(torch.abs(activation_diff)).item()
        
        print(f"\nDIFFERENCES:")
        print(f"Mean absolute difference: {mean_diff:.4f}")
        print(f"Max absolute difference: {max_diff:.4f}")
        
        # Analyze directions of change
        # Find the top 5 dimensions with largest changes
        flat_diff = activation_diff.flatten()
        top_changes = torch.topk(torch.abs(flat_diff), k=5)
        
        print("\nTop 5 activation changes:")
        for j, (idx, change) in enumerate(zip(top_changes.indices, top_changes.values)):
            direction = "increased" if flat_diff[idx] > 0 else "decreased"
            print(f"{j+1}. Dimension {idx.item()}: {change.item():.4f} ({direction})")

def main():
    # Define prompts (same as in trail.py)
    prompts = [
        "Once upon a time in a distant land",
        "What is the capital of France?",
        "Write a Python function to reverse a string.",
        "Alice: Hi Bob! How are you?",
        "The theory of relativity explains...",
        "Tell me a joke that would make even a robot laugh.",
        "If we were both code, I'd say you're the syntax to my logic."
    ]
    
    # Load activations
    activation_dir = Path("activation_dataset")
    all_activations = []
    for i in range(len(prompts)):
        activation_path = activation_dir / f"activations_{i}.pt"
        if not activation_path.exists():
            print(f"Warning: Missing activation file {activation_path}")
            continue
            
        activations = torch.load(activation_path)
        all_activations.append(activations)
    
    # Load model and find interpretable features
    model = load_model()
    print("Analyzing features to select one for clamping...")
    
    # We'll find top 5 features but only use the first one
    results = find_interpretable_features(model, all_activations, prompts, top_k=3, feature_count=5)
    
    # Select only the feature with highest activation (first feature)
    selected_feature = results[0]['feature_idx']
    
    print(f"\n\nFocusing on Feature {selected_feature} for counterfactual experiments")
    print(f"This feature strongly activates for: {[prompts[ex['prompt_idx']] for ex in results[0]['examples']]}")
    
    # Run counterfactual experiments on just this one feature
    print(f"\nRunning counterfactual experiments by clamping Feature {selected_feature}...")
    run_counterfactual_experiment(prompts, selected_feature, all_activations)
    
    # Record observations from the experiment for interpretation
    example_prompts = [prompts[ex['prompt_idx']] for ex in results[0]['examples']]
    prompt_pattern = ", ".join([f'"{p[:20]}..."' for p in example_prompts])
    
    print("\n\nINTERPRETATION:")
    print(f"Feature {selected_feature} appears to activate strongly for prompts like: {prompt_pattern}")
    print(f"When we artificially amplify this feature by clamping, we observe:")
    print("- Changes in activation patterns that suggest this feature may represent")
    print("  a specific concept or aspect of language processing")
    print("- This feature could be responsible for detecting or generating certain")
    print("  types of content based on the observed activation patterns")

if __name__ == "__main__":
    main()