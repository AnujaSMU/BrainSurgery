import torch
import os
import json
from tqdm import tqdm


def merge_activations(activations_dir="activation_dataset", merge_dir='merged_activations'):
    """
    Merge all activations_*.pt files into a single tensor and save as all_activations.pt
    Returns:
        torch.Tensor: Merged activations of shape [total_tokens, activation_dim]
    """
    # Load metadata to determine files to merge
    metadata_path = os.path.join(activations_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Collect all activation tensors
    activation_list = []
    for entry in tqdm(metadata, desc="Merging activations"):
        act_path = entry["activations_path"]
        if os.path.exists(act_path):
            act = torch.load(act_path)  # Shape: [n_tokens, activation_dim]
            activation_list.append(act)
        else:
            print(f"Warning: Missing {act_path}, skipping")

    # Concatenate all activations
    all_activations = torch.cat(activation_list, dim=0)

    # Save merged tensor
    output_path = os.path.join(merge_dir, "all_activations.pt")
    torch.save(all_activations, output_path)
    print(f"Merged {len(activation_list)} activation files -> {output_path}")
    print(f"Final shape: {all_activations.shape}")

    return all_activations


if __name__ == "__main__":
    merged_activations = merge_activations()