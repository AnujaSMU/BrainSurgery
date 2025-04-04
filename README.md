# BrainSurgery Part-3

# Sparse Autoencoder for LLM Activation Analysis

## Overview
Implements a sparse autoencoder to extract interpretable features from the activations of the Gemma-3-1B language model. The autoencoder learns a sparse, higher-dimensional representation of neural activations to identify human-interpretable patterns.

### Activation Space
Activation space refers to the high-dimensional vector space where each point represents the neural activations for a given input token. The autoencoder transforms these activations into a feature space (24,576 in our implementation) where:
- Each dimension ideally corresponds to a distinct concept, as outlined in the prompts
- Representations are sparse (<5% active features per input)

### Autoencoder Architecture
```
SparseAutoencoder(
  (input_norm): LayerNorm(4096)
  (encoder): Sequential(
    (0): Linear(4096 → 24576, bias=True)
    (1): LeakyReLU(negative_slope=0.1)
    (2): Linear(24576 → 24576, bias=True)
  )
  (decoder): Sequential(...)
```
### Other Notes
We tried two different model implementations, one with just the basic definitions and hyperparameters (as outlined in the paper) and another that we had DeepSeek optimize.

We got significantly lower loss in the second implementation (`sparse_ae.pth`).

In this implementation, DeepSeek suggested we normalize the activations aggressivly, along with lowering the L1 regularization and using a learning rate schedule based on plateauing 
