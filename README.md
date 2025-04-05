# Brain Surgery
## Part 1
## Reasoning

## Part 2
## 📄 Dataset Overview

- **Model used:** google/gemma-3-1b-it
- **Number of prompts:** 7
- **Activation files:** 7 (`activations_0.pt` through `activations_6.pt`)
- **Activation tensor shape:** `(sequence_length, hidden_size)`  
  - `sequence_length`: Number of tokens in generated text (varies per prompt)
  - `hidden_size`: 1152

- Shape of activations_0.pt: torch.Size([49, 1152])
- Shape of activations_1.pt: torch.Size([29, 1152])
- Shape of activations_2.pt: torch.Size([50, 1152])
- Shape of activations_3.pt: torch.Size([50, 1152])
- Shape of activations_4.pt: torch.Size([47, 1152])
- Shape of activations_5.pt: torch.Size([53, 1152])
- Shape of activations_6.pt: torch.Size([31, 1152])

Each activation tensor corresponds to one forward pass of the model over generated tokens, capturing the hidden states from a selected transformer layer.

---

## 🧠 Prompt Selection Strategy

The prompts were chosen to cover a **diverse range of semantic and syntactic patterns**. This diversity increases the likelihood that the autoencoder will learn **interpretable and generalizable features** from the model’s internal representations.

| Prompt | Purpose |
|--------|---------|
| **"Once upon a time in a distant land"** | Explore story structure, entities, and continuity |
| **"What is the capital of France?"** | Probe factual knowledge and memory |
| **"Write a Python function to reverse a string."** | Test code generation and reasoning |
| **"Alice: Hi Bob! How are you?"** | Evaluate conversational context and dialogue flow |
| **"The theory of relativity explains..."** | Test scientific coherence and abstraction |
| **"Tell me a joke that would make even a robot laugh."** | Assess humor and creativity |
| **"If we were both code, I'd say you're the syntax to my logic."** | Capture figurative, metaphorical expressions |

These categories allow us to test whether the autoencoder can isolate latent features related to **syntax**, **entities**, **semantic meaning**, and **context switching**.

---

## 📊 Summary Statistics

| Metric                       | Value                          |
|-----------------------------|---------------------------------|
| Total prompts               | 7                               |
| Avg. tokens per output      | ~45                             |
| Hidden size per token       | 1152                             |
| Total activation vectors    | ~315 (7 prompts × ~45 tokens)   |

> Note: Exact token counts may vary depending on the prompt and model output.

---

## 🗂 Files

- `activation_dataset/activations_*.pt` — Torch tensors for each prompt
- `activation_dataset/metadata.json` — Contains prompt, tokens, and generated text for each activation file

---

## 📌 Next Step

These activations will now be used to train a **sparse autoencoder**, with the goal of uncovering **human-interpretable features** encoded in the model’s latent space.



## Part-3

## Sparse Autoencoder for LLM Activation Analysis

### Overview
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
