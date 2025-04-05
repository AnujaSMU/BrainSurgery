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

## Summary Statistics

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



## Part 3

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

## Part 4 - Interpretable Feature Analysis

This part outlines the analysis of sparse autoencoder features and their activation patterns across various prompts. By identifying highly active features and tracing them back to the original text snippets, we attempted to interpret the "concepts" these features capture.

### Dataset & Setup

- Total Prompts Analyzed: 7
- Sparse Autoencoder: Loaded and used for dimensionality reduction
- Top 5 Most Active Features were examined
- Counterfactual clamping was performed on Feature 6024

---


### 🔹 Feature 6024

Top Activating Snippets:
- "The theory of relativity explains..."
- (Repeated across multiple activations)

Interpretation:  
This feature appears to represent scientific or academic explanations, particularly related to physics or formal knowledge-sharing. The repetition and specificity suggest it activates in the presence of structured scientific content.

Counterfactual Clamping Observations:
- When clamped, it causes strong decreases in dimensions tied to logical, factual, and structured language across all prompts.
- Affects even unrelated prompts like jokes or fairy tales, suggesting it's a dominant "science-related" concept suppressor when removed.

---

### 🔹 Feature 3519
Top Activating Snippets:
- "Write a Python function to reverse a string."
- "What is the capital of France?"

Interpretation:  
This feature likely represents direct question answering or instructional content, especially coding-related instructions or factual Q&A.

---

### 🔹 Feature 3178
Top Activating Snippets:
- "Tell me a joke that would make even a robot laugh."
- "Write a Python function to reverse a string."

Interpretation:  
This feature seems to highlight entertaining or playful prompts, possibly humor with technical undertones. The co-occurrence with a coding prompt might hint at overlap with creative tech-focused expressions.

---

### 🔹 Feature 6285
Top Activating Snippets:
- "Tell me a joke that would make even a robot laugh."
- "Once upon a time in a distant land"

Interpretation:  
This feature likely corresponds to narrative or imaginative storytelling, especially when there's a whimsical or humorous tone.

---

### 🔹 Feature 6663
Top Activating Snippets:
- "The theory of relativity explains..."
- "Alice: Hi Bob! How are you?"

Interpretation:  
Feature 6663 may activate for structured dialog or explanatory statements. The presence of both academic and conversational text suggests this feature might capture formality or coherence in sentence construction.

---

## Part 6 - Feature 6024 – Counterfactual Activation Clamping Experiment

### Goal

To prove that Feature 6024 corresponds to an interpretable concept, we clamp its activation in the encoded representation and observe the effect on downstream model behavior. This is done via **counterfactual intervention**, where we compare:

- **Baseline activations**: Normal model behavior.
- **Clamped activations**: Feature 6024 is artificially activated to a fixed high value.

---

### Setup

- The model is first trained and compressed using an autoencoder.
- We analyze activations over a fixed set of prompts and identify the most *interpretable* features.
- Feature 6024 was selected for intervention due to consistently strong activations on the prompt:

```
"The theory of relativity explains..."
```

---

### 🧪 Methodology

### Step 1: Feature Selection

We identified the top 5 most active features using `find_interpretable_features()`. Feature 6024 stood out with a high average activation and repeated strong activation on prompts related to scientific or technical language.

### Step 2: Clamping Feature 6024

The function `clamp_feature()` manually sets Feature 6024 to a constant high value (5.0) in the latent space:

```python
modified_encoded[:, feature_idx] = clamp_value
```

The autoencoder's decoder is then used to reconstruct the modified activation vector.

### Step 3: Counterfactual Comparison

For each prompt, we compare:

- **Baseline**: Original activation.
- **Clamped**: Activation with Feature 6024 forced to 5.0.

We report:
- Mean absolute difference
- Max difference
- Top 5 dimensions with largest activation shifts

---

### 🔎 Results Summary

Across a range of prompts (casual, factual, programming-related, humorous), clamping Feature 6024 consistently caused **widespread and high-magnitude activation changes**. For example:

### Sample result for prompt:
`"Write a Python function to reverse a string."`

- **Mean abs diff**: 0.1596
- **Max diff**: 2.6862
- **Top change**: Dimension 26382 decreased by 2.6862

### Sample result for prompt:
`"The theory of relativity explains..."`

- **Mean abs diff**: 0.1482
- **Max diff**: 2.6325
- **Top change**: Dimension 11406 decreased by 2.6325

### General observations:

- Top changes occurred in **consistent dimensions** across different prompts (e.g., 26382, 2190, 1038).
- Changes were mostly **negative**, suggesting the feature suppresses certain latent dimensions.
- Even non-technical prompts (e.g., jokes, casual chat) were affected—indicating that Feature 6024 represents a **dominant semantic factor**.

---

## Interpretation

- **Feature 6024** appears to be linked to **scientific or technical exposition**, based on its activation pattern—specifically its strong response to prompts like _"The theory of relativity explains..."_.
- While we did **not decode** the clamped latent space back into text, the **magnitude and consistency of activation changes** suggest that Feature 6024 has a substantial influence on the representation space.
- The widespread and systematic changes across unrelated prompts (e.g., jokes, questions, code) imply that Feature 6024 represents a **high-impact semantic dimension** within the model's internal representations.
- This supports the hypothesis that some latent features may act as **controllable knobs**, shaping the overall interpretation or direction of the model's internal processing—even without generating text.
