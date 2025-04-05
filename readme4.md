# Interpretable Feature Analysis of LLM Activations

This README outlines the analysis of sparse autoencoder features and their activation patterns across various prompts. By identifying highly active features and tracing them back to the original text snippets, we attempted to interpret the "concepts" these features capture.

## Dataset & Setup

- *Total Prompts Analyzed*: 7
- *Sparse Autoencoder*: Loaded and used for dimensionality reduction
- *Top 5 Most Active Features* were examined
- Counterfactual clamping was performed on Feature 6024

---

## Feature Interpretations

### 🔹 Feature 6024
*Top Activating Snippets:*
- "The theory of relativity explains..."
- (Repeated across multiple activations)

*Interpretation:*  
This feature appears to represent *scientific or academic explanations*, particularly related to physics or formal knowledge-sharing. The repetition and specificity suggest it activates in the presence of structured scientific content.

*Counterfactual Clamping Observations:*
- When clamped, it causes strong decreases in dimensions tied to logical, factual, and structured language across all prompts.
- Affects even unrelated prompts like jokes or fairy tales, suggesting it's a dominant "science-related" concept suppressor when removed.

---

### 🔹 Feature 3519
*Top Activating Snippets:*
- "Write a Python function to reverse a string."
- "What is the capital of France?"

*Interpretation:*  
This feature likely represents *direct question answering or instructional content*, especially coding-related instructions or factual Q&A.

---

### 🔹 Feature 3178
*Top Activating Snippets:*
- "Tell me a joke that would make even a robot laugh."
- "Write a Python function to reverse a string."

*Interpretation:*  
This feature seems to highlight *entertaining or playful prompts*, possibly humor with technical undertones. The co-occurrence with a coding prompt might hint at overlap with creative tech-focused expressions.

---

### 🔹 Feature 6285
*Top Activating Snippets:*
- "Tell me a joke that would make even a robot laugh."
- "Once upon a time in a distant land"

*Interpretation:*  
This feature likely corresponds to *narrative or imaginative storytelling*, especially when there's a whimsical or humorous tone.

---

### 🔹 Feature 6663
*Top Activating Snippets:*
- "The theory of relativity explains..."
- "Alice: Hi Bob! How are you?"

*Interpretation:*  
Feature 6663 may activate for *structured dialog or explanatory statements*. The presence of both academic and conversational text suggests this feature might capture formality or coherence in sentence construction.

---
